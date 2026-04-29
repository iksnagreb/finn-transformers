import os

# Disable ONNX Runtime CPU affinity to prevent pthread errors on ARM systems
os.environ['ORT_DISABLE_CPU_AFFINITY'] = '1'

import tensorrt as trt
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import time
import json
import torch
import onnx
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import pycuda.driver as cuda
import gc
import yaml
from onnxconverter_common import float16 
import onnxruntime as ort
import dvc.api
from vision.model import Model
from measure.latency_throughput_log import latency_throughput
from dvclive import Live
from torchvision import datasets, transforms
from language.model_wrapper import ModelTopKWrapper


FP16 = os.environ.get("FP16", "0") == "1"
GPU_MEM_LIMIT_GB = float(os.environ.get("GPU_MEM_LIMIT_GB", "2.0"))
GPU_MEM_LIMIT_BYTES = int(GPU_MEM_LIMIT_GB * 1024 * 1024 * 1024)

MODEL_TYPE = os.environ.get("MODEL_TYPE", "vision")
if MODEL_TYPE != "radioml" and MODEL_TYPE != "language" and MODEL_TYPE != "vision":
    MODEL_TYPE = "vision"
    print("Defaulting Model Type to vision model.")

# look up quantisation type from params
with open(f"{MODEL_TYPE}/params.yaml", "r") as f:
    cfg = yaml.safe_load(f)

bits = cfg["model"]["embedding"].get("bits", 0)
INT8 = (bits == 8)

if INT8:
    dtype = torch.int8
    print("INT8 enabled")
elif FP16:
    dtype = torch.float16
    print("FP16 enabled")
else:
    dtype = torch.float32
    print("FP32")

print(f"GPU memory budget: {GPU_MEM_LIMIT_GB:.2f} GB ({GPU_MEM_LIMIT_BYTES} bytes)")

RADIOML_PATH = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.hdf5"
RADIOML_PATH_NPZ = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"
CIFAR10_ROOT = R"/data/gitlab/cifar-10-batches-py"
CIFAR10_PATH_NPZ = R"/data/gitlab/cifar-10-batches-py/cifar10.npz"
LANG_PATH_NPZ = R"/data/gitlab/language.npz"

if MODEL_TYPE == "radioml":
    DATA_PATH_NPZ = RADIOML_PATH_NPZ
if MODEL_TYPE == "vision":
    DATA_PATH_NPZ = CIFAR10_PATH_NPZ
if MODEL_TYPE == "language":
    DATA_PATH_NPZ = LANG_PATH_NPZ

def to_device(data,device):
    if isinstance(data, (list,tuple)): 
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking=True)

class DeviceDataLoader():
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
    
    def __len__(self):
        return len(self.dl)

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def save_json(log, filepath):
    filepath = Path(filepath)
    filepath.parent.parent.mkdir(parents=True, exist_ok=True)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(log, f, indent=4)


def parse_shape(shape, batch_value):
    """Ersetzt 'batch_size' durch batch_value in der shape-Liste."""
    resolved = []
    for i, d in enumerate(shape):
        if isinstance(d, str):
            if d == "batch_size" or i == 0:
                resolved.append(batch_value)
            elif d == "sequence_length":
                resolved.append(128)
            elif d == "Muloutput_dim_2":
                resolved.append(64)
            elif d == "channels":
                resolved.append(3)
            else:
                resolved.append(1)
        elif d is None:
            resolved.append(batch_value if i == 0 else 1)
        else:
            resolved.append(int(d))

    return tuple(resolved)


ONNX_TO_TORCH_DTYPE = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(double)": torch.float64,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
    "tensor(uint8)": torch.uint8,
    "tensor(int8)": torch.int8,
    "tensor(bool)": torch.bool,
}

# ONNX proto dtype to torch dtype mapping (for integer dtype values)
ONNX_PROTO_DTYPE_TO_TORCH = {
    1: torch.float32,    # FLOAT
    2: torch.uint8,      # UINT8
    3: torch.int8,       # INT8
    5: torch.int32,      # INT32
    6: torch.int64,      # INT64
    7: torch.int64,      # INT64 (alternate)
    10: torch.float16,   # FLOAT16
    12: torch.float64,   # DOUBLE
}


def onnx_dtype_to_torch(onnx_dtype):
    """
    Convert ONNX dtype (int or string) to torch dtype.
    Handles both string format ("tensor(float)") and integer format (1, 7, etc.)
    """
    if isinstance(onnx_dtype, int):
        # Integer dtype from ONNX proto
        return ONNX_PROTO_DTYPE_TO_TORCH.get(onnx_dtype, torch.float32)
    else:
        # String format
        return ONNX_TO_TORCH_DTYPE.get(onnx_dtype, torch.float32) 


# def get_model_io_info(model_path):
#     """
#     Liest Input- und Output-Infos aus einem ONNX-Modell.
#     Gibt Listen von Dictionaries mit Name, Shape und Dtype zurück.
#     """
#     sess_options = ort.SessionOptions()

#     sess_options.intra_op_num_threads = 8
#     session = ort.InferenceSession(model_path, sess_options)    
#     input_info = [
#         {
#             "name": inp.name,
#             "shape": inp.shape,
#             "dtype": inp.type
#         }
#         for inp in session.get_inputs()
#     ]
#     output_info = [
#         {
#             "name": out.name,
#             "shape": out.shape,
#             "dtype": out.type
#         }
#         for out in session.get_outputs()
#     ]
#     return input_info, output_info

def get_model_io_info(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    inputs = []
    for inp in graph.input:
        shape = [
            d.dim_value if (d.dim_value > 0) else None
            for d in inp.type.tensor_type.shape.dim
        ]
        inputs.append({
            "name": inp.name,
            "shape": shape,
            "dtype": inp.type.tensor_type.elem_type
        })

    outputs = []
    for out in graph.output:
        shape = [
            d.dim_value if (d.dim_value > 0) else None
            for d in out.type.tensor_type.shape.dim
        ]
        outputs.append({
            "name": out.name,
            "shape": shape,
            "dtype": out.type.tensor_type.elem_type
        })

    return inputs, outputs

def print_latency(latency_ms, latency_synchronize, latency_datatransfer, end_time, start_time, num_batches, throughput_batches, throughput_images, batch_size):
    print("For Batch Size: ", batch_size)
    print(f"Gemessene durchschnittliche Latenz für Inference : {latency_ms:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Synchronisation : {latency_synchronize:.4f} ms")
    print(f"Gemessene durchschnittliche Latenz mit Datentransfer : {latency_datatransfer:.4f} ms")
    print(f"Gesamtzeit: {end_time-start_time:.4f} s")
    print("num_batches", num_batches)
    print(f"Throughput: {throughput_batches:.4f} Batches/Sekunde")
    print(f"Throughput: {throughput_images:.4f} Bilder/Sekunde")


def create_test_dataloader(DATA_PATH_NPZ, batch_size, onnx_model_path):
    """
    Erstellt den DataLoader für die Testdaten.
    :param RADIOML_PATH: Pfad zur Testdaten-Datei.
    :param batch_size: Die Batchgröße.
    :return: DataLoader-Objekt für die Testdaten.
    """
    data = np.load(DATA_PATH_NPZ)
    input_info, output_info = get_model_io_info(onnx_model_path)
    key_list = list(data.keys())
    print("Keys in NPZ file:", key_list)
    if len(input_info) == 2:
        input_key = key_list[0]
        attention_mask_key = key_list[1]
        output_key = key_list[2]
    elif len(input_info) == 3:
        input_key = key_list[0]
        attention_mask_key = key_list[1]
        output_key = key_list[2]
    else:   # nur 1 input
        input_key = key_list[0]
        attention_mask_key = None
        output_key = key_list[1]

    input_ids = torch.from_numpy(data[input_key])


    attention_mask = torch.from_numpy(data[attention_mask_key]) if attention_mask_key else None
    labels = torch.from_numpy(data[output_key])

    # Nur das erste Sample auswählen
    if len(input_info) > 1:
        test_dataset = TensorDataset(input_ids, attention_mask, labels)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
    else:
        test_dataset = TensorDataset(input_ids, labels)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
    return test_loader

def test_data(context, batch_size, input_info, output_info):
    device_inputs = {}
    device_outputs = {}
    device_attention_masks = {}
    device_token_type = {}
    torch_stream = torch.cuda.Stream()
    stream_ptr = torch_stream.cuda_stream

    inp = input_info[0]
    name = inp["name"]
    shape = parse_shape(inp["shape"], batch_size)
    dtype = onnx_dtype_to_torch(inp["dtype"])  
    dtype_out = onnx_dtype_to_torch(output_info[0]["dtype"]) 
    tensor = torch.empty(shape, dtype=dtype, device='cuda')
    context.set_tensor_address(name, tensor.data_ptr())
    context.set_input_shape(name, shape)
    device_inputs[name] = tensor
    
    if len(input_info) > 1:
        att_mask_name = input_info[1]["name"]
        att_mask_shape = parse_shape(input_info[1]["shape"], batch_size)
        att_mask_dtype = onnx_dtype_to_torch(input_info[1]["dtype"])
        att_mask_tensor = torch.empty(att_mask_shape, dtype=att_mask_dtype, device='cuda')
        context.set_tensor_address(att_mask_name, att_mask_tensor.data_ptr())
        context.set_input_shape(att_mask_name, att_mask_shape)

        device_attention_masks[att_mask_name] = att_mask_tensor

    if len(input_info) > 2:
        token_type_name = input_info[2]["name"]
        token_type_shape = parse_shape(input_info[2]["shape"], batch_size)
        token_type_dtype = onnx_dtype_to_torch(input_info[2]["dtype"])
        token_type_tensor = torch.empty(token_type_shape, dtype=token_type_dtype, device='cuda')
        context.set_tensor_address(token_type_name, token_type_tensor.data_ptr())
        context.set_input_shape(token_type_name, token_type_shape)

        device_token_type[token_type_name] = token_type_tensor

    for out in output_info:
        name = out["name"]
        shape = parse_shape(out["shape"], batch_size)
        # Use the correct dtype for each output, not just the first output's dtype
        out_dtype = onnx_dtype_to_torch(out["dtype"])
        tensor = torch.empty(shape, dtype=out_dtype, device='cuda')
        context.set_tensor_address(name, tensor.data_ptr())
        device_outputs[name] = tensor
        print(f"  Output '{name}': shape {shape}, dtype {out_dtype}, address {tensor.data_ptr()}")

    device_input = next(iter(device_inputs.values()))
    
    # For language models, use the top_indices output specifically
    if MODEL_TYPE == "language" and "462" in device_outputs:
        device_output = device_outputs["462"]
    else:
        device_output = next(iter(device_outputs.values()))
    
    if len(input_info) > 1:
        device_attention_mask = next(iter(device_attention_masks.values()))
    else:
        device_attention_mask = None

    if len(input_info) > 2:
        device_token_type = next(iter(device_token_type.values()))
    else:
        device_token_type = None

    return device_input, device_output, device_attention_mask, device_token_type, stream_ptr, torch_stream, device_outputs


def build_tensorrt_engine(onnx_model_path, test_loader, batch_size, input_info=None, min_bs=1, opt_bs=8, max_bs=1024):
    """
    Erstellt und gibt die TensorRT-Engine und den Kontext zurück.
    :param onnx_model_path: Pfad zur ONNX-Modell-Datei.
    :param logger: TensorRT-Logger.
    :return: TensorRT-Engine und Execution Context.
    """
    if INT8:
        min_bs = batch_size
        opt_bs = batch_size
        max_bs = batch_size
        
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_model_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX Parsing failed")

    config = builder.create_builder_config()
    print("config created")
    
    # DLA configuration for INT8 - disabled by default due to 16 loadables limit on Jetson
    # Language models with transformers/attention are poorly suited for DLA due to many unsupported ops
    USE_DLA = os.environ.get("USE_DLA", "0") == "1"
    if INT8 and USE_DLA and MODEL_TYPE != "language":
        config.default_device_type = trt.DeviceType.DLA
        print("use dla")
        config.DLA_core = 0  # 0 oder 1
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        print("fallback: gpu")
        # DLA doesn't need optimization profile
    else:
        # GPU mode - needs optimization profile for dynamic shapes
        # (or fallback for models unsuitable for DLA like transformers)
        config.default_device_type = trt.DeviceType.GPU
        print("config.default_device_type = trt.DeviceType.GPU")
        
        profile = builder.create_optimization_profile()
        for inp in input_info:
            name = inp["name"]
            shape = inp["shape"]
            min_shape = parse_shape(shape, min_bs)
            opt_shape = parse_shape(shape, opt_bs)
            max_shape = parse_shape(shape, max_bs)
            profile.set_shape(name, min_shape, opt_shape, max_shape)

        config.add_optimization_profile(profile)

        print(f"Optimization profile for input '{name}':")
        print(f"  min_shape: {min_shape}")
        print(f"  opt_shape: {opt_shape}")
        print(f"  max_shape: {max_shape}")

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GPU_MEM_LIMIT_BYTES)

    if FP16 == True:
        config.set_flag(trt.BuilderFlag.FP16)
    if INT8 == True: 
        config.set_flag(trt.BuilderFlag.INT8)
        print("int 8 builder flag gesetzt")

    print("Serialized engine: ")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Fehler beim Bauen der TensorRT-Engine: serialized_engine ist None.")

    print("Logger:")
    runtime = trt.Runtime(logger)
    print("engine ")
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")
    
    print("context: ")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create execution context. If using DLA, you may have hit the 16 loadables per core limit. Try setting USE_DLA=0 environment variable.")
    print("after context:")
    return engine, context


def run_inference(context, test_loader, device_input, device_output, device_attention_mask, device_token_type, stream_ptr, torch_stream, batch_size=1, input_info=None, output_info=None, accuracy_flag=False, device_outputs=None):
    """
    Funktion zur Bestimmung der Inferenzlatenz.
    """
    total_time = 0
    total_time_synchronize = 0
    total_time_datatransfer = 0  
    iterations = 0 

    total_predictions = 0
    correct_predictions = 0
    do_prints=True

    for batch in test_loader: 
        if len(batch) == 2:
            xb, yb = batch
            att_mask = None
            token_type = None
        elif len(batch) == 3:
            xb, att_mask, yb = batch
            token_type = None
        elif len(batch) == 4:
            xb, att_mask, yb, token_type = batch
        else:
            raise ValueError("Unerwartete Batch-Größe!", len(batch))
        
        start_time_datatransfer = time.time()  

        dtype = onnx_dtype_to_torch(input_info[0]["dtype"])

        input_name = input_info[0]["name"]

       

        device_input.copy_(xb.to(dtype))
        context.set_tensor_address(input_name, device_input.data_ptr())
        context.set_input_shape(input_name, device_input.shape)

        if att_mask is not None:
            att_mask_name = input_info[1]["name"]
            device_attention_mask.copy_(att_mask.to(dtype))
            context.set_tensor_address(att_mask_name, device_attention_mask.data_ptr())
            context.set_input_shape(att_mask_name, device_attention_mask.shape)
        
        if token_type is not None:
            token_type_name = input_info[2]["name"]
            device_token_type.copy_(token_type.to(dtype))
            context.set_tensor_address(token_type_name, device_token_type.data_ptr())
            context.set_input_shape(token_type_name, device_token_type.shape)

        # For language models, set tensor address for top_indices output
        if MODEL_TYPE == "language" and "462" in device_outputs:
            output_name = "462"
        else:
            output_name = output_info[0]["name"]
        
        # Set addresses for ALL outputs
        for out_info in output_info:
            out_name = out_info["name"]
            if out_name in device_outputs:
                addr = device_outputs[out_name].data_ptr()
                if iterations == 0:
                    print(f"DEBUG: Setting output '{out_name}' address to {addr}, shape {device_outputs[out_name].shape}, dtype {device_outputs[out_name].dtype}")
                context.set_tensor_address(out_name, addr)
        
        # Also set addresses for any other outputs to prevent memory errors
        for out_info in output_info:
            out_name = out_info["name"]
            if out_name in device_outputs and out_name != output_name:
                context.set_tensor_address(out_name, device_outputs[out_name].data_ptr()) 

        
        # torch_stream.synchronize()

        start_time_synchronize = time.time()  
        torch_stream.synchronize()  

        start_time_inference = time.time() 
        try:
            with torch.cuda.stream(torch_stream):
                context.execute_async_v3(stream_ptr)
        except Exception as e:
            print("TensorRT Error:", e)
        torch_stream.synchronize() 
    
        end_time = time.time()

        # Extract outputs - handle both TopK wrapper (2 outputs) and simple model (1 output)
        topk_indices = None
        topk_values = None
        
        if MODEL_TYPE == "language" and len(device_outputs) > 1:
            # TopK wrapper has 2 outputs: values (float32) and indices (int64)
            # ONNX uses numeric names like '461' and '462', so extract by dtype
            outputs_list = list(device_outputs.items())
            
            # Find indices output (int64) and values output (float32)
            for name, tensor in outputs_list:
                tensor_np = tensor.cpu().numpy()
                if tensor_np.dtype == np.int64:
                    topk_indices = tensor_np  # [batch, k]
                elif tensor_np.dtype == np.float32:
                    topk_values = tensor_np   # [batch, k]
            
            output = topk_indices if topk_indices is not None else next(iter(device_outputs.values())).cpu().numpy()
        else:
            # Simple model: single output with full logits
            output = next(iter(device_outputs.values())).cpu().numpy()

        end_time_datatransfer = time.time() 
        
        latency = end_time - start_time_inference  
        latency_synchronize = end_time - start_time_synchronize  
        latency_datatransfer = end_time_datatransfer - start_time_datatransfer  

        total_time += latency
        total_time_synchronize += latency_synchronize
        total_time_datatransfer += latency_datatransfer
        iterations += 1

        if accuracy_flag:
            if MODEL_TYPE == "language":
                labels = yb.numpy()  # [batch, seq_len]
                
                if topk_indices is not None:
                    # TopK wrapper: indices are [batch, 5] token IDs for the LAST token
                    # Extract the label for the last token to compare
                    pred = topk_indices  # [batch, 5]
                    
                    # Get labels for last position (what we're predicting)
                    if labels.ndim == 2:
                        # [batch, seq_len] - get last position
                        last_token_labels = labels[:, -1]  # [batch]
                    else:
                        # Already [batch]
                        last_token_labels = labels
                    
                    labels_expanded = last_token_labels.reshape(-1, 1)  # [batch, 1]
                    matches = (pred == labels_expanded).any(axis=1)  # Check if label in top-5
                    
                    correct = matches.sum()
                    total = len(matches)
                else:
                    # Simple model: full logits [batch, seq_len, vocab_size]
                    # Use argmax to get predictions
                    pred = output.argmax(axis=-1).astype(np.int64)  # [batch, seq_len]
                    
                    # Filter out padding tokens (label == -100)
                    mask = labels != -100
                    valid_preds = pred[mask]
                    valid_labels = labels[mask]
                    
                    correct = (valid_preds == valid_labels).sum()
                    total = len(valid_preds) if len(valid_preds) > 0 else 1
                
                correct_predictions += correct
                total_predictions += total
            else:
                pred = output.argmax(axis=-1)
                correct = (pred == yb.numpy()).sum()
                total = yb.shape[0]
                correct_predictions += correct
                total_predictions += total

        if accuracy_flag and do_prints==True:
            if MODEL_TYPE == "language":
                # TopK wrapper: print the predictions and last token label
                print("=" * 60)
                print("TopK Wrapper Output:")
                print("=" * 60)
                print("Prediction (Top-5 indices): ", output[0])  # [batch, 5]
                print("Output dtype: ", output.dtype)
                print("Output shape: ", output.shape)
                # Print the LAST token label (what we're predicting)
                last_label = yb.numpy()[0, -1] if yb.numpy().ndim == 2 else yb.numpy()[0]
                print(f"True label (last token): {last_label}")
                print(f"Match in top-5: {last_label in output[0]}")
            else:
                # Simple model: print predictions
                print("=" * 60)
                print("Simple Model Output:")
                print("=" * 60)
                print("Output dtype: ", output.dtype)
                print("Output shape: ", output.shape)
                
                # Handle both language sequences and vision scalars
                label_val = yb.numpy()
                # Vision: scalar label
                print("Prediction (logits): ", output[0])
                print("True label: ", label_val[0] if label_val.ndim == 1 else label_val)
                if hasattr(pred, '__len__'):
                    print("Predicted class: ", pred[0])
                else:
                    print("Predicted class: ", pred)
            do_prints = False

    accuracy = 0
    if accuracy_flag:
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    average_latency = (total_time / iterations) * 1000  
    average_latency_synchronize = (total_time_synchronize / iterations) * 1000 
    average_latency_datatransfer = (total_time_datatransfer / iterations) * 1000  

    del context
    torch_stream.synchronize()
    del torch_stream


    return average_latency, average_latency_synchronize, average_latency_datatransfer, accuracy


def calculate_latency_and_throughput(batch_sizes, onnx_model_path, input_info, output_info):
    """
    Berechnet die durchschnittliche Latenz und den Durchsatz (Bilder und Batches pro Sekunde) für verschiedene Batchgrößen.
    :param context: TensorRT-Execution-Context.
    :param test_loader: DataLoader mit Testdaten.
    :param device_input: Eingabebuffer auf der GPU.
    :param device_output: Ausgabebuffer auf der GPU.
    :param stream_ptr: CUDA-Stream-Pointer.
    :param torch_stream: PyTorch CUDA-Stream.
    :param batch_sizes: Liste der Batchgrößen.
    :return: (Throughput-Log, Latenz-Log).
    """
    

    throughput_log = []
    latency_log = []
    latency_log_batch = []

    for batch_size in batch_sizes:
        print("Measuring for batch size:", batch_size)
        current_onnx_model_path = onnx_model_path
        if INT8:
            current_onnx_model_path = f"outputs/{MODEL_TYPE}/model_brevitas_{batch_size}_simple.onnx"
            print(f"Using INT8 simple model for batch size {batch_size}: {current_onnx_model_path}")

        input_info, output_info = get_model_io_info(current_onnx_model_path)

        test_loader = create_test_dataloader(DATA_PATH_NPZ, batch_size, current_onnx_model_path)
        engine, context = build_tensorrt_engine(current_onnx_model_path, test_loader, batch_size, input_info)
        device_input, device_output, device_attention_mask, device_token_type, stream_ptr, torch_stream, device_outputs = test_data(context, batch_size, input_info, output_info)

        
        # for the average
        latency_ms_sum = 0
        latency_synchronize_sum = 0
        lantency_datatransfer_sum = 0
        total_time_sum = 0
        num_executions = 1
        for i in range(num_executions):
            start_time = time.time()
            latency_ms, latency_synchronize, latency_datatransfer, _ = run_inference(
                context=context,
                test_loader=test_loader,
                device_input=device_input,
                device_output=device_output,
                device_attention_mask=device_attention_mask,
                device_token_type=device_token_type,
                stream_ptr=stream_ptr,
                torch_stream=torch_stream,
                batch_size=batch_size,
                input_info=input_info,
                output_info=output_info,
                device_outputs=device_outputs
            )
            latency_ms_sum = latency_ms_sum + latency_ms
            latency_synchronize_sum = latency_synchronize_sum + (latency_synchronize-latency_ms)
            lantency_datatransfer_sum = lantency_datatransfer_sum + (latency_datatransfer-latency_synchronize)

            end_time = time.time()
            total_time_sum = total_time_sum + (end_time-start_time)


        latency_avg = float(latency_ms_sum/num_executions)
        latency_synchronize_avg = float(latency_synchronize_sum/num_executions)
        latency_datatransfer_avg = float(lantency_datatransfer_sum/num_executions)
        total_time_avg = float(total_time_sum/num_executions)

        num_batches = int(7600/batch_size) 
        print("old num batches:", num_batches)
        num_batches = len(test_loader)
        print("new num batches:", num_batches)
        throughput_batches = num_batches/(total_time_avg) 
        throughput_images = (num_batches*batch_size)/(total_time_avg)


        log_latency_inference = {"batch_size": batch_size, "type":"inference", "value": latency_avg/batch_size} # pro datensatz?
        log_latency_synchronize = {"batch_size": batch_size, "type":"synchronize", "value": (latency_synchronize_avg/batch_size)} # pro datensatz?
        log_latency_datatransfer = {"batch_size": batch_size, "type":"datatransfer", "value": (latency_datatransfer_avg/batch_size)} # pro datensatz?
        log_latency_inference_batch = {"batch_size": batch_size, "type":"inference", "value": latency_avg} #pro batch
        log_latency_synchronize_batch = {"batch_size": batch_size, "type":"synchronize", "value": (latency_synchronize_avg)} #pro batch
        log_latency_datatransfer_batch = {"batch_size": batch_size, "type":"datatransfer", "value": (latency_datatransfer_avg)} #pro batch 
        throughput = {"batch_size": batch_size, "throughput_images_per_s": throughput_images, "throughput_batches_per_s": throughput_batches}


        throughput_log.append(throughput)
        latency_log.extend([log_latency_inference, log_latency_synchronize, log_latency_datatransfer])
        latency_log_batch.extend([log_latency_inference_batch, log_latency_synchronize_batch, log_latency_datatransfer_batch])
        print_latency(latency_avg, latency_synchronize_avg+latency_avg, latency_datatransfer_avg+latency_synchronize_avg+latency_avg, end_time, start_time, num_batches, throughput_batches, throughput_images, batch_size)

        # Clean up TensorRT resources to avoid DLA loadables limit
        torch_stream.synchronize()
        del device_input
        del device_output
        if device_attention_mask is not None:
            del device_attention_mask
        if device_token_type is not None:
            del device_token_type
        del context
        del engine
        torch.cuda.empty_cache()
        print(f"Cleaned up resources for batch size {batch_size}")

    return throughput_log, latency_log, latency_log_batch


def run_accuracy_eval(batch_size, input_info, output_info, DATA_PATH_NPZ, onnx_model_path):
    print("batch_size", batch_size)
    print(" input_info", input_info)
    print("output_info", output_info)
    print("DATA_PATH_NPZ", DATA_PATH_NPZ)
    print("onnx_model_path for accuracy validation", onnx_model_path)
    input_info, output_info = get_model_io_info(onnx_model_path)
    test_loader = create_test_dataloader(DATA_PATH_NPZ, 1, onnx_model_path)
    engine, context = build_tensorrt_engine(onnx_model_path, test_loader, 1, input_info)
    device_input, device_output, device_attention_mask, device_token_type, stream_ptr, torch_stream, device_outputs = test_data(context, 1, input_info, output_info)
    
    _, _, _, accuracy = run_inference(
                context=context,
                test_loader=test_loader,
                device_input=device_input,
                device_output=device_output,
                device_attention_mask=device_attention_mask,
                device_token_type=device_token_type,
                stream_ptr=stream_ptr,
                torch_stream=torch_stream,
                batch_size=batch_size,
                input_info=input_info,
                output_info=output_info,
                accuracy_flag=True,
                device_outputs=device_outputs
            )
    
    # Clean up resources
    torch_stream.synchronize()
    del device_input, device_output
    if device_attention_mask is not None:
        del device_attention_mask
    if device_token_type is not None:
        del device_token_type
    del context, engine
    torch.cuda.empty_cache()
    
    return accuracy


if __name__ == "__main__":

    if (MODEL_TYPE == "language") or (MODEL_TYPE == "vision"):
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    else:
        # Vision and RadioML can handle larger batches
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    onnx_model_path = f"outputs/{MODEL_TYPE}/model_dynamic_batchsize.onnx"

    if INT8:
        if MODEL_TYPE == "language":
            # Try to use TopK wrapper model first (reduced data transfer 210x)
            topk_model_path = f"outputs/{MODEL_TYPE}/model_topk_5.onnx"
            if Path(topk_model_path).exists():
                onnx_model_path = topk_model_path
                print("Using TopK wrapper model for language INT8")
            else:
                # Fallback to simple model if TopK not available
                onnx_model_path = f"outputs/{MODEL_TYPE}/model_brevitas_1_simple.onnx"
                print(f"TopK model not found, using simple model instead")
        else:
            # For vision/radioml: use simple model
            onnx_model_path = f"outputs/{MODEL_TYPE}/model_brevitas_1_simple.onnx"

    model = onnx.load(onnx_model_path)

    input_info, output_info = get_model_io_info(onnx_model_path)

    batch_size = 1
    accuracy = run_accuracy_eval(batch_size, input_info, output_info, DATA_PATH_NPZ, onnx_model_path)
    print(f"Accuracy : {accuracy:.2%}")

    if FP16:
        accuracy_path = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "eval_results" /"accuracy_FP16.json"
        quantisation_type = "FP16"
    elif INT8: 
        accuracy_path = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "eval_results" /"accuracy_INT8.json"
        quantisation_type = "INT8"
    else:
        accuracy_path = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "eval_results" /"accuracy_FP32.json"
        quantisation_type = "FP32"

 
    accuracy_result = {
        "quantisation_type": quantisation_type,
        "value": accuracy
    }
    save_json(accuracy_result, accuracy_path)
    


    throughput_log, latency_log, latency_log_batch = calculate_latency_and_throughput(batch_sizes, onnx_model_path, input_info=input_info, output_info=output_info)
    if FP16:
        throughput_results = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / "FP16" / "throughput_results.json"
        latency_results = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / "FP16"/ "latency_results.json"
        latency_results_batch = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / "FP16"/ "latency_results_batch.json"
        latency_throughput_path = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / "FP16"/ "latency_throughput.json"
    elif INT8:
        throughput_results = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / "INT8" / "throughput_results.json"
        latency_results = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / "INT8"/ "latency_results.json"
        latency_results_batch = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / "INT8"/ "latency_results_batch.json"
        latency_throughput_path = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / "INT8"/ "latency_throughput.json"
    else:
        throughput_results = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / "FP32" / "throughput_results.json"
        os.makedirs(os.path.dirname(throughput_results), exist_ok=True)
        latency_results_batch = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / "FP32"/ "latency_results_batch.json"
        latency_throughput_path = Path(__file__).resolve().parent.parent / "outputs" / MODEL_TYPE / "plot" / "FP32"/ "latency_throughput.json"

    save_json(throughput_log, throughput_results)
    save_json(latency_log_batch, latency_results_batch)

    latency_throughput(latency_results_batch, throughput_results, latency_throughput_path) 

    with Live(save_dvc_exp=True, report="md") as live:
        print("Start DVC Live Report....", flush=True)
        print("throughput result: ")
        print(throughput_results)
        live.log_artifact(throughput_results, name=f"throughput_results_{quantisation_type}_{MODEL_TYPE}")
        print("latency batch result:")
        print(latency_results_batch)
        live.log_artifact(latency_results_batch, name=f"latency_results_batch_{quantisation_type}_{MODEL_TYPE}")
        print("latency throughput result: ")
        print(latency_throughput_path)
        live.log_artifact(latency_throughput_path, name=f"latency_throughput_{quantisation_type}_{MODEL_TYPE}")      
        
        live.next_step() 

    print("DVC Live Report ready!")
    gc.collect()
    os._exit(0)


