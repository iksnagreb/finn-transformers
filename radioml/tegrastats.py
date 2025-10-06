import tensorrt as trt
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import time
import json
import torch
import onnx
#from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
import pycuda.driver as cuda
# import pycuda.autoinit
import os
import yaml
from onnxconverter_common import float16 # zu requirements hinzufügen
import onnxruntime as ort
import dvc.api
import model
import subprocess
import parse_tegrastats_to_json
import power_averages
from datetime import datetime
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# tensorrt, datasets(hugging face), pycuda
FP16 = os.environ.get("FP16", "0") == "1"
INT8 = os.environ.get("INT8", "0") == "1"

if FP16:
    dtype = torch.float16
    print("FP16 enabled")
elif INT8:
    dtype = torch.int8
    print("INT8 enabled")
else:
    dtype = torch.float32
    print("FP32")


RADIOML_PATH = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.hdf5"
RADIOML_PATH_NPZ = R"/home/hanna/git/radioml-transformer/data/GOLD_XYZ_OSC.0001_1024.npz"


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
    with open(filepath, "w") as f:
        json.dump(log, f, indent=4)


def parse_shape(shape, batch_value):
    """Ersetzt 'batch_size' durch batch_value in der shape-Liste."""
    return tuple(
        batch_value if d == "batch_size"
        else 1 if (i == 1 and INT8)  # zweite Dimension immer 1 im INT8-Modus
        else batch_value if (i == 0 and INT8)
        else 128 if d == "sequence_length"
        else 64 if d == "Muloutput_dim_2"
        else d
        for i, d in enumerate(shape)
    )


ONNX_TO_TORCH_DTYPE = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(double)": torch.float64,
    "tensor(int32)": torch.int32,
    "tensor(int64)": torch.int64,
    "tensor(uint8)": torch.uint8,
    "tensor(int8)": torch.int8,
    "tensor(bool)": torch.bool,
    # Füge weitere Typen bei Bedarf hinzu
}


def onnx_dtype_to_torch(onnx_dtype_str):
    """
    Wandelt einen ONNX-Datentyp-String in einen torch.dtype um.
    """
    return ONNX_TO_TORCH_DTYPE.get(onnx_dtype_str, torch.float32)  # Default: float32


def get_model_io_info(model_path):
    """
    Liest Input- und Output-Infos aus einem ONNX-Modell.
    Gibt Listen von Dictionaries mit Name, Shape und Dtype zurück.
    """
    sess_options = ort.SessionOptions()

    sess_options.intra_op_num_threads = 8
    session = ort.InferenceSession(model_path, sess_options)
    input_info = [
        {
            "name": inp.name,
            "shape": inp.shape,
            "dtype": inp.type
        }
        for inp in session.get_inputs()
    ]
    output_info = [
        {
            "name": out.name,
            "shape": out.shape,
            "dtype": out.type
        }
        for out in session.get_outputs()
    ]
    return input_info, output_info



def create_test_dataloader(RADIOML_PATH_NPZ, batch_size):
    """
    Erstellt den DataLoader für die Testdaten.
    :param RADIOML_PATH: Pfad zur Testdaten-Datei.
    :param batch_size: Die Batchgröße.
    :return: DataLoader-Objekt für die Testdaten.
    """
    data = np.load(RADIOML_PATH_NPZ)
    input_info, output_info = get_model_io_info(onnx_model_path)
    key_list = list(data.keys())
    if len(input_info) == 2:
        input_key = key_list[0]
        attention_mask_key = key_list[1]
        output_key = key_list[2]
    elif len(input_info) == 3:
        input_key = key_list[0]
        attention_mask_key = key_list[1]
        output_key = key_list[2]
    else:
        input_key = key_list[0]
        attention_mask_key = None
        output_key = key_list[1]
    
    input_ids = torch.from_numpy(data[input_key])

    input_ids = input_ids.reshape(-1, 1, 1024, 2)


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


    num_samples = len(test_dataset)
    return test_loader

def test_data(context, batch_size, input_info, output_info):
    device_inputs = {}
    device_outputs = {}
    device_attention_masks = {}
    device_token_type = {}
    torch_stream = torch.cuda.Stream()
    stream_ptr = torch_stream.cuda_stream

    # Inputs vorbereiten
    inp = input_info[0]
    name = inp["name"]
    shape = parse_shape(inp["shape"], batch_size)
    dtype = onnx_dtype_to_torch(inp["dtype"])  # ONNX-Datentyp in PyTorch-Datentyp umwandeln
    dtype_out = onnx_dtype_to_torch(output_info[0]["dtype"])  # Ausgabe-Datentyp
    tensor = torch.empty(shape, dtype=dtype, device='cuda')
    context.set_tensor_address(name, tensor.data_ptr())
    context.set_input_shape(name, shape)
    device_inputs[name] = tensor
    
    # Wenn mehrere Inputs: attention_mask
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

    # Outputs vorbereiten
    for out in output_info:
        name = out["name"]
        shape = parse_shape(out["shape"], batch_size)
        dtype = onnx_dtype_to_torch(out["dtype"])  # ONNX-Datentyp in PyTorch-Datentyp umwandeln
        tensor = torch.empty(shape, dtype=dtype_out, device='cuda')
        context.set_tensor_address(name, tensor.data_ptr())
        device_outputs[name] = tensor

    device_input = next(iter(device_inputs.values()))
    device_output = next(iter(device_outputs.values()))
    # if another input: nächste attmask adresse
    if len(input_info) > 1:
        device_attention_mask = next(iter(device_attention_masks.values()))
    else:
        device_attention_mask = None

    if len(input_info) > 2:
        device_token_type = next(iter(device_token_type.values()))
    else:
        device_token_type = None

    return device_input, device_output, device_attention_mask, device_token_type, stream_ptr, torch_stream


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
    
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 40)

    if FP16 == True:
        config.set_flag(trt.BuilderFlag.FP16)
    if INT8 == True:
        config.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()

    for inp in input_info:
        name = inp["name"]
        shape = inp["shape"]
        min_shape = parse_shape(shape, min_bs)
        opt_shape = parse_shape(shape, opt_bs)
        max_shape = parse_shape(shape, max_bs)
        profile.set_shape(name, min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Fehler beim Bauen der TensorRT-Engine: serialized_engine ist None.")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    return engine, context


def run_inference(context, test_loader, device_input, device_output, device_attention_mask, device_token_type, stream_ptr, torch_stream, batch_size=1, input_info=None, output_info=None, accuracy_flag=False):
    """
    Funktion zur Bestimmung der Inferenzlatenz.
    """


    total_predictions = 0
    correct_predictions = 0
    iterations=0

    for batch in test_loader: 
        # je nach Aufbau des Modells: mit Attention Mask oder ohne
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
        
        start_time_datatransfer = time.time()  # Startzeit        

        dtype = onnx_dtype_to_torch(input_info[0]["dtype"])

        input_name = input_info[0]["name"]
        if device_input.shape != xb.shape:
            device_input.resize_(xb.shape)  # Dynamisch anpassen
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

        output_name = output_info[0]["name"]
        context.set_tensor_address(output_name, device_output.data_ptr()) 

        
        torch_stream.synchronize()

        
        try:
            with torch.cuda.stream(torch_stream):
                context.execute_async_v3(stream_ptr)
        except Exception as e:
            print("TensorRT Error:", e)
        torch_stream.synchronize() 
    
        
        output = device_output.cpu().numpy()
        iterations += 1


        if accuracy_flag:
            pred = output.argmax(axis=-1)  # [batch, seq_len]
            correct = (pred == yb.numpy()).sum()
            total = len(yb)
            correct_predictions += correct
            total_predictions += total
            

    accuracy = 0
    if accuracy_flag:
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0


    return 0, 0, 0, accuracy


def start_tegrastats(logfile_path: Path):
    # tegrastats im Hintergrund starten, Ausgabe in Logdatei
    proc = subprocess.Popen(['sudo', 'tegrastats', '--interval', '1000'], stdout=open(logfile_path, 'w'))
    return proc

def stop_tegrastats(proc: subprocess.Popen):
    proc.terminate()  # schickt SIGTERM an tegrastats
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()

def run_accuracy_eval(batch_size, input_info, output_info, RADIOML_PATH_NPZ, onnx_model_path, tegrastats_log, timestamps_file):
    test_loader = create_test_dataloader(RADIOML_PATH_NPZ, batch_size)
    engine, context = build_tensorrt_engine(onnx_model_path, test_loader, batch_size, input_info)
    device_input, device_output, device_attention_mask, device_token_type, stream_ptr, torch_stream = test_data(context, batch_size, input_info, output_info)

    
    tegra_proc = start_tegrastats(tegrastats_log)

    time.sleep(10)

    print("Startzeit: ", time.time())

    timestamp = time.time()
    start_iso = datetime.fromtimestamp(timestamp).isoformat(timespec='milliseconds')

    data = {"start_time": start_iso}


    
        
    for i in range(10):
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
                    accuracy_flag=True
                )

    

    timestamp = time.time()
    end_iso = datetime.fromtimestamp(timestamp).isoformat(timespec='milliseconds')

    time.sleep(10)
    

    timestamps = {
        "start_time": start_iso,
        "end_time": end_iso
    }
    with open(timestamps_file, "w") as f:
        json.dump(timestamps, f, indent=2)

    stop_tegrastats(tegra_proc)

    

    return accuracy


if __name__ == "__main__":

    
    if FP16:
        params = dvc.api.params_show(stages="radioml/dvc.yaml:measure_16FP")
    elif INT8:
        params = dvc.api.params_show(stages="radioml/dvc.yaml:measure_INT8_brevitas")
    else:
        params = dvc.api.params_show(stages="radioml/dvc.yaml:measure_32FP")

    batch_sizes = params["batch_sizes"]

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


    onnx_model_path = "outputs/radioml/model_dynamic_batchsize.onnx"

    tegrastats_logs = []

    for batch_size in batch_sizes:
        if INT8:
            onnx_model_path = f"outputs/radioml/model_brevitas_{batch_size}_simpl.onnx"
        input_info, output_info = get_model_io_info(onnx_model_path)
        tegrastats_log = Path(__file__).resolve().parent.parent / "outputs" / "radioml" / "energy_metrics" / f"tegrastats_{batch_size}.log"
        timestamps = Path(__file__).resolve().parent.parent / "outputs" / "radioml" / "energy_metrics" / f"timestamps_{batch_size}.json"
        accuracy = run_accuracy_eval(batch_size, input_info, output_info, RADIOML_PATH_NPZ, onnx_model_path, tegrastats_log, timestamps)
        print(f"Accuracy for batch size {batch_size}: {accuracy:.4f}")

        tegrastats_logs.append((tegrastats_log, batch_size))

    parse_tegrastats_to_json.parse_tegrastats(tegrastats_logs)
    power_averages.power_averages(batch_sizes)
    


    # erster wert: current
    # /
    # zweiter wert: average

