
import onnx
from onnx import helper, TensorProto
import sys

def create_argmax_wrapper(input_model_path, output_model_path):
    print(f"Loading model: {input_model_path}")
    original_model = onnx.load(input_model_path)
    
    # Original Output Info
    original_output_name = original_model.graph.output[0].name
    original_output_info = original_model.graph.output[0]
    print(f"Original output name: {original_output_name}")
    print(f"Original output shape: {original_output_info.type.tensor_type.shape}")
    
    # 2. Konstante k=5 erstellen
    k_tensor = helper.make_tensor(
        name='k_value',
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[5]
    )
    original_model.graph.initializer.append(k_tensor)


    last_dq = None
    for node in original_model.graph.node:
        if node.op_type == "DequantizeLinear":
            if node.output[0] == original_output_name:
                last_dq = node
                break

    assert last_dq is not None, "No Dequantize feeding model output found"

    int8_tensor = last_dq.output[0]

    # cast = helper.make_node(
    #     "Cast",
    #     inputs=[int8_tensor],
    #     outputs=["fp32_boundary"],
    #     to=TensorProto.FLOAT
    # )

    # identity_node = helper.make_node(
    #     "Identity",
    #     inputs=[int8_tensor],
    #     outputs=["fp32_boundary"]
    # )

    topk_node = helper.make_node(
        'TopK',
        inputs=[int8_tensor, 'k_value'],
        outputs=['top_values', 'top_indices'],
        axis=-1,
        largest=1,
        sorted=1
    )


    #  original_model.graph.node.extend([identity_node, topk_node])
    original_model.graph.node.extend([topk_node])

    original_model.graph.output.clear()
    original_model.graph.output.append(
        helper.make_tensor_value_info(
            "top_indices",
            TensorProto.INT64,
            [1,256,5]
        )
    )
    
    # Modell überprüfen und speichern
    print("Checking model...")
    onnx.checker.check_model(original_model)
    
    print(f"Saving wrapped model: {output_model_path}")
    onnx.save(original_model, output_model_path)
    
    print("✓ Top-5 Wrapper erstellt!")
    print("Original output:", original_output_name)



if __name__ == "__main__":
    # INT8 Modell mit ArgMax Wrapper
    input_model = "outputs/language/model_brevitas_1_simple.onnx"
    output_model = "outputs/language/model_brevitas_1_argmax.onnx"
    
    create_argmax_wrapper(input_model, output_model)
    
# next step: quant_type='int' in export?




# 2. onnxscript, onnxir, onnx-passes 
# 1. pytorch methode ändern: topk (vor dem export)