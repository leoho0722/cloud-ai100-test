import os

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM

model_name = "gpt2"
qeff_model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"{model_name} optmized for AI 100 \n", qeff_model)

onnx_model_path = qeff_model.export()

generated_qpc_path = qeff_model.compile(
    num_cores=14,
    mxfp6=True,
    device_group=[0],
)
