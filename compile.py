import qai_hub as hub

compile_job = hub.submit_compile_job(
    model="dehazing_model.onnx",
    device=hub.Device("QCS8550 (Proxy)"),
    options="--target_runtime onnx",
    input_specs=dict(image=(1, 3, 224, 224)),
)
assert isinstance(compile_job, hub.CompileJob)