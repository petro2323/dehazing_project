import os

import numpy as np
from PIL import Image

import qai_hub as hub

unquantized_onnx_model = "job_jgn0endq5_optimized_onnx_mqkgdr1wm.onnx"

mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
sample_inputs = []

images_dir = "hazy"
for image_path in os.listdir(images_dir):
    image = Image.open(os.path.join(images_dir, image_path))
    image = image.convert("RGB").resize((1, 3, 224, 224)[2:])
    sample_input = np.array(image).astype(np.float32) / 255.0
    sample_input = np.expand_dims(np.transpose(sample_input, (2, 0, 1)), 0)
    sample_inputs.append(((sample_input - mean) / std).astype(np.float32))
calibration_data = dict(image=sample_inputs)

quantize_job = hub.submit_quantize_job(
    model=unquantized_onnx_model,
    calibration_data=calibration_data,
    weights_dtype=hub.QuantizeDtype.INT8,
    activations_dtype=hub.QuantizeDtype.INT8,
)

quantized_onnx_model = quantize_job.get_target_model()
assert isinstance(quantized_onnx_model, hub.Model)