# CODE WITH DUMMY INPUTS

# import numpy as np

# import qai_hub as hub

# sample = np.random.random((1, 3, 224, 224)).astype(np.float32)

# inference_job = hub.submit_inference_job(
#     model="job_jpvqe28rg_optimized_onnx_mngry295q.onnx",
#     device=hub.Device("QCS8550 (Proxy)"),
#     inputs=dict(image=[sample]),
# )

# assert isinstance(inference_job, hub.InferenceJob)
# inference_job.download_output_data()

# CODE WITH HAZY DATASET

import os
import numpy as np
from PIL import Image
import qai_hub as hub

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((3,1,1))
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((3,1,1))

sample = []
images_dir = "hazy"

for img_file in os.listdir(images_dir):
    path = os.path.join(images_dir, img_file)
    image = Image.open(path).convert("RGB")
    image = image.resize((224, 224))

    img_np = np.array(image, dtype=np.float32) / 255.0
    img_np = (img_np - mean.transpose((1,2,0))) / std.transpose((1,2,0))
    img_np = np.transpose(img_np, (2,0,1))
    img_np = np.expand_dims(img_np, 0)

    sample.append(img_np)

inference_job = hub.submit_inference_job(
    model="job_jgn0endq5_optimized_onnx_mqkgdr1wm.onnx",
    device=hub.Device("QCS8550 (Proxy)"),
    inputs=dict(image=sample),
)

assert isinstance(inference_job, hub.InferenceJob)