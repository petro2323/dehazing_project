import torch
from models.ChaIR import build_net

model = build_net()

checkpoint = torch.load("model_ots_4073_9968.pkl", map_location="cpu")
model.load_state_dict(checkpoint["model"])  # ✅ Učitaj state_dict iz ključa "model"

model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

onnx_path = "dehazing_model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11)