import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from models.ChaIR import build_net

def load_image(image_path, new_size=(1024, 768)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(new_size, Image.BICUBIC)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor

def save_image(tensor, save_path):
    tensor = torch.clamp(tensor, 0.0, 1.0)
    np_img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    np_img = (np_img * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(np_img)
    pil_img.save(save_path)
    print(f"Image saved as: {save_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_net()
    model.to(device)

    print(f"Loading model: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if os.path.isdir(args.input_path):
        images = [
            os.path.join(args.input_path, f) 
            for f in os.listdir(args.input_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)
    else:
        images = [args.input_path]

    factor = 4

    for img_path in images:
        input_img = load_image(img_path, new_size=(1024, 768)).to(device)
        orig_h, orig_w = input_img.shape[2], input_img.shape[3]

        H = ((orig_h + factor) // factor) * factor
        W = ((orig_w + factor) // factor) * factor
        padh = H - orig_h if orig_h % factor != 0 else 0
        padw = W - orig_w if orig_w % factor != 0 else 0

        input_img_pad = F.pad(input_img, (0, padw, 0, padh), mode='reflect')

        if torch.cuda.is_available(): 
            torch.cuda.empty_cache()

        with torch.no_grad():
            outputs = model(input_img_pad)
            pred = outputs[-1][:, :, :orig_h, :orig_w]

        if os.path.isdir(args.output_path):
            base_name = os.path.basename(img_path)
            out_name = os.path.join(args.output_path, f"dehazed_{base_name}")
        else:
            out_name = args.output_path

        save_image(pred, out_name)

def parse_args():
    parser = argparse.ArgumentParser(description="Script for dehazing images using ChaIR model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path to the trained model (e.g, Desktop/model_ots_4073_9968.pkl)."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="The path to the input image (or folder) we want to dehaze."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="The path where the result will be saved. If it is a folder and input_path is a folder, it saves everything."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)