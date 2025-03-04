# Outdoor Image Dehazing Using ChaIR Model on QCS8550

## Description

This project focuses on outdoor image dehazing using the ChaIR model, optimized for execution on Qualcomm's QCS8550 platform via Qualcomm AI Hub. The goal is to achieve real-time dehazing performance, particularly from aerial perspectives (e.g., helicopter-mounted cameras).

## Installation
The project is built with PyTorch 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
For installing, follow these instructions:
~~~
conda install pytorch=1.8.1 torchvision=0.9.1 -c pytorch
pip install tensorboard einops scikit-image pytorch_msssim opencv-python
~~~
Install warmup scheduler:
~~~
cd pytorch-gradual-warmup-lr/
python setup.py install
cd ..
~~~

## ChaIR Results Summary
We compared the quantized and non-quantized versions of the model. Both were tested using the hazy [SOTS](https://drive.google.com/file/d/16j2dwVIa9q_0RtpIXMzhu-7Q6dwz_D1N/view) dataset.

### Image Results [here](https://drive.google.com/drive/folders/1jT7Sx52Cu0POTZQcDkdJouYK2rARRQw9?usp=sharing)
### Download the model [here](https://drive.google.com/drive/folders/1jayXhd7K9NiwcNZXsoGxH8HLNHxdL4XT?usp=sharing)

### Non-Quantized Model

| **Metric**             | **Value**       |
|------------------------|----------------|
| **Minimum Inference Time** | 260.8 ms     |
| **RAM Usage**          | 137-225 MB      |
| **PSNR**               | 39.34 dB        |
| **SSIM**               | 0.996 dB        |

### Quantized Model

| **Metric**             | **Value**       |
|------------------------|----------------|
| **Minimum Inference Time** | 135.8 ms     |
| **RAM Usage**          | 161-213 MB      |
| **PSNR**               | 42.1 dB        |
| **SSIM**               | 0.997 dB        |

## Evaluation

#### Dehazing One Image

~~~
python dehaze_inference.py --model_path path_to_ots_model --input_path your_path/image_name --output_path your_path/image_name
~~~

#### Dehazing More Images

~~~
python dehaze_inference.py --model_path path_to_ots_model --input_path --input_path your_path/folder_name --output_path your_path/folder_name
~~~

#### Testing on SOTS-Outdoor
~~~
cd OTS
python main.py --data_dir your_path/reside-outdoor --test_model path_to_ots_model
~~~

For testing, your directory structure should look like this

`Your path` <br/>
`├──reside-indoor` <br/>
     `├──train`  <br/>
          `├──gt`  <br/>
          `└──hazy`  
     `└──test`  <br/>
          `├──gt`  <br/>
          `└──hazy`  
`└──reside-outdoor` <br/>
     `├──train`  <br/>
          `├──gt`  <br/>
          `└──hazy`  
     `└──test`  <br/>
          `├──gt`  <br/>
          `└──hazy` 

## Acknowledgments

This project extends the work of the [ChaIR repository](https://github.com/c-yn/ChaIR) and was developed for real-time outdoor aerial dehazing using Qualcomm AI Hub.