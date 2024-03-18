# Blind image deblurring with noise-robust kernel estimation 
# Overview
Blind deblurring is an ill-posed inverse problem involving the retrieval of a clear image and blur kernel from a single blurry image. The challenge arises considerably when strong noise, where its level remains unknown, is introduced. Existing blind deblurring approaches heavily depend on designed priors for natural images and blur kernels. However, these methods are highly sensitive to noise due to the disturbance of the solution space. Here, we propose a noise-robust blind deblurring framework based on a novel kernel estimation function and deep image prior (DIP). Specifically, the proposed kernel estimation function mitigates noise and acquires the blur kernel, leveraging the capability of DIP to capture the priors of natural images. Additionally, the multiple kernel estimation scheme enables the successful execution of the deblurring task when the noise level is unknown. Extensive experimental studies, including simulated images and real-world examples, demonstrate the superior deblurring performance of the proposed method.
Last update: 24.03.18
## Packages
The following libraries are necessary for running the codes.
- torch==2.1.2
- torchvision==0.16.2
- numpy==1.24.1
- matplotlib==3.8.2
- pillow==9.3.0
- tqdm==4.66.1

Please install requirements using below command.
```
conda create -n bd_noise python=3.11

conda activate bd_noise

pip install -r requirements.txt

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```
which should install in about few minutes.

## Environments
The code is tested on the following system and drivers.

Windows 11 / CUDA 11.8 / RTX3080

## Motion deblur
To run motion deblur on AFHQ-dog or AFHQ-cat datasets
```
python main_afhq_motion_blur.py --data_name afhq_dog
python main_afhq_motion_blur.py --data_name afhq_cat
```
The deblurring results on each dataset are uploded in ./results folder

Code and datasets for further experiements will be uploaded soon.
