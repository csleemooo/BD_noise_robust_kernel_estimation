# Blind image deblurring with noise-robust kernel estimation 
# Overview
Blind deblurring is an ill-posed inverse problem involving the retrieval of a clear image and blur kernel from a single blurry image. The challenge arises considerably when strong noise, where its level remains unknown, is introduced. Existing blind deblurring approaches heavily depend on designed priors for natural images and blur kernels. However, these methods are highly sensitive to noise due to the disturbance of the solution space. Here, we propose a noise-robust blind deblurring framework based on a novel kernel estimation function and deep image prior (DIP). Specifically, the proposed kernel estimation function mitigates noise and acquires the blur kernel, leveraging the capability of DIP to capture the priors of natural images. Additionally, the multiple kernel estimation scheme enables the successful execution of the deblurring task when the noise level is unknown. Extensive experimental studies, including simulated images and real-world examples, demonstrate the superior deblurring performance of the proposed method.
Last update: 24.03.15
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
pip install -r requirements.txt
```
which should install in about few minutes.

To run motion deblur on AFHQ-dog datasets,
```
python main_afhq_motion_blur.py --data_name afhq_dog
```
