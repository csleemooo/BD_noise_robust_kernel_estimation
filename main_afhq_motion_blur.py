import os
from tqdm import tqdm
import argparse
from glob import glob
from PIL import Image
import numpy as np
from torch import nn
import torch
from torchvision.transforms.functional import rgb_to_grayscale as rgb2gray

from core import get_noise, kef, save_imgs, TV_sparsity
from model_skip import skip_my

torch.manual_seed(777)

def parse_args():
    parser = argparse.ArgumentParser()

    # path for experiment
    parser.add_argument('--data_path', type=str, default="./datasets", help='path to blurry image')
    parser.add_argument("--save_path", default="./results", type=str)
    parser.add_argument("--data_name", default='afhq_dog', help='The name of dataset to evaluate: levin, sun, or ~.', type=str)
    parser.add_argument("--device", default=0, type=int)

    parser.add_argument("--channels", default=3, type=int)
    parser.add_argument("--filters", default=128, type=int)
    parser.add_argument("--ch_in", default=8, type=int)
    parser.add_argument("--inter_img", default=1, type=int)
    parser.add_argument("--input_mean", default=0, type=float)
    parser.add_argument("--input_std", default=0.1, type=float)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_df", default=1, type=float)
    parser.add_argument("--weight_sparsity", default=0.0001, type=float)
    parser.add_argument("--weight_psf_sparsity", default=1, type=float)
    parser.add_argument("--chk_iter", default=200, type=int)
    parser.add_argument("--iteration", default=1400, type=int)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    device=0
        
    save_root = os.path.join(args.save_path, args.data_name)
    k_list = [i**2 for i in [5, 15, 25]]
    num_k = len(k_list)

    for data_root in glob(os.path.join(args.data_path, args.data_name, 'blur', 'noise*')):
        for img_path in glob(os.path.join(data_root, '*.png')):
            
            imgname = os.path.splitext(os.path.basename(img_path))[0]
            noise_name = os.path.split(data_root)[1]
            noise_level = int(noise_name.split('_')[-1])

            path_save_f = os.path.join(save_root, noise_name)
            os.makedirs(path_save_f, exist_ok = True)
            for i in range(num_k):
                os.makedirs(os.path.join(path_save_f, 'kernel_%d'%i), exist_ok = True)
            
            print("Loaded file: img: %s"%img_path) 
            print('result is saved at: %s'%path_save_f)

            if args.channels == 3:
                color_mode = 'RGB'
                ch=3
            else:
                color_mode = 'L'
                ch=1
            y = np.array(Image.open(img_path).convert(color_mode), dtype=float)/(2**8 -1)

            img_size = y.shape
            h = img_size[0]
            w = img_size[1]

            if args.channels == 3:
                blur_img = torch.permute(torch.from_numpy(y), (2, 0, 1)).unsqueeze(0).float().to(device)
            else:
                blur_img = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float().to(device)

            kernel_path = os.path.join(args.data_path, args.data_name, 'gt_kernel', imgname.split('_')[-1] + '.png')
            gt_kernel = np.array(Image.open(kernel_path).convert("L"), dtype=float)/(2**8 -1)
            k_size = gt_kernel.shape[0]
    
            k = torch.Tensor(k_list).to(device).float().view(num_k, 1, 1, 1).expand(num_k, 1, h, w).repeat(1, 1, 1, 1)

            tensor_size = (1, args.ch_in, h, w)
            input_t = get_noise(args.ch_in, 'noise', (tensor_size[-2], tensor_size[-1], 'u')).to(device)
            num_output_channels = (num_k+1)*ch
            
            generator = skip_my(args.ch_in, num_output_channels,
                    num_channels_down = [args.filters]*5,
                    num_channels_up   = [args.filters]*5,
                    num_channels_skip = [args.filters//2]*5,
                    upsample_mode='nearest',
                    need_sigmoid=True, need_bias=True, pad='reflection',        
                    act_fun='LeakyReLU', drop=True, num_k=num_k).to(device).float()
            
            op = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.99))
            tv_sparsity = TV_sparsity(do_sqrt=False)

            adjust_lr = [600, 800, 1000, 1200, 1400, 1600]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=op, milestones=adjust_lr, gamma=0.5)
            loss_sum_sparsity=0
            loss_sum_id = 0
            loss_sum_psf_sparsity = 0

            loss_list=[]
            blur_img_gray = rgb2gray(blur_img)
            for it in tqdm(range(args.iteration), desc='Iteration:'):

                generator.train()
                op.zero_grad()
                
                noise = 0.001 * torch.zeros(input_t.shape).type_as(input_t.data).normal_()
                input_tensor = input_t + noise
                # generate image
                gen_out = generator(input_tensor)

                if args.inter_img:
                    img_pred_4_k = torch.cat([gen_out[:,i*ch:(i+1)*ch:, :, :] for i in range(num_k)], dim=0)
                else:
                    img_pred_4_k = gen_out[:,ch*num_k:, :, :].repeat(num_k,1,1,1)
                if ch==3:
                    img_pred_4_k = rgb2gray(img_pred_4_k)

                img_pred = gen_out[:,ch*num_k:, :, :]
                
                psf_full = kef(img_pred_4_k, blur_img_gray, num_k, k)
                psf_sparsity = torch.norm(psf_full, p=2, dim=(-2,-1)).mean()*args.weight_psf_sparsity

                img_sparsity = args.weight_sparsity*tv_sparsity(img_pred)

                # cropping operation
                psf = psf_full[:, :, h//2-k_size//2:h//2+ k_size//2+1, w//2-k_size//2: w//2 + k_size//2+1]
                psf = psf/(psf.sum(dim=(-2, -1), keepdim=True).expand(num_k, 1, k_size, k_size)+1e-8)

                # multiple psf
                diff_list = []
                img_pred_conv_list= []
                img_pred_grad_conv_list = []
                loss_id = 0
                
                for i in range(num_k):
                    psf_tmp = psf[i, :].view(1, 1, k_size, k_size).repeat(ch, 1, 1, 1)
                    
                    blur_img_est = nn.functional.conv2d(img_pred, psf_tmp, padding=k_size//2, stride=1, groups=ch)
                    loss_id = loss_id + torch.linalg.norm((blur_img_est-blur_img).view(-1))*args.weight_df/num_k
                        
                loss_total = img_sparsity + loss_id + psf_sparsity

                loss_list.append(loss_id.item())
                loss_sum_sparsity += img_sparsity.item()
                loss_sum_id += loss_id.item()
                loss_sum_psf_sparsity += psf_sparsity.item()
                
                
                loss_total.backward()
                op.step()
                
                if (it+1) in adjust_lr:
                    lr_scheduler.step(epoch=it+1)
                
                if (it+1)%args.chk_iter ==0 or it==0:
                    print('Loss identity: %1.6f, img sparsity: %1.6f, psf sparsity: %1.6f'%(loss_sum_id/args.chk_iter, loss_sum_sparsity/args.chk_iter, 
                            loss_sum_psf_sparsity/args.chk_iter))
                    
                    loss_sum_sparsity=0
                    loss_sum_id = 0     
                    loss_sum_psf_sparsity = 0
                    save_imgs(img_pred, img_pred_4_k, psf, color_mode, num_k, h, w, k_size, path_save_f, imgname)