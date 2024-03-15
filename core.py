# all functions 
import numpy as np
import torch
from torch import nn
from math import exp
from PIL import Image
import os

class TV_map(nn.Module):

    def __init__(self, do_sqrt=True):
        super(TV_map, self).__init__()
        self.do_sqrt = do_sqrt

    def forward(self, x):
        _, ch, Nx, Ny = x.shape
        if self.do_sqrt:
            gradient = torch.sqrt(torch.pow(torch.abs(x[:, :, 0:Nx-1, :-1] - x[:, :, 1:Nx, :-1]), 2)
                              +torch.pow(torch.abs(x[:, :, :-1, 0:Ny-1] - x[:, :, :-1, 1:Ny]), 2) + 1e-8)
            # gradient = torch.abs(x[:, :, 0:Nx-1, 1:] - x[:, :, 1:Nx, 1:]).sum(axis=(-2, -1), keepdims=False) 
            # + torch.abs(x[:, :, 1:, 0:Ny-1] - x[:, :, 1:, 1:Ny]).sum(axis=(-2, -1), keepdims=False)

        else:
            xx = x[:,:,1:,:] - x[:,:,:-1,:]
            yy = x[:,:,:,1:] - x[:,:,:,:-1]
            gradient = (torch.sum(torch.abs(xx), dim=(-2, -1)) + torch.sum(torch.abs(yy), dim=(-2, -1)))
            # gradient = (torch.abs(x[:, :, 0:Nx-1, 1:] - x[:, :, 1:Nx, 1:]) + 1e-8)**(2/3)
            # + (torch.abs(x[:, :, 1:, 0:Ny-1] - x[:, :, 1:, 1:Ny]) + 1e-8)**(2/3)

        return gradient       

class TV_sparsity(nn.Module):

    def __init__(self, do_sqrt=True):
        super(TV_sparsity, self).__init__()
        self.tv_map = TV_map(do_sqrt=do_sqrt)

    def forward(self, x, average=True):
        gradient = self.tv_map(x)
        tv_value = gradient

        if average:
            return tv_value.mean()
        else:
            return tv_value
        

def kef(img_pred_4_k, blur_img_gray, num_k, k):
    """kernel estimation function"""
    img_pred_4_k_fft = torch_fft(img_pred_4_k)
    img_tar_fft = torch_fft(blur_img_gray).repeat(num_k, 1, 1, 1) 

    inverse_filt_fft = torch.conj(img_pred_4_k_fft)/(img_pred_4_k_fft.abs().pow(2) + k)
    psf_full_fft = inverse_filt_fft*img_tar_fft
    psf_full = torch.real(torch_ifft(psf_full_fft)).abs()

    return psf_full

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    torch.manual_seed(0)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def gaussian_2d ():
    return 0

def torch_fft(H):

    H = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(H, dim=(-2, -1))), dim=(-2, -1))

    return H

def torch_ifft(H):

    H = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(H, dim=(-2, -1))), dim=(-2, -1))

    return H

def save_img(root, img):

    if img.shape[1] == 3:
        img = Image.fromarray(np.uint8(img.cpu().detach().numpy()[0].transpose(1,2,0)*255))
    else:
        img = Image.fromarray(np.uint8(img.cpu().detach().numpy()[0][0]*255))
    img.save(root)
    
def get_generator():
    global operation_seed_counter
    operation_seed_counter += 1
    g_cuda_generator = torch.Generator(device="cuda")
    g_cuda_generator.manual_seed(operation_seed_counter)
    return g_cuda_generator
    
    
def space_to_depth(x, block_size):
    n, c, h, w = x.size()
    unfolded_x = torch.nn.functional.unfold(x, block_size, stride=block_size)
    return unfolded_x.view(n, c * block_size**2, h // block_size,
                           w // block_size)
def save_imgs(img_pred, img_pred_4_k, psf, color_mode, num_k, h, w, k_size, path_save_f, imgname):
    if color_mode=='RGB':
        img_pred = torch.permute(img_pred, (0, 2, 3, 1))
        img_pred = Image.fromarray(np.uint8(img_pred.cpu().detach().numpy()[0]*255))
    else:
        img_pred = Image.fromarray(np.uint8(img_pred.cpu().detach().numpy()[0, 0]*255))
    
    inter_imgs = Image.new('L', (w*num_k, h))
    entire_psf = Image.new('L', (k_size*num_k, k_size))
    for i in range(num_k):
        
        # save intermediate images
        n = img_pred_4_k.cpu().detach().numpy()[i, 0, :, :]
        n = Image.fromarray(np.uint8(n*255))
        inter_imgs.paste(n, (w*i, 0))
        
        # save blur kernels
        p = psf.cpu().detach().numpy()[i][0]
        p = p/np.max(p)
        p = Image.fromarray(np.uint8(p*255))                                        
        p.save(os.path.join(path_save_f, 'kernel_%d'%i, '%s_kernel.png'%imgname))
        entire_psf.paste(p, (k_size*i, 0))
        
    entire_psf.save(os.path.join(path_save_f, '%s_kernel.png'%imgname))
    inter_imgs.save(os.path.join(path_save_f, '%s_inter_img.png'%imgname))
    img_pred.save(os.path.join(path_save_f, '%s_result.png'%imgname))