import os,time,scipy.io

from PIL import Image
import numpy as np
import rawpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim

from src.LookInDark import LookInDark
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
m_path = './saved_model/'
m_name = 'checkpoint_sony_e4000.pth'
result_dir = './result/'

split = 512
WIDTH = 4096
HEIGHT = 2048+512

def pack_raw(im):
    #pack Bayer image to 4 channels
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out

def ResultGenerator(input_paths, gt_paths):
    model = LookInDark()
    model.load_state_dict(torch.load( m_path + m_name ,map_location={'cuda:1':'cuda:0'}))
    print('model loaded')
    model = model.to(device)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    input_img, gt_img, scale_img, output_img = [], [], [], []
    for i in range(len(input_paths)):
        st = time.time()
        _, in_fn = os.path.split(input_paths[i])
        print(in_fn)
        _, gt_fn = os.path.split(gt_paths[i])
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        ratio = min(gt_exposure/in_exposure,300)

        raw = rawpy.imread(input_paths[i])
        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        im = im[:HEIGHT, :WIDTH]
        scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)

        gt_raw = rawpy.imread(gt_paths[i])
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        print(im.shape)
        im = im[:HEIGHT, :WIDTH]
        gt_full = np.expand_dims(np.float32(im/65535.0),axis = 0)

        im_ori = raw.raw_image_visible.astype(np.float32)

        output = np.zeros(gt_full.shape)
        for im_hei in np.arange(0, HEIGHT, split):
            for im_wei in np.arange(0, WIDTH, split):
                im = im_ori[im_hei:im_hei+split, im_wei:im_wei+split]
                input_part = np.expand_dims(pack_raw(im),axis=0) *ratio
                input_part = np.minimum(input_part,1.0)

                in_img = torch.from_numpy(input_part).permute(0,3,1,2).to(device)
                out_img = model(in_img)
                output_part = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
                output_part = np.minimum(np.maximum(output_part,0),1)

                output[:,im_hei:im_hei+split, im_wei:im_wei+split] = output_part

        output = output[0,:,:,:]
        gt_full = gt_full[0,:,:,:]
        scale_full = scale_full[0,:,:,:]
        origin_full = scale_full
        scale_full = scale_full*np.mean(gt_full)/np.mean(scale_full) # scale the low-light image to the same mean of the groundtruth
        
        scipy.misc.toimage(origin_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_ori.png'%(int(gt_fn[0:5]),ratio))
        scipy.misc.toimage(output*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_out.png'%(int(gt_fn[0:5]),ratio))
        scipy.misc.toimage(scale_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_scale.png'%(int(gt_fn[0:5]),ratio))
        scipy.misc.toimage(gt_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_gt.png'%(int(gt_fn[0:5]),ratio))

        print('elapse time:', time.time() - st)
        input_img.append(Image.fromarray((origin_full*255).astype('uint8'), 'RGB'))
        gt_img.append(Image.fromarray((gt_full*255).astype('uint8'), 'RGB'))
        scale_img.append(Image.fromarray((scale_full*255).astype('uint8'), 'RGB'))
        output_img.append(Image.fromarray((output*255).astype('uint8'), 'RGB'))

    return input_img, gt_img, scale_img, output_img