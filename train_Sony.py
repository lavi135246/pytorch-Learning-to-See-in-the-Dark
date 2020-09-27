import os
import time

import numpy as np
import rawpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image

from model import SeeInDark

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
result_dir = './result_Sony/'
model_dir = './saved_model/'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

#get train and test IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))

test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))



ps = 512 #patch size for training
save_freq = 100

DEBUG = 0
if DEBUG == 1:
    save_freq = 100
    train_ids = train_ids[0:5]
    test_ids = test_ids[0:5]

def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32) 
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

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


#Raw data takes long time to load. Keep them in memory after loaded.
gt_images=[None]*6000
input_images = {}
input_images['300'] = [None]*len(train_ids)
input_images['250'] = [None]*len(train_ids)
input_images['100'] = [None]*len(train_ids)

g_loss = np.zeros((5000,1))



allfolders = glob.glob('./result/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
model = SeeInDark().to(device)
model._initialize_weights()
opt = optim.Adam(model.parameters(), lr = learning_rate)
for epoch in range(lastepoch,4001):
    if os.path.isdir("result/%04d"%epoch):
        continue    
    cnt=0
    if epoch > 2000:
        for g in opt.param_groups:
            g['lr'] = 1e-5
  

    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW'%train_id)
        in_path = in_files[np.random.randint(0,len(in_files))]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%train_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        ratio = min(gt_exposure/in_exposure,300)
          
        st=time.time()
        cnt+=1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw),axis=0) *ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im/65535.0),axis = 0)

         
        #crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0,W-ps)
        yy = np.random.randint(0,H-ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:,yy:yy+ps,xx:xx+ps,:]
        gt_patch = gt_images[ind][:,yy*2:yy*2+ps*2,xx*2:xx*2+ps*2,:]
       

        if np.random.randint(2,size=1)[0] == 1:  # random flip 
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1: 
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose 
            input_patch = np.transpose(input_patch, (0,2,1,3))
            gt_patch = np.transpose(gt_patch, (0,2,1,3))
        
        
        input_patch = np.minimum(input_patch,1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        
        in_img = torch.from_numpy(input_patch).permute(0,3,1,2).to(device)
        gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2).to(device)

        model.zero_grad()
        out_img = model(in_img)

        loss = reduce_mean(out_img, gt_img)
        loss.backward()

        opt.step()
        g_loss[ind]=loss.data.cpu()

        mean_loss = np.mean(g_loss[np.where(g_loss)])
        print(f"Epoch: {epoch} \t Count: {cnt} \t Loss={mean_loss:.3} \t Time={time.time()-st:.3}")

        if epoch%save_freq==0:
            epoch_result_dir = result_dir + f'{epoch:04}/'

            if not os.path.isdir(epoch_result_dir):
                os.makedirs(epoch_result_dir)

            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output,0),1)
            
            temp = np.concatenate((gt_patch[0,:,:,:], output[0,:,:,:]),axis=1)
            Image.fromarray((temp*255).astype('uint8')).save(epoch_result_dir + f'{train_id:05}_00_train_{ratio}.jpg')
            torch.save(model.state_dict(), model_dir+'checkpoint_sony_e%04d.pth'%epoch)

