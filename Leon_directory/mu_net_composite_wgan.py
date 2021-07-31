# SOME BASIC IMPORTS
import math
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import random
from PIL import Image
from scipy import ndimage
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import gc

randseed = 27512838
random.seed(randseed)
np.random.seed(randseed)
np.random.RandomState(randseed)
torch.manual_seed(randseed)
torch.use_deterministic_algorithms(mode=True)

AUGMENT = False
NORMALIZE = False
SMOOTH = False
M_NOISE = False
IMG_SIDE = 384
IMG_SIZE = (IMG_SIDE, IMG_SIDE)

GENERATOR_EPOCHS = 5
DISCRIMINATOR_EPOCHS = 2
GAN_EPOCHS = 15

DRAW = False
def plt_draw():
    if DRAW:
        plt.draw()
        plt.pause(0.001)
    return


#load data into np array function
def load_all_from_path(path):
    # loads all HxW gpngs contained in path as a 4D np.array of shape (n_images,
    # H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return np.stack([np.array(Image.open(f)) for f in sorted(glob(path + '/*.png'))]).astype(np.float32) / 255.

def load_indexed_from_path(path, indices):
    # loads all HxW gpngs contained in path as a 4D np.array of shape (n_images,
    # H, W, 3)
    # images are loaded as floats with values in the interval [0., 1.]
    return np.stack([np.array(Image.open(f)) for k, f in enumerate(sorted(glob(path + '/*.png'))) if k in indices]).astype(np.float32) / 255.

#load data
FULL = 15
SIZE = 15
idx = random.sample(set(range(FULL)), SIZE)
imgs = load_all_from_path(os.path.join('training', 'images'))
masks = load_all_from_path(os.path.join('training', 'groundtruth'))

#RANDOM ROTATE IMAGE IN STEP OF 90, THEN EITHER FLIP VERTICAL OR HORIZONTAL
#AUGMENT x9
def augment_data_random(imgs,masks):
 
    new_imgs = []
    new_masks = []
    for x, y in zip(imgs, masks):
        new_imgs.append(x)
        new_masks.append(y)
        angle = random.randint(0, 90)
        new_imgs.append(ndimage.rotate(x, angle, reshape=False, mode='reflect'))
        new_masks.append(ndimage.rotate(y, angle, reshape=False, mode='reflect'))
        new_imgs.append(np.fliplr(ndimage.rotate(x, angle, reshape=False, mode='reflect')))
        new_masks.append(np.fliplr(ndimage.rotate(y, angle, reshape=False, mode='reflect')))
        angle = random.randint(90, 180)
        new_imgs.append(ndimage.rotate(x, angle, reshape=False, mode='reflect'))
        new_masks.append(ndimage.rotate(y, angle, reshape=False, mode='reflect'))
        new_imgs.append(np.flipud(ndimage.rotate(x, angle, reshape=False, mode='reflect')))
        new_masks.append(np.flipud(ndimage.rotate(y, angle, reshape=False, mode='reflect')))
        angle = random.randint(180, 270)
        new_imgs.append(ndimage.rotate(x, angle, reshape=False, mode='reflect'))
        new_masks.append(ndimage.rotate(y, angle, reshape=False, mode='reflect'))
        new_imgs.append(np.fliplr(ndimage.rotate(x, angle, reshape=False, mode='reflect')))
        new_masks.append(np.fliplr(ndimage.rotate(y, angle, reshape=False, mode='reflect')))
        angle = random.randint(270, 360)
        new_imgs.append(ndimage.rotate(x, angle, reshape=False, mode='reflect'))
        new_masks.append(ndimage.rotate(y, angle, reshape=False, mode='reflect'))
        new_imgs.append(np.flipud(ndimage.rotate(x, angle, reshape=False, mode='reflect')))
        new_masks.append(np.flipud(ndimage.rotate(y, angle, reshape=False, mode='reflect')))

    return  np.asarray(new_imgs), np.asarray(new_masks)

def normalize_images(imgs):
    norm_imgs = []
    for img in imgs:
        n_img = img - np.mean(img)
        n_img = n_img / np.std(n_img)
        norm_imgs.append(n_img)
    return np.asarray(norm_imgs).astype(np.float32)

def smooth_masks(masks):
    smooth_masks = []
    for mask in masks:
        smooth_masks.append(ndimage.gaussian_filter(mask, sigma=(6, 6), order=0))
    return np.asarray(smooth_masks).astype(np.float32)

def noise_masks(masks):
    noisy_masks = []
    for mask in masks:
        noisy_masks.append(mask + np.random.normal(0, 0.01, (400, 400)))
    return np.asarray(noisy_masks).astype(np.float32)

def show_first_n(imgs, masks, n=5):
    # visualizes the first n elements of a series of images and segmentation masks
    imgs_to_draw = min(5, len(imgs))
    plt.close()
    fig, axs = plt.subplots(2, imgs_to_draw, figsize=(18.5, 6))
    for i in range(imgs_to_draw):
        axs[0, i].imshow(imgs[i])
        axs[1, i].imshow(masks[i])
        axs[0, i].set_title(f'Image {i}')
        axs[1, i].set_title(f'Mask {i}')
        axs[0, i].set_axis_off()
        axs[1, i].set_axis_off()
    plt.draw()
    plt.pause(0.001)

# TRAINING AND VALIDATION SPLIT
from sklearn.model_selection import train_test_split

if NORMALIZE:
    imgs = normalize_images(imgs)

if SMOOTH:
    masks = smooth_masks(masks)

if M_NOISE:
    masks = noise_masks(masks)


train_data, val_data, train_masks, val_masks = train_test_split(imgs, masks, test_size=0.10, shuffle=True, random_state=42)


# AUGMENT IMAGES AND MASKS
if AUGMENT:
    train_data, train_masks = augment_data_random(train_data, train_masks)
    val_data, val_masks = augment_data_random(val_data, val_masks)
    data, masks = augment_data_random(imgs, masks)
else:
    data, masks = imgs, masks


# WEIGHT COMPUTATION FOR BALANCE

#like sklearn wieght class balanced
def balanced_weights(masks):
    labels_1 = 0
    labels_0 = 0
    n_samples = masks.shape[0] * masks.shape[1] * masks.shape[2]
    for mask in masks:
        h, w = mask.shape
        label_1 = np.sum(mask, axis = None)
        label_0 = h * w - label_1
        labels_1 += label_1
        labels_0 += label_0
    return  np.array([(n_samples / (2 * labels_0)), (n_samples / (2 * labels_1))])

class_weights = balanced_weights(train_masks)
class_weights


# UTIL FUNCTIONS

# some constants
PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing
               # road



# MODEL



# TRANSFORM FOR ONLINE AUGMENTATION OF TRAINING SET
#train_transform = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Rotate(interpolation=1, border_mode=cv2.BORDER_REFLECT, p=0.5)])


# DATASET
def np_to_tensor(x, device):
    # allocates tensors from np.arrays
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)

def tensor_to_np(x, device):
    if device == 'cpu':
        return x.numpy()
    else:
        return x.cpu().numpy()

class ImageDataset(torch.utils.data.Dataset):
    # dataset class that deals with loading the data and making it available by
    # index.

    def __init__(self, imgs, masks, path, device, resize_to=(400, 400), transform=None, train_transform=False):
        self.path = path
        self.device = device
        self.resize_to = resize_to
        self.transform = transform
        #self.x, self.y, self.n_samples = None, None, None
        #self._load_data()
        self.x, self.y = imgs, masks
        self.n_samples = len(self.x)
        if self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.train_transform = train_transform



    def _load_data(self):  # not very scalable, but good enough for now
        self.x = load_all_from_path(os.path.join(self.path, 'images'))
        self.y = load_all_from_path(os.path.join(self.path, 'groundtruth'))
        if self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
            self.x = np.stack([cv2.resize(img, dsize=self.resize_to) for img in self.x], 0)
            self.y = np.stack([cv2.resize(mask, dsize=self.resize_to) for mask in self.y], 0)
        self.x = np.moveaxis(self.x, -1, 1)  # pytorch works with CHW format instead of HWC
        self.n_samples = len(self.x)


    def _preprocess(self, x, y):
        # to keep things simple we will not apply transformations to each sample,
        # but it would be a very good idea to look into preprocessing
        if self.transform is not None:
            transformed = self.transform(image=x, mask=y)
            x = transformed["image"]
            y = transformed["mask"]
        return x, y


    def __getitem__(self, item):
        return self._preprocess(np_to_tensor(self.x[item], self.device), np_to_tensor(self.y[[item]], self.device))


    def __len__(self):
        return self.n_samples

def show_val_samples(x, y, y_hat, segmentation=False):
    # training callback to show predictions on validation set
    imgs_to_draw = min(5, len(x))
    if x.shape[-2:] == y.shape[-2:]:  # segmentation
        plt.close()
        fig, axs = plt.subplots(3, imgs_to_draw, figsize=(18.5, 12))
        if imgs_to_draw <= 1:
            axs[0].imshow(np.moveaxis(x.squeeze(), 0, -1))
            axs[1].imshow(y_hat.squeeze())
            axs[2].imshow(y.squeeze())
            axs[0].set_title('Sample')
            axs[1].set_title('Predicted')
            axs[2].set_title('True')
            axs[0].set_axis_off()
            axs[1].set_axis_off()
            axs[2].set_axis_off()
        else:
          for i in range(imgs_to_draw):
              axs[0, i].imshow(np.moveaxis(x[i], 0, -1))
              axs[1, i].imshow(np.concatenate([np.moveaxis(y_hat[i], 0, -1)] * 3, -1))
              axs[2, i].imshow(np.concatenate([np.moveaxis(y[i], 0, -1)]*3, -1))
              axs[0, i].set_title(f'Sample {i}')
              axs[1, i].set_title(f'Predicted {i}')
              axs[2, i].set_title(f'True {i}')
              axs[0, i].set_axis_off()
              axs[1, i].set_axis_off()
              axs[2, i].set_axis_off()
    else:  # classification
        plt.close()
        fig, axs = plt.subplots(1, imgs_to_draw, figsize=(18.5, 6))
        for i in range(imgs_to_draw):
            axs[i].imshow(np.moveaxis(x[i], 0, -1))
            axs[i].set_title(f'True: {np.round(y[i]).item()}; Predicted: {np.round(y_hat[i]).item()}')
            axs[i].set_axis_off()
    plt_draw()


#COMPOSITE TRAINING 1
def comptrain_1(train_dataloader, eval_dataloader, modelG, loss_fn_r, metric_fns, optimizerG, n_epochs, patience, batch_size):
    # training loop
    logdir = './tensorboard/net'
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch

    max_val_patch_acc = 0
    epochs_no_improve = 0

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        # initialize metric list
        metrics = {'loss': [], 'val_loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
        # training
        modelG.train()
        for (x, y) in pbar:
            optimizerG.zero_grad()  # zero out gradients
            gc.collect()
            y_hat = modelG(x)  # forward pass
            loss = loss_fn_r(y_hat, y)
            loss.backward()  # backward pass
            optimizerG.step()  # optimize weights

            # log partial metrics
            metrics['loss'].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(y_hat, y).item())
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        # validation
        modelG.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                y_hat = modelG(x)  # forward pass
                loss = loss_fn_r(y_hat, y)
                
                # log partial metrics
                metrics['val_loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y).item())

        
        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
          writer.add_scalar(k, v, epoch)
        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))
        
        curr_lr = optimizerG.param_groups[0]['lr']
        print('learning rate')
        print(curr_lr)
        

        #early stopping
        val_patch_acc = sum(metrics['val_patch_acc']) / len(metrics['val_patch_acc'])
        if val_patch_acc > max_val_patch_acc:
          #Saving the model
          #if min_loss > loss.item():
              #min_loss = loss.item()
          best_model = copy.deepcopy(modelG.state_dict())
          print('saving model')
          epochs_no_improve = 0
          max_val_patch_acc = val_patch_acc

        else:
          epochs_no_improve += 1
          print('val_patch_acc not improving')
          # Check early stopping condition
          if epochs_no_improve == patience:
              print('Early stopping!' )
              modelG.load_state_dict(best_model)
              break



#COMPOSITE TRAINING 2
def d_train(train_dataloader, eval_dataloader, modelD, loss_fn, loss_fn_d, repetitions, metric_fns, optimizerD, n_d_epochs, batch_size):
    # training loop
    
    torch.cuda.empty_cache()
    real_label = torch.ones(batch_size, dtype = torch.float).to(device) # label of ones to desginate that the input is "real" (ground truth)
    fake_label = torch.zeros(batch_size, dtype = torch.float).to(device)  # label of zeroes to designate that the input is "fake" (generated by GAN)
    
    for epoch in range(n_d_epochs): #loop over the dataset multiple times
    
        # initialize metric list
        metrics = {'discriminator loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []
    
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_d_epochs}')
        modelD.train()
        
        for (x, y) in pbar:
            
            batch_size = x.size()[0]
            
            # TRAIN DISCRIMINATOR
            modelD.train()
            real_label = torch.ones(batch_size, dtype = torch.float).to(device) # label of ones to desginate that the input is "real" (ground truth)
            fake_label = -real_label
            
            real_label_2 = 2*real_label
            fake_label_2 = 2*fake_label
            
            

            # TRAIN DISCRIMINATOR
            for rep in range(repetitions):
                optimizerD.zero_grad()  # zero out gradients
    
                generated_seg = modelG(x)  # segmentations generated by our GAN #generated_seg = torch.randn((2, 1, 384, 384)).to(device)#
                false_masks = np_to_tensor(cv2.resize(random.choice(masks), dsize=IMG_SIZE), device).view(1, 1, IMG_SIDE, IMG_SIDE)
                for batch in range(batch_size-1):
                    false_masks = torch.cat((false_masks, np_to_tensor(cv2.resize(random.choice(masks), dsize=IMG_SIZE), device).view(1, 1, IMG_SIDE, IMG_SIDE)))
                input = torch.cat((generated_seg.detach(), y, false_masks))
                img = torch.cat((x, x, x))
                
                input = torch.cat((input, img), 1)
                
    
                pred = modelD(input).view(-1)
                label = torch.cat((real_label, fake_label, real_label))
    
                loss = loss_fn_d(pred, label)
                loss.backward()
            
                optimizerD.step()  # optimize weights
                
                for p in modelD.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # log partial metrics
            metrics['discriminator loss'].append(loss.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(generated_seg, y).item())
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

# ALTERNATE TRAINING
def alt_train(train_dataloader, eval_dataloader, modelD, modelG, loss_fn, loss_fn_d, repetitions, metric_fns, optimizerD, optimizerG, n_gan_epochs, batch_size):
    # training loop
    logdir = './tensorboard/net'
    writer = SummaryWriter(logdir)  # tensorboard writer (can also log images)

    history = {}  # collects metrics at the end of each epoch
    
    torch.cuda.empty_cache()
    real_label = torch.ones(batch_size, dtype = torch.float).to(device) # label of ones to desginate that the input is "real" (ground truth)
    fake_label = torch.zeros(batch_size, dtype = torch.float).to(device)  # label of zeroes to designate that the input is "fake" (generated by GAN)
    

    for epoch in range(n_gan_epochs):  # loop over the dataset multiple times
        
        # initialize metric list
        metrics = {'discriminator loss': [], 'generator loss': []}
        for k, _ in metric_fns.items():
            metrics[k] = []
            metrics['val_'+k] = []

        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_gan_epochs}')
        # training
        
        for (x, y) in pbar:
            
            batch_size = x.size()[0]
            
            modelD.train()
            real_label = torch.ones(batch_size, dtype = torch.float).to(device) # label of ones to desginate that the input is "real" (ground truth)
            fake_label = -real_label
            generated_seg = modelG(x)  # segmentations generated by our GAN #generated_seg = torch.randn((2, 1, 384, 384)).to(device)#

            loss = None
            # TRAIN DISCRIMINATOR
            for rep in range(repetitions):
    

                false_masks = np_to_tensor(cv2.resize(random.choice(masks), dsize=IMG_SIZE), device).view(1, 1, IMG_SIDE, IMG_SIDE)
                for batch in range(batch_size-1):
                    false_masks = torch.cat((false_masks, np_to_tensor(cv2.resize(random.choice(masks), dsize=IMG_SIZE), device).view(1, 1, IMG_SIDE, IMG_SIDE)))
                if random.randint(0, 100) < 20: #one-sided label flipping
                    input = torch.cat((generated_seg.detach(), y, false_masks))
                    label = torch.cat((fake_label, fake_label, fake_label))
                    img = torch.cat((x, x, x))
                else:
                    input = torch.cat((generated_seg.detach(), y, false_masks))
                    label = torch.cat((fake_label, real_label, fake_label))
                    img = torch.cat((x, x, x))
                
                input = torch.cat((input, img), 1)
                
                optimizerD.zero_grad()  # zero out gradients
    
                pred = modelD(input).view(-1)
                
    
                loss = loss_fn_d(pred, label)
                loss.backward()
            
                optimizerD.step()  # optimize weights
                
                for p in modelD.parameters():
                    p.data.clamp_(-0.01, 0.01)
            
            modelG.train()

            # TRAIN GENERATOR
            optimizerG.zero_grad()  # zero out gradients
            gc.collect()
            
            input = torch.cat((modelG(x), x), 1)
            new_output = modelD(input).view(-1)  # new output from our updated discriminator
            lossG = loss_fn_d(new_output, real_label)  # we look at the loss with 'real' labels as target, as we now want to train in an adversarial fashion.
            lossG.backward()  # backward pass

            optimizerG.step()  #optimize weights
            
            
            """optimizerG.zero_grad()  # zero out gradients
            gc.collect()
            y_hat = modelG(x)  # forward pass
            lossG2 = loss_fn(y_hat, y)
            lossG2.backward()  # backward pass
            optimizerG.step()  # optimize weights"""



            # log partial metrics
            if loss is not None:
                metrics['discriminator loss'].append(loss.item())
            else:
                metrics['discriminator loss'].append(0)
            metrics['generator loss'].append(lossG.item())
            for k, fn in metric_fns.items():
                metrics[k].append(fn(generated_seg, y).item())
            pbar.set_postfix({k: sum(v)/len(v) for k, v in metrics.items() if len(v) > 0})

        # validation
        modelG.eval()
        with torch.no_grad():  # do not keep track of gradients
            for (x, y) in eval_dataloader:
                y_hat = modelG(x)  # forward pass
                loss = loss_fn(y_hat, y)
                
                # log partial metrics
                metrics['discriminator loss'].append(loss.item())
                for k, fn in metric_fns.items():
                    metrics['val_'+k].append(fn(y_hat, y).item())

        
        # summarize metrics, log to tensorboard and display
        history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
        for k, v in history[epoch].items():
          writer.add_scalar(k, v, epoch)
        print(' '.join(['\t- '+str(k)+' = '+str(v)+'\n ' for (k, v) in history[epoch].items()]))
        
        curr_lr = optimizerG.param_groups[0]['lr']
        print('learning rate')
        print(curr_lr)


    print('Finished Training')
    # plot loss curves
    plt.plot([v['discriminator loss'] for k, v in history.items()], label='Discriminator Loss')
    plt.plot([v['generator loss'] for k, v in history.items()], label='Generator Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


# ADDING CHANNELS
def add_channels(imgs):
    new_imgs = []
    for i in imgs:
        img = np.moveaxis(i, 0, -1)
        Lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        Lab[:, :, 0] = Lab[:, :, 0] / 100.
        Lab[:, :, 1] = (Lab[:, :, 1] + 128.) / 255.
        Lab[:, :, 2] = (Lab[:, :, 2] + 128.) / 255.
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        g8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, t75 = cv2.threshold(gray, 0.75, 1, cv2.THRESH_BINARY)
        _, t50 = cv2.threshold(gray, 0.50, 1, cv2.THRESH_BINARY)
        _, t33 = cv2.threshold(gray, 0.33, 1, cv2.THRESH_BINARY)
        tOt = cv2.threshold(g8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1].astype(np.float32) / 255.
        tAd = cv2.adaptiveThreshold(g8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2).astype(np.float32) / 255.
        edges = cv2.Canny(g8, 100, 200, L2gradient=True).astype(np.float32) / 255.
        new_img = np.stack((gray, t75, t50, t33, tOt, tAd, edges), axis=2)
        new_img = np.concatenate((img, Lab, new_img), axis=2)
        new_img = np.moveaxis(new_img, -1, 0)
        new_imgs.append(new_img)
    return np.asarray(new_imgs)

# U-NET
class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch
    # normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, padding_mode = 'reflect'),
                                    nn.PReLU(),
                                    nn.BatchNorm2d(out_ch),
                                    nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, padding_mode = 'reflect'),
                                    nn.PReLU())

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])])  # encoder blocks
        self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, 2, 2) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # deconvolution
        self.dec_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # decoder blocks
        self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], 1, 1), nn.Sigmoid()) # 1x1 convolution for producing the output

    def forward(self, x):
        # add channels
        # x = np_to_tensor(add_channels(tensor_to_np(x, x.device.type)), x.device.type)
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(self.dec_blocks, self.upconvs, enc_features[::-1]):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.head(x)  # reduce to 1 channel


# MU-NET
class MBlock(nn.Module):
    # a repeating structure composed of two convolutional layers with batch
    # normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1),
                                    nn.PReLU(),
                                    nn.BatchNorm2d(out_ch),
                                    nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                    nn.PReLU())

    def forward(self, x):
        return self.block(x)

class MupBlock(nn.Module):
    # a repeating structure composed of two convolutional layers with batch
    # normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=4, stride=2, padding=1),
                                    nn.PReLU(),
                                    nn.BatchNorm2d(out_ch),
                                    nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                    nn.PReLU())

    def forward(self, x):
        return self.block(x)

class MUNet(nn.Module):  # MU-Net without max-pooling layers
    def __init__(self, chs=(3,64,128,256,512,1024,2048), chs_2=(4096,2048,1024,512,256,128,64,1)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs_2  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList([MBlock(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])])  # encoder blocks
        #self.upconvs = nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, 2, 2) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # deconvolution
        self.dec_blocks = nn.ModuleList([MupBlock(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-2], dec_chs[2:])])  # decoder blocks
        self.sig = nn.Tanh()

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
        # decode
        for block, feature in zip(self.dec_blocks, enc_features[::-1]):
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.sig(x)  # reduce to 1 channel

class Discriminator(nn.Module):  # inspired by 
    def __init__(self):
        super(Discriminator, self).__init__()
        channels = 64
        self.main = nn.Sequential(
            # input is 4 x 384 x 384
            nn.Conv2d(4, channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (channels) x 192 x 192
            nn.Conv2d(channels, channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (channels*2) x 96 x 96
            nn.Conv2d(channels * 2, channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (channels*4) x 48 x 48
            nn.Conv2d(channels * 4, channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (channels*8) x 24 x 24
            nn.Conv2d(channels * 8, channels *16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (channels*16) x 12 x 12
            nn.Conv2d(channels * 16, channels * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (channels*32) x 6 x 6
            nn.Conv2d(channels * 32, channels * 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(channels * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (channels*64) x 3 x 3
            nn.Conv2d(channels * 64, 1, 3, 1, 0, bias=False)
        )

    def forward(self, input):
        input = self.main(input)
        #print(input.size())
        return input


# METRICS

# some constants
PATCH_SIZE = 16  # pixels per side of square patches
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing
               # road
def patch_accuracy_fn(y_hat, y):
    # computes accuracy weighted by patches (metric used on Kaggle for evaluation)
    h_patches = y.shape[-2] // PATCH_SIZE
    w_patches = y.shape[-1] // PATCH_SIZE
    patches_hat = y_hat.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    patches = y.reshape(-1, 1, h_patches, PATCH_SIZE, w_patches, PATCH_SIZE).mean((-1, -3)) > CUTOFF
    return (patches == patches_hat).float().mean()

def accuracy_fn(y_hat, y):
    # computes classification accuracy
    return (y_hat.round() == y.round()).float().mean()


# TRAIN
def BCELoss_class_weighted(weights):

    def loss(input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    return loss

def Wasserstein_loss():
    
    def loss(fake, real):
        return torch.mean(fake*real)
    
    return loss

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1: # for all convolutional layers
    nn.init.normal_(m.weight.data, 0.00, 0.02)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# reshape the image to simplify the handling of skip connections and maxpooling
train_dataset = ImageDataset(train_data,train_masks, 'training', device, resize_to=IMG_SIZE)
val_dataset = ImageDataset(val_data,val_masks,'validation', device, resize_to=IMG_SIZE)
#dataset = ImageDataset(data, masks, 'training', device, resize_to=(384, 384))

BATCH_SIZE = 1
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

modelD = Discriminator().to(device)
modelD.apply(weights_init)
modelG = MUNet().to(device)
modelG.apply(weights_init)

#for class imbalance
weights= np_to_tensor(class_weights,device)

loss_fn = BCELoss_class_weighted(weights = weights)
loss_fn_D = Wasserstein_loss()

metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}


optimizer = torch.optim.Adam(modelG.parameters(), lr = 0.0002)

#for learning rate updtate
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5)

n_epochs = GENERATOR_EPOCHS
n_d_epochs = DISCRIMINATOR_EPOCHS
n_gan_epochs = GAN_EPOCHS

patience_early_stop = 50

comptrain_1(train_dataloader, val_dataloader, modelG, loss_fn, metric_fns, optimizer, n_epochs, patience_early_stop, BATCH_SIZE)
optimizerG = torch.optim.RMSprop(modelG.parameters(), lr = 0.0002)
optimizerD = torch.optim.RMSprop(modelD.parameters(), lr = 0.0002)
d_train(train_dataloader, val_dataloader, modelD, loss_fn, loss_fn_D, 5, metric_fns, optimizerD, n_d_epochs, BATCH_SIZE)
alt_train(train_dataloader, val_dataloader, modelD, modelG, loss_fn, loss_fn_D, 5, metric_fns, optimizerD, optimizerG, n_gan_epochs, BATCH_SIZE)

#save model
torch.save(modelG.state_dict(), 'pth/u_net.pth')


# PREDICT ON TEST AND CREATE SUBMISSION

# paths
train_path = 'training'
val_path = 'validation'
test_path = 'test_images/test_images'

def create_submission(labels, test_filenames, submission_filename):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn, patch_array in zip(sorted(test_filenames), test_pred):
            img_number = int(re.search(r"\d+", fn).group(0))
            for i in range(patch_array.shape[0]):
                for j in range(patch_array.shape[1]):
                    f.write("{:03d}_{}_{},{}\n".format(img_number, j * PATCH_SIZE, i * PATCH_SIZE, int(patch_array[i, j])))
                    
def create_masks(filenames, pred, folder_name):
    for fn, patch_array in zip(sorted(filenames), pred):
        img_number = int(re.search(r"\d+", fn).group(0))
        Image.fromarray(patch_array*255).convert('L').save(folder_name + '/pred_' + '%.3d' % img_number + '.png')
        
# predictions on the training set
train_filenames = (glob('training/images/*.png'))
train_images = load_all_from_path('training/images')
if NORMALIZE:
    train_images = normalize_images(train_images)
train_images = np.stack([cv2.resize(img, dsize=IMG_SIZE) for img in train_images], 0)
train_images = np_to_tensor(np.moveaxis(train_images, -1, 1), device)
train_pred = [modelG(img).detach().cpu().numpy() for img in train_images.unsqueeze(1)]
train_pred = np.concatenate(train_pred, 0)
train_pred = np.moveaxis(train_pred, 1, -1)
train_pred = np.stack([cv2.resize(img, dsize=(400, 400)) for img in train_pred], 0)
create_masks(train_filenames, train_pred, 'prediction_masks_train')

# predict on test set
test_filenames = (glob(test_path + '/*.png'))
test_images = load_all_from_path(test_path)
if NORMALIZE:
    test_images = normalize_images(test_images)
batch_size = test_images.shape[0]
size = test_images.shape[1:3]

# we also need to resize the test images.  This might not be the best ideas
# depending on their spatial resolution.
test_images = np.stack([cv2.resize(img, dsize=IMG_SIZE) for img in test_images], 0)
test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
test_pred = [modelG(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
test_pred = np.concatenate(test_pred, 0)
test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)  # resize to original shape
create_masks(test_filenames, test_pred, 'prediction_masks')

# now compute labels
test_pred = test_pred.reshape((-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
test_pred = np.moveaxis(test_pred, 2, 3)
test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
create_submission(test_pred, test_filenames, submission_filename='unet_submission.csv')

torch.cuda.empty_cache()