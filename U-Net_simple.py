import os
import sys
import re
import cv2
import copy
import gc
import random
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import albumentations as A
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
from tqdm import tqdm

#REPRODUCIBILITY
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

#GET RUNTIME PARAMETERS
args = {
	"n_epochs": 100, 
	"batch_size": 4, 
	"valid_ratio": 0.10, 
	"augment": False, 
	"early_stop": False, 
	"use_class_weights": False, 
	"update_lr": False, 
	"draw": False, 
	"model_filename": "u_net", 
	"test_output": ""
	}

for arg in sys.argv[1:]:
	if arg[:9] == "n_epochs=":
		args["n_epochs"] = int(arg[9:])
	elif arg[:11] == "batch_size=":
		args["n_epochs"] = int(arg[11:])
	elif arg[:12] == "valid_ratio=":
		args["valid_ratio"] = float(arg[12:])
	elif arg == "augment":
		args["augment"] = True
	elif arg == "early_stop":
		args["early_stop"] = True
	elif arg == "use_class_weights":
		args["use_class_weights"] = True
	elif arg == "update_lr":
		args["update_lr"] = True
	elif arg == "draw":
		args["draw"] = True
	elif arg[:15] == "model_filename=":
		args["model_filename"] = arg[15:]
	elif arg[:12] == "test_output=":
		args["test_output"] = arg[12:]

def plt_draw():
	if args["draw"]:
		plt.draw()
		plt.pause(0.001)
	return

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load data into np array function
def load_all_from_path(path):
	# loads all HxW gpngs contained in path as a 4D np.array of shape (n_images,
	# H, W, 3)
	# images are loaded as floats with values in the interval [0., 1.]
	return np.stack([np.array(Image.open(f)) for f in sorted(glob(path + '/*.png'))]).astype(np.float32) / 255.

#load data
imgs = load_all_from_path(os.path.join('training', 'images'))
masks = load_all_from_path(os.path.join('training', 'groundtruth'))

# PRE TRAIN DATA AUGMENTATION

#RANDOM ROTATE IMAGE IN STEP OF 90, THEN EITHER FLIP VERTICAL OR HORIZONTAL
#AUGMENT x9
def augment_data_random(imgs,masks):
 
	new_imgs = []
	new_masks = []
	for x, y in zip(imgs, masks):
		transformed1 = A.Rotate(limit = (0, 90), interpolation=1, border_mode=cv2.BORDER_REFLECT, p= 1)(image = x,mask =  y)
		transformed2 = A.Rotate(limit = (90, 180), interpolation=1, border_mode=cv2.BORDER_REFLECT, p= 1)(image = x,mask =  y)
		transformed3 = A.Rotate(limit = (180, 270), interpolation=1, border_mode=cv2.BORDER_REFLECT, p= 1)(image = x,mask =  y)
		transformed4 = A.Rotate(limit = (270, 360), interpolation=1, border_mode=cv2.BORDER_REFLECT, p= 1)(image = x,mask = y)

		transformed5 = A.HorizontalFlip(p=1)(image = transformed1['image'],mask =   transformed1['mask'])
		transformed6 = A.VerticalFlip(p=1)(image = transformed2['image'],mask =   transformed2['mask'])
		transformed7 = A.HorizontalFlip(p=1)(image = transformed3['image'],mask =   transformed3['mask'])
		transformed8 = A.VerticalFlip(p=1)(image = transformed4['image'],mask =   transformed4['mask'])

		new_imgs.append(x)
		new_imgs.append(transformed1['image'])
		new_imgs.append(transformed2['image'])
		new_imgs.append(transformed3['image'])
		new_imgs.append(transformed4['image'])
		new_imgs.append(transformed5['image'])
		new_imgs.append(transformed6['image'])
		new_imgs.append(transformed7['image'])
		new_imgs.append(transformed8['image'])

		new_masks.append(y)
		new_masks.append(transformed1['mask'])
		new_masks.append(transformed2['mask'])
		new_masks.append(transformed3['mask'])
		new_masks.append(transformed4['mask'])
		new_masks.append(transformed5['mask'])
		new_masks.append(transformed6['mask'])
		new_masks.append(transformed7['mask'])
		new_masks.append(transformed8['mask'])

	return  np.asarray(new_imgs), np.asarray(new_masks)

#AUGMENTED IMAGES AND MASKS
if args["augment"]:
	imgs, masks = augment_data_random(imgs, masks)

# TRAINING AND VALIDATION SPLIT
train_data, val_data, train_masks, val_masks = train_test_split(imgs, masks, test_size=args["valid_ratio"], shuffle=True)

# WEIGHT COMPUTATION FOR BALANCE
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

if args["use_class_weights"]:
	class_weights = balanced_weights(train_masks)

# UTIL FUNCTIONS

# some constants
PATCH_SIZE = 16  # pixels per side of square patches
CUTOFF = 0.25  # minimum average brightness for a mask patch to be classified as containing
               # road
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

# RETURNS LIST OF PATCHES AND LABELS FROM IMAGES
def image_to_patches(images, masks=None):
	# takes in a 4D np.array containing images and (optionally) a 4D np.array
	# containing the segmentation masks
	# returns a 4D np.array with an ordered sequence of patches extracted from the
	# image and (optionally) a np.array containing labels
	n_images = images.shape[0]  # number of images
	h, w = images.shape[1:3]  # shape of images
	assert (h % PATCH_SIZE) + (w % PATCH_SIZE) == 0  # make sure images can be patched exactly

	h_patches = h // PATCH_SIZE
	w_patches = w // PATCH_SIZE
	patches = images.reshape((n_images, h_patches, PATCH_SIZE, h_patches, PATCH_SIZE, -1))
	patches = np.moveaxis(patches, 2, 3)
	patches = patches.reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
	if masks is None:
		return patches

	masks = masks.reshape((n_images, h_patches, PATCH_SIZE, h_patches, PATCH_SIZE, -1))
	masks = np.moveaxis(masks, 2, 3)
	labels = np.mean(masks, (-1, -2, -3)) > CUTOFF  # compute labels
	labels = labels.reshape(-1).astype(np.float32)
	return patches, labels

def show_patched_image(patches, labels, h_patches=25, w_patches=25):
	# reorders a set of patches in their original 2D shape and visualizes them
	plt.close()
	fig, axs = plt.subplots(h_patches, w_patches, figsize=(18.5, 18.5))
	for i, (p, l) in enumerate(zip(patches, labels)):
		# the np.maximum operation paints patches labeled as road red
		axs[i // w_patches, i % w_patches].imshow(np.maximum(p, np.array([l.item(), 0., 0.])))
		axs[i // w_patches, i % w_patches].set_axis_off()
	plt_draw()

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

# DATASET

class ImageDataset(torch.utils.data.Dataset):
	# dataset class that deals with loading the data and making it available by index.

	def __init__(self, imgs, masks, path, device, use_patches=True, resize_to=(400, 400), transform=None):
		self.path = path
		self.device = device
		self.use_patches = use_patches
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

	def _load_data(self):  # not very scalable, but good enough for now
		self.x = load_all_from_path(os.path.join(self.path, 'images'))
		self.y = load_all_from_path(os.path.join(self.path, 'groundtruth'))
		if self.use_patches:  # split each image into patches
			self.x, self.y = image_to_patches(self.x, self.y)
		elif self.resize_to != (self.x.shape[1], self.x.shape[2]):  # resize images
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
		for i in range(imgs_to_draw):
			axs[0, i].imshow(np.moveaxis(x[i], 0, -1))
			axs[1, i].imshow(np.concatenate([np.moveaxis(y_hat[i], 0, -1)] * 3, -1))
			axs[2, i].imshow(np.concatenate([np.moveaxis(y[i], 0, -1)] * 3, -1))
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


# TRAIN FUNCTION
def train(train_dataloader, eval_dataloader, model, loss_fn, metric_fns, optimizer, n_epochs, patience, scheduler):
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
			metrics['val_' + k] = []

		pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{n_epochs}')
		# training
		model.train()
		for (x, y) in pbar:			
			optimizer.zero_grad()  # zero out gradients
			gc.collect()
			y_hat = model(x)  # forward pass
			loss = loss_fn(y_hat, y)
			loss.backward()  # backward pass
			optimizer.step()  # optimize weights

			# log partial metrics
			metrics['loss'].append(loss.item())
			for k, fn in metric_fns.items():
				metrics[k].append(fn(y_hat, y).item())
			pbar.set_postfix({k: sum(v) / len(v) for k, v in metrics.items() if len(v) > 0})

		# validation
		model.eval()
		with torch.no_grad():  # do not keep track of gradients
			for (x, y) in eval_dataloader:
				y_hat = model(x)  # forward pass
				loss = loss_fn(y_hat, y)

				# log partial metrics
				metrics['val_loss'].append(loss.item())
				for k, fn in metric_fns.items():
					metrics['val_' + k].append(fn(y_hat, y).item())

		# summarize metrics, log to tensorboard and display
		history[epoch] = {k: sum(v) / len(v) for k, v in metrics.items()}
		for k, v in history[epoch].items():
			writer.add_scalar(k, v, epoch)
		metrics_str = ' '.join(['\t- ' + str(k) + ' = ' + str(v) + '\n ' for (k, v) in history[epoch].items()])
		print(metrics_str)
		#with open("pth/metrics.txt", "w") as f:
		#	f.write(metrics_str)
		show_val_samples(x.detach().cpu().numpy(), y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())

		#scheduler for learning rate update
		if args["update_lr"]:
			scheduler.step(sum(metrics['val_loss']) / len(metrics['val_loss']))
			curr_lr = optimizer.param_groups[0]['lr']
			print('learning rate')
			print(curr_lr)

		#early stopping
		if args["early_stop"]:
			val_patch_acc = sum(metrics['val_patch_acc']) / len(metrics['val_patch_acc'])
			if val_patch_acc > max_val_patch_acc:
				#Saving the model
				#if min_loss > loss.item():
					#min_loss = loss.item()
				best_model = copy.deepcopy(model.state_dict())
				print('saving model')
				epochs_no_improve = 0
				max_val_patch_acc = val_patch_acc

			else:
				epochs_no_improve += 1
				print('val_patch_acc not improving')
				# Check early stopping condition
				if epochs_no_improve == patience:
					print('Early stopping!')
					model.load_state_dict(best_model)
					break

	print('Finished Training')
	# plot loss curves
	plt.close()
	plt.plot([v['loss'] for k, v in history.items()], label='Training Loss')
	plt.plot([v['val_loss'] for k, v in history.items()], label='Validation Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend()
	plt_draw()


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
		self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, padding_mode = 'zeros'),
									nn.PReLU(),
									nn.BatchNorm2d(out_ch),
									nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, padding_mode = 'zeros'),
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


# METRICS

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
		input = torch.clamp(input,min=1e-7,max=1 - 1e-7)
		bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
		return torch.mean(bce)

	return loss

# reshape the image to simplify the handling of skip connections and maxpooling
train_dataset = ImageDataset(train_data,train_masks, 'training', device, use_patches=False, resize_to=(384, 384))
val_dataset = ImageDataset(val_data,val_masks,'validation', device, use_patches=False, resize_to=(384, 384))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=True)

model = UNet().to(device)

#for class imbalance
if args["use_class_weights"]:
	weights = np_to_tensor(class_weights,device)
	loss_fn = BCELoss_class_weighted(weights = weights)
else:
	loss_fn = nn.BCELoss()

metric_fns = {'acc': accuracy_fn, 'patch_acc': patch_accuracy_fn}

optimizer = torch.optim.Adam(model.parameters())

#for learning rate updtate
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = 5)

patience_early_stop = 10

if os.path.isfile('pth/u_net.pth'):
	model.load_state_dict(torch.load('pth/u_net.pth'))

train(train_dataloader, val_dataloader, model, loss_fn, metric_fns, optimizer, args["n_epochs"], patience_early_stop, scheduler)

#save model
torch.save(model.state_dict(), args["model_filename"] + ".pth")

# PREDICT ON TEST AND CREATE SUBMISSION

# paths
train_path = 'training'
val_path = 'validation'
test_path = 'test_images/test_images'

def create_submission(labels, test_filenames, submission_filename):
	test_path = 'test_images/test_images'
	with open(submission_filename, 'w') as f:
		f.write('id,prediction\n')
		for fn, patch_array in zip(sorted(test_filenames), test_pred):
			img_number = int(re.search(r"\d+", fn).group(0))
			for i in range(patch_array.shape[0]):
				for j in range(patch_array.shape[1]):
					f.write("{:03d}_{}_{},{}\n".format(img_number, j * PATCH_SIZE, i * PATCH_SIZE, int(patch_array[i, j])))

# predict on test set
test_filenames = (glob(test_path + '/*.png'))
test_images = load_all_from_path(test_path)
batch_size = test_images.shape[0]
size = test_images.shape[1:3]

# we also need to resize the test images.  This might not be the best ideas
# depending on their spatial resolution.
test_images = np.stack([cv2.resize(img, dsize=(384, 384)) for img in test_images], 0)
test_images = np_to_tensor(np.moveaxis(test_images, -1, 1), device)
test_pred = [model(t).detach().cpu().numpy() for t in test_images.unsqueeze(1)]
test_pred = np.concatenate(test_pred, 0)
test_pred = np.moveaxis(test_pred, 1, -1)  # CHW to HWC
test_pred = np.stack([cv2.resize(img, dsize=size) for img in test_pred], 0)  # resize to original shape

if args["test_output"]:
	if not os.path.exists(args["test_output"]):
	 os.makedirs(args["test_output"])

	for i in range(len(test_filenames)):
		Image.fromarray(np.uint8(test_pred[i, :, :] * 255), 'L').save(args["test_output"] + "/" + os.path.basename(test_filenames[i]))

# now compute labels
test_pred = test_pred.reshape((-1, size[0] // PATCH_SIZE, PATCH_SIZE, size[0] // PATCH_SIZE, PATCH_SIZE))
test_pred = np.moveaxis(test_pred, 2, 3)
test_pred = np.round(np.mean(test_pred, (-1, -2)) > CUTOFF)
create_submission(test_pred, test_filenames, submission_filename=args["model_filename"]+".csv")