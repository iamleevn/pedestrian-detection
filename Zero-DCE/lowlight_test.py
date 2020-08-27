import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time



def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)



	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)

	end_time = (time.time() - start)
	print(end_time)

	image_path = image_path.replace('test_data','result')
	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

	torchvision.utils.save_image(enhanced_image, result_path)

def mylowlight(image_path, output_dir):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)

	data_lowlight = (np.asarray(data_lowlight)/255.0)

	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool().cuda()
	DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)

	end_time = (time.time() - start)
	print(end_time)

	result_path = os.path.join(outputdir, 'zeroDCE_' + os.path.basename(iamge_path))
	torchvision.utils.save_image(enhanced_image, result_path)


def get_args():
    parser = argparse.ArgumentParser('ZeroDCE enhance low light images')
    parser.add_argument('-input_dir', type=str,
                        default='../demo_images/',
                        help='path contains input images', dest='inputdir')
    parser.add_argument('-output_dir', type=str,
                        default='../result_images/',
                        help='path contains detection results', dest='outputdir')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
	args = get_args()
	file_path = args.inputdir
	output_dir = args.outputdir
# test_images
	with torch.no_grad():
		file_list = []
		for r, d, f in os.walk(file_path):
			for i in f:
				file_list.append(os.path.join(r, i))

		for image in file_list:
			print(image)
			mylowlight(image)
