import cv2
import os
import shutil
import numpy as np
import scipy.misc as misc
from PIL import Image
import distance

#cell_titles = ['Lines','Poles','Sign 100','Sign 110','Sign 60','Metal barriers','Trees','Rumble strips','Roads','Concrete barriers','Curvature signs','Guide posts','Merge lanes','Defects','Warning signs','Railway signs','Median concrete','Bicycle paths','Flexiposts','Road work signs','Median grass','Houses and buildings','Signal signs','Pedestrian crossing signs','Pedestrian ccrossings','School zone warning signs','Speed humps','Speed hump signs','Roundabout signs'] 

def name2class(name):
	if 'line' in name:
		return 1
	if 'pole' in name:
		return 2
	if '100' in name:
		return 3
	if '110' in name:
		return 4
	if '60' in name:
		return 5
	if 'metal' in name and 'bar' in name:
		return 6
	if 'tree' in name:
		return 7
	if 'rumble' in name:
		return 8
	if 'road' in name and 'work' not in name:
		return 9
	if 'concrete' in name and 'bar' in name:
		return 10
	if 'curvature' in name:
		return 11
	if 'guide' in name:
		return 12
	if 'merge' in name:
		return 13
	if 'defect' in name:
		return 14
	if 'warning' in name:
		return 15
	if 'railway' in name:
		return 16
	if 'median' in name and 'concrete' in name:
		return 17
	if 'bicycle' in name:
		return 18
	if 'flexipost' in name:
		return 19
	if 'road' in name and 'work' in name:
		return 20
	if 'grass' in name:
		return 21
	if 'building' in name:
		return 22
	if 'signal' in name:
		return 23
	if 'pedestrian' in name and 'sign' in name:
		return 24
	if 'pedestrian' in name and 'sign' not in name:
		return 25
	if 'school' in name:
		return 26
	if 'hump' in name and 'sign' not in name:
		return 27
	if 'hump' in name and 'sign' in name:
		return 28
	if 'roundabout' in name:
		return 29
	return -1
	
def clear_folder(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.mkdir(path)
		
def fuse_seg(path, output_path, cut_flag, keep_bottom, keep_left, keep_right):
	clear_folder(output_path)
	folders = os.listdir(path)
	segs = []
	for f in folders:
		if '_seg' in f and name2class(f) != -1:
			segs.append(f)
	ims = os.listdir(path+segs[0]+'/')
	im_size = np.array(Image.open(path+segs[0]+'/'+ims[0]))
	# for images
	for i in range(len(ims)):
		temp_im = np.zeros([im_size.shape[0], im_size.shape[1]], dtype = np.uint8)
		# for folders
		for j in range(len(segs)):
			im = np.array(Image.open(path+segs[j]+'/'+ims[i]))
			idx = im==1
			temp_im[idx] = name2class(segs[j])
		# remove far points
		if cut_flag:
			for x in range(temp_im.shape[0]):
				if x > int(keep_left*temp_im.shape[0]) and x < int((1-keep_right)*temp_im.shape[0]):
					for y in range(temp_im.shape[0]):
						if y < int((1-keep_bottom)*temp_im.shape[0]):
							temp_im[y][x] = 0
		misc.imsave(output_path+ims[i], temp_im)
