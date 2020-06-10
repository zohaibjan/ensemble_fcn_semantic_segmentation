import numpy as np
import scipy.misc as misc
import os
import cv2
import shutil

def clear_folder(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.mkdir(path)

def _read_images(path):
	files = os.listdir(path)
	images = np.array([_transform(path + filename) for filename in files])
	print (images.shape)
	return images, files

def _transform(filename):
	image = cv2.imread(filename)
	if len(image.shape) < 3:  # make sure images are of shape(h,w,3)
		image = np.array([image for i in range(3)])
	return np.array(image)
	
def resize(im, w, h):
	dim = (w, h)
	resized = cv2.resize(im, dim, interpolation = cv2.INTER_AREA)
	if len(resized.shape) > 2 and resized.shape[2] == 4:
		resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)
	return resized
	
def _resize_images(w, h,input_path, output_path):
	clear_folder(output_path)
	ims, names = _read_images(input_path)
	for i in range(ims.shape[0]):
		im = resize(ims[i], w, h)
		print(output_path + names[i])
		cv2.imwrite(output_path + names[i], im)
		
def _read_images_fcn(path, w, h):
	files = os.listdir(path)
	images = np.array([_transform_fcn(path + filename, w, h) for filename in files])
	print (images.shape)
	return images, files

def _transform_fcn(filename, w, h):
	image = misc.imread(filename)
	if len(image.shape) < 3:  # make sure images are of shape(h,w,3)
		image = np.array([image for i in range(3)])
	image = misc.imresize(image, [w, h], interp='nearest')
	return np.array(image)
	