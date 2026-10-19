import numpy as np
import os
import distance
import TensorflowUtils as utils
import black2color as b2c
import shutil
import fuse_models
import black2color as b2c
import test_image_reader


def clear_folder(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.mkdir(path)

nvgx_path = '10A_1.nvgx'
model_folder_path = 'trained_models\\'
frame_path = 'output\\ori\\'
output_path = 'output\\'
'''
keep_bottom, keep_left, keep_right = 0.6, 0.3, 0.3
test_image_reader._resize_images(800, 800, frame_path, 'output\\ori_resized\\')

models = os.listdir(model_folder_path)
for m in models:
	print('testing model: ' + m + ' ...')
	model_path = model_folder_path + m + '\\'
	seg_path = output_path + m + '_seg\\'
	os.system('python segmenter.py --model_path ' + model_path + ' --input_path ' + frame_path + ' --output_path ' + seg_path)
print('fusing models ...')
fuse_models.fuse_seg(output_path, 'output\\fuse_seg\\', True, keep_bottom, keep_left, keep_right)
print('generating excel ...')
'''

distance.generate_excel(nvgx_path, 'output\\ori_resized\\', 'output\\fuse_seg\\', 30, output_path)
'''
print('generating visualization ...')
b2c.b2c('output\\fuse_seg\\', 'output\\fuse_vis\\')
'''


