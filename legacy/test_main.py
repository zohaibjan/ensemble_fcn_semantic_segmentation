import numpy as np
import os
import distance
import TensorflowUtils as utils
import black2color as b2c
import shutil
import fuse_models
import black2color as b2c
import test_image_reader
import video_utils


nvgx_path = '10A_1.nvgx'
output_path = 'output1/'

#video_utils.extract_frame_from_folder('video/', output_path+'ori_resized/', 800)
print('fusing models ...')
#fuse_models.fuse_seg(output_path, 'output/fuse_seg1/')
print('generating excel ...')
distance.generate_excel(nvgx_path, 'output1/ori_resized/', 'output1/fuse_seg/', 30, output_path)
print('generating visualization ...')
#b2c.b2c('output/fuse_seg1/', 'output/fuse_vis1/')


