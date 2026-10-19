import cv2
import os
import shutil
import numpy as np

### note in BGR order
def black2color(black_im, output_path):
	img = np.zeros((black_im.shape[0], black_im.shape[1], 3), dtype=np.uint8)
	for i in range(black_im.shape[0]):
		for j in range(black_im.shape[1]):
			num = black_im[i][j]
			if num == 0:
				img[i][j] = [0, 0, 0]
			elif num == 1:
				img[i][j] = [255, 0, 0]
			elif num == 2:
				img[i][j] = [0, 0, 255]
			elif num == 3:
				img[i][j] = [255, 0, 255]
			elif num == 4:
				img[i][j] = [0, 255, 0]
			elif num == 5:
				img[i][j] = [150, 150, 0]
			elif num == 6:
				img[i][j] = [255, 0, 150]
			elif num == 7:
				img[i][j] = [0, 255, 255]
			elif num == 8:
				img[i][j] = [255, 150, 255]
			elif num == 9:
				img[i][j] = [0, 150, 255]
			elif num == 10:
				img[i][j] = [80, 210, 150]
			elif num == 11:
				img[i][j] = [100, 150, 100]
			elif num == 12:
				img[i][j] = [150, 255, 0]
			elif num == 13:
				img[i][j] = [120, 130, 220]
			elif num == 14:
				img[i][j] = [150, 0, 255]
			elif num == 15:
				img[i][j] = [255, 150, 0]
			elif num == 16:
				img[i][j] = [255, 170, 150]
			elif num == 17:
				img[i][j] = [180, 120, 50]
			elif num == 18:
				img[i][j] = [0, 50, 180]
			elif num == 19:
				img[i][j] = [100, 160, 255]
			elif num == 20:
				img[i][j] = [100, 0, 150]
			elif num == 21:
				img[i][j] = [180, 225, 200]
			elif num == 22:
				img[i][j] = [150, 255, 150]
			elif num == 23:
				img[i][j] = [80, 190, 210]
			elif num == 24:
				img[i][j] = [150, 0, 0]
			elif num == 25:
				img[i][j] = [0, 150, 0]
			elif num == 26:
				img[i][j] = [150, 0, 150]
			elif num == 27:
				img[i][j] = [100, 255, 255]
			elif num == 28:
				img[i][j] = [100, 0, 255]
			elif num == 29:
				img[i][j] = [255, 255, 0]
	cv2.imwrite(output_path, img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
def clear_output_folder(path):
	if os.path.isdir(path):
		shutil.rmtree(path)
	os.mkdir(path)

def b2c(black_folder, color_folder):
	clear_output_folder(color_folder)
	image_names = os.listdir(black_folder)
	for name in image_names:
		print('processing: ' + name)
		black_im = cv2.imread(black_folder + name, cv2.IMREAD_UNCHANGED)
		black2color(black_im, color_folder + name)
