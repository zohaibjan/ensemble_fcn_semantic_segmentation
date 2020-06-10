# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:23:46 2020

@author: zj-19
"""

import cv2
import numpy as np
import vector
import matplotlib.pyplot as plt
from numpy.linalg import inv


# GLobal params
nwindows = 10
margin=110 
minpix=20

# Source points for homography
src = np.array([[500, 50], [686, 41], [1078, 253], [231, 259]], dtype="float32")

# Destination points for homography
dst = np.array([[50, 0], [250, 0], [250, 500], [0, 500]], dtype="float32")

#Homography 
H_matrix = cv2.getPerspectiveTransform(src, dst)
Hinv = inv(H_matrix)

def correct_dist(initial_img):
	# Intrnsic camera matrix
	k = [[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [0.00000000e+00, 1.14818221e+03, 3.86046312e+02],
		 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
	k = np.array(k)
	# Distortion Matrix
	dist = [[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]]
	dist = np.array(dist)
	img_2 = cv2.undistort(initial_img, k, dist, None, k)

	return img_2

# Turn prediction for lanes
def turn_predict(image_center, right_lane_pos, left_lane_pos):
    lane_center = left_lane_pos + (right_lane_pos - left_lane_pos)/2
    
    if (lane_center - image_center < 0):
        return ("Turning left")
    elif (lane_center - image_center < 8):
        return ("straight")
    else:
    	return ("Turning right")


# Process image and homography operations
def image_preprocessing(img):

	crop_img = img[420:720, 40:1280, :]  # To get the region of interest

	undist_img = correct_dist(crop_img)

	hsl_img = cv2.cvtColor(undist_img, cv2.COLOR_BGR2HLS)

	# To seperate out Yellow colored lanes
	lower_mask_yellow = np.array([20, 120, 80], dtype='uint8')
	upper_mask_yellow = np.array([45, 200, 255], dtype='uint8')
	mask_yellow = cv2.inRange(hsl_img, lower_mask_yellow, upper_mask_yellow)

	yellow_detect = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_yellow).astype(np.uint8)

	# To seperate out White colored lanes
	lower_mask_white = np.array([0, 200, 0], dtype='uint8')
	upper_mask_white = np.array([255, 255, 255], dtype='uint8')
	mask_white = cv2.inRange(hsl_img, lower_mask_white, upper_mask_white)

	white_detect = cv2.bitwise_and(hsl_img, hsl_img, mask=mask_white).astype(np.uint8)

	# Combine both
	lanes = cv2.bitwise_or(yellow_detect, white_detect)

	new_lanes = cv2.cvtColor(lanes, cv2.COLOR_HLS2BGR)

	final = cv2.cvtColor(new_lanes, cv2.COLOR_BGR2GRAY)

	# Filter noise
	img_blur = cv2.bilateralFilter(final, 9, 120, 100)

	# Apply edge detection
	img_edge = cv2.Canny(img_blur, 100, 200)

	# Apply homography to get bird's view
	new_img = cv2.warpPerspective(img_edge, H_matrix, (300, 600))
	
	# Use histogram to get pixels with max Y axis value
	histogram = np.sum(new_img, axis=0)
	out_img = np.dstack((new_img,new_img,new_img))*255
	
	midpoint = np.int(histogram.shape[0]/2)
	
	# Compute the left and right max pixels
	leftx_ = np.argmax(histogram[:midpoint])
	rightx_ = np.argmax(histogram[midpoint:]) + midpoint
	#print(leftx_base)
	
	left_lane_pos = leftx_
	right_lane_pos = rightx_
	image_center = int(new_img.shape[1]/2)

	# Use the lane pixels to predict the turn
	prediction = turn_predict(image_center, right_lane_pos, left_lane_pos)	
			
	window_height = np.int(new_img.shape[0]/nwindows)

	nonzero = new_img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	
	# Update current position for each window
	leftx_p = leftx_
	rightx_p = rightx_
	
	# left and right lane pixel indices
	left_lane_inds = []
	right_lane_inds = []
	
	# Step through the windows one by one
	for window in range(nwindows):
		# Identify window boundaries in x and y (and right and left)
		win_y_down = new_img.shape[0] - (window+1)*window_height
		win_y_up = new_img.shape[0] - window*window_height
		win_x_left_down = leftx_p - margin
		win_x_left_up = leftx_p + margin
		win_x_right_down = rightx_p - margin
		win_x_right_up = rightx_p + margin
		
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_left_down) & (nonzerox < win_x_left_up)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_down) & (nonzeroy < win_y_up) & (nonzerox >= win_x_right_down) & (nonzerox < win_x_right_up)).nonzero()[0]
		
		# Append these indices to the list
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
		
		# If found > minpix pixels, move to next window
		if len(good_left_inds) > minpix:
			leftx_p = np.int(np.mean(nonzerox[good_left_inds]))
		if len(good_right_inds) > minpix:        
			rightx_p = np.int(np.mean(nonzerox[good_right_inds]))
	
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 


	if leftx.size == 0 or rightx.size == 0 or lefty.size == 0 or righty.size == 0:
		return

	left_fit = np.polyfit(lefty, leftx, 2)
	right_fit = np.polyfit(righty, rightx, 2)
	
	ploty = np.linspace(0, new_img.shape[0]-1, new_img.shape[0] )

	out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
	out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
	
	# Fit a second order polynomial to each
	left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
	
	# Extract points from fit
	left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx,
                              ploty])))])

	image_center = img_edge.shape[0]/2

	
	pts = np.hstack((left_line_pts, right_line_pts))
	pts = np.array(pts, dtype=np.int32)

	color_blend = np.zeros_like(img).astype(np.uint8)
	cv2.fillPoly(color_blend, pts, (0,255, 0))
	
	# Project the image back to the orignal coordinates
	newwarp = cv2.warpPerspective(color_blend, inv(H_matrix), (crop_img.shape[1], crop_img.shape[0]))
	result = cv2.addWeighted(crop_img, 1, newwarp, 0.5, 0)
	cv2.putText(result, prediction, (200, 50),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,255,0),2, cv2.LINE_AA)

	# Show the output image
	cv2.imshow('result-image',result)


def lanePredict(f, frame_path):
    # Read the frames
    cap = cv2.VideoCapture('project_video.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # print(fps)
