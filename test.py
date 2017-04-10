#! /usr/bin/env python

######################################################################################
### Name: zedBuoy.py
### Author: Chandler Lagarde
### Date: 12/17/2016
###
###
### Purpose: Detect all buoy centroids publishing ROS messages and saving the video
### Notes: Added delay for zed to wake up, adding aspect ratio check 
### Rev: 1
######################################################################################

import datetime
# from boat_vision.msg import buoy
# import rospy
import cv2
import numpy as np
import imutils
import time

# date and time at time of running script, turn it into string, edit one value to _, replace other colons with -
now = datetime.datetime.now()
now = str(now)[0:19]
now = now[:10] + '_' + now[11:]
now = now.replace(":", "-")

# connect and capture video from cameras
zed = cv2.VideoCapture(0)
time.sleep(1.0)

# # filepath and filename
# zedStr = "/home/ubuntu/boat_ws/src/boat_vision/vidSaves/zed_%s.mp4" % (now)
# # define the codec (H264 for best results) and create VideoWriter objects
# fourcc = cv2.VideoWriter_fourcc(*'H264')
# # res zed set to image cropping (if image cropping change, change the video to fit)
# zedOut = cv2.VideoWriter(zedStr, fourcc, 20.0, (2560, 742))


while True:
	# read each frame of each camera
	ret, frame = zed.read()

	# crop the image, left camera only, and top portion cut out
	# frame = frame[500:99999, 0:2560]

	# image centroid
	height, width = frame.shape[:2]
	CHeight = height / 2
	CWidth = width / 2
	imgCent = CWidth, CHeight
	# print("image centroid: width = {}, height = {}".format(CWidth, CHeight))
	# draw the image's centerpoint circle
	cv2.circle(frame, imgCent, 5, (255, 0, 255), -1)

	# define hsv values for frame
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	# define lower and upper bounds of red
	lRed = np.array([0, 75, 75])
	uRed = np.array([10, 255, 255])
	lRed2 = np.array([165, 75, 75])
	uRed2 = np.array([179, 255, 255])
	# define lower and upper bounds of the other colors (red loops around in hsv)
	lGreen = np.array([50, 50, 50])
	uGreen = np.array([90, 255, 255])
	lBlue = np.array([105, 50, 50])
	uBlue = np.array([135, 255, 255])
	lYellow = np.array([20, 50, 50])
	uYellow = np.array([40, 255, 255])
	lWhite = np.array([0, 0, 0])
	uWhite = np.array([0, 0, 255])



	# define red mask, erode and dilate them
	maskR1 = cv2.inRange(hsv, lRed, uRed)
	maskR2 = cv2.inRange(hsv, lRed2, uRed2)
	maskR = cv2.addWeighted(maskR1, 1.0, maskR2, 1.0, 0)
	maskR = cv2.erode(maskR, None, iterations=2)
	maskR = cv2.dilate(maskR, None, iterations=2)	
	# define green mask
	maskG = cv2.inRange(hsv, lGreen, uGreen)
	maskG = cv2.erode(maskG, None, iterations=2)
	maskG = cv2.dilate(maskG, None, iterations=2)
	# define blue mask
	maskB = cv2.inRange(hsv, lBlue, uBlue)
	maskB = cv2.erode(maskB, None, iterations=2)
	maskB = cv2.dilate(maskB, None, iterations=2)
	# define yellow mask
	maskY = cv2.inRange(hsv, lYellow, uYellow)
	maskY = cv2.erode(maskY, None, iterations=2)
	maskY = cv2.dilate(maskY, None, iterations=2)
	# define white mask
	maskW = cv2.inRange(hsv, lWhite, uWhite)
	maskW = cv2.erode(maskW, None, iterations=2)
	maskW = cv2.dilate(maskW, None, iterations=2)
	# set mask as list of masks (RGBYW)
	mask = [maskR, maskG, maskB, maskY, maskW]



	# define res for each color and thier respective mask values
	resR = cv2.bitwise_and(frame, frame, mask = mask[0])
	resG = cv2.bitwise_and(frame, frame, mask = mask[1])
	resB = cv2.bitwise_and(frame, frame, mask = mask[2])
	resY = cv2.bitwise_and(frame, frame, mask = mask[3])
	resW = cv2.bitwise_and(frame, frame, mask = mask[4])
	# define res as list of res (RGBWY)
	res = [resR, resG, resB, resY, resW]



	# contour detection, looking for buoy aspect ratio, colors, max contours, etc.
	# initialize the difference array and counter i at 0
	difference = []
	i = 0
	# loop over each array value within array res (loop for each color's res)
	# obtain contours of each mask
	for array in res:
		print('i = {}'.format(i))
		gray = cv2.cvtColor(res[i], cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (9, 9), 0)
		edged = cv2.Canny(blurred, 50, 150)

		# find contours in the edged and initialize the current
		# (x, y) center of the buoy
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)[-2]
		center = 'none'


		####################################################################################
		# possibly add removing contours (pyimagesearch tutorial) that aren't approx 4
		####################################################################################


		# only proceed if at least one contour was found
		if len(cnts) > 0:
			z = []
			# loop over the contours
			for c in cnts:
				# approximate the contour
				peri = cv2.arcLength(c, True)
				approx = cv2.approxPolyDP(c, 0.01 * peri, True)

				# ensure that the approximated contour is "roughly" rectangular
				if len(approx) >= 4 and len(approx) <= 8:
					# compute the bounding box of the approximated contour and
					# use the bounding box to compute the aspect ratio
					(x, y, w, h) = cv2.boundingRect(approx)
					aspectRatio = w / float(h)

					# compute the solidity of the original contour
					area = cv2.contourArea(c)
					hullArea = cv2.contourArea(cv2.convexHull(c))
					solidity = area / float(hullArea)

					# compute whether or not the width and height, solidity, and
					# aspect ratio of the contour falls within appropriate bounds
					keepDims = w > 15 and h > 30
					keepSolidity = solidity > 0.6
					keepAspectRatio = aspectRatio >= 0.15 and aspectRatio <= 0.5

					# ensure that the contour passes all our tests
					if keepDims and keepSolidity and keepAspectRatio:
						z.append(c)
			
			# print('c = {}'.format(c))
			print('z = {}'.format(z))
			if z != []:
				# find the largest contour in the mask, then use it to 
				# compute the minimum enclosing rectangle and centroid
				z = max(z, key=cv2.contourArea)
		
				rect = cv2.minAreaRect(z)
				# obtain height and width of minAreaRect
				sqArea = rect[1]
				h = sqArea[0]
				w = sqArea[1]
				# convert to image distances
				box = cv2.boxPoints(rect)
				box = np.int0(box)


				M = cv2.moments(z)
				area = M["m00"]
				# avoid division by 0 in center calculation
				if area > 0:
					cX = int(M["m10"] / area)
					cY = int(M["m01"] / area)
					center = cX, cY

					# # minimum size height and width to return difference of centroids
					# if h > 80 and w > 40:
					# draw a bounding box
					cv2.drawContours(frame, [box], 0, (255, 0, 255), 2)
					# draw the centerpoint circle
					cv2.circle(frame, center, 5, (255, 0, 255), -1)
					# put color detected by the center
					color = ['red', 'green', 'blue', 'yellow', 'white']
					cv2.putText(frame, color[i], (cX - 20, cY - 20),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
					# return the first value of center, the x-value
					difference.append(CWidth - center[0])
					i += 1
			else:
				difference.append(99999.99)
				i += 1
		else:
			difference.append(99999.99)
			i += 1


	print(difference)


	# # ROS subscribe
	# pub = rospy.Publisher('fore_buoy_centroids', buoy, queue_size=10)
	# rospy.init_node('fore_buoy_centroid_talker', anonymous=True)
	# rate = rospy.Rate(10) # 10hz
	# # ROS publish
	# pub.publish(diff)
	# rospy.loginfo(diff)
	# rate.sleep()

	# # write frames as video in defined locations for each camera
	# zedOut.write(frame)
	cv2.imshow('frame', frame)



	key = cv2.waitKey(1) & 0xFF 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

# When everything done, release the capture
zed.release()
# zedOut.release()

cv2.destroyAllWindows()


