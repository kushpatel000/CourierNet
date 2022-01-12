#!/usr/bin/env python3

'''
Motion Detect from this recipe:
	https://www.pyimagesearch.com/2015/06/01/home-surveillance-and-motion-detection-with-the-raspberry-pi-python-and-opencv/
'''

from picamera.array import PiRGBArray
from picamera import PiCamera

import argparse
import warnings
from datetime import datetime
import cv2
import imutils
import json
import time
import io
import os

import tflite_runtime.interpreter as tflite
import numpy as np
import sys

import paho.mqtt.client as mqtt


def fatal_error(msg):
	print(msg)
	exit(1)

def load_config():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-c","--conf", required=True,help="path to JSON config file")
	args = vars(ap.parse_args())

	# filter warnings, load configuration
	# no going to initialize the dropbox client
	warnings.filterwarnings('ignore')
	conf = json.load(open(args["conf"]))

	# check for required keywords:
	keys = [
		"min_upload_seconds",
		"min_motion_frames",
		"camera_warmup_time",
		"delta_thresh",
		"resolution",
		"fps",
		"min_area",
		"model",
		"label_file",
		"broker_address"
	]
	kill_flag = False
	print("[INFO] Checking configuration file for required keys...")
	for key in keys:
		if key in conf:
			print(f"  [\u2713] {key}" )
		else:
			print(f"  [x] {key}" )
			kill_flag = True
	if kill_flag:
		fatal_error("Configuration file is missing keys")

	return conf

def load_labels(path):
	if not path:
		fatal_error("Label path not supplied")
	if not os.path.exists(path):
		fatal_error("Label file not found")

	labels = open(path,'r').read().split()
	return labels

def load_interpreter(model_path="./models/0.0/0.0.tflite",nt=None):
	interpreter = tflite.Interpreter(
		model_path  = conf['model'],
		num_threads = conf['num_threads']
	)
	interpreter.allocate_tensors()

	# information about the model
	input_details  = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	# input_height, input width = input_details[0]['shape'][1:3]

	details = {
		"input_dims":    input_details[0]['shape'][1:3], # height, width
		"input_index":   input_details[0]['index'],
		"output_index": output_details[0]['index']
	}
	return interpreter, details

def make_capture_dirs():
	if not os.path.exists("./captures"):
		os.mkdir("./captures")
	if not os.path.exists("./grays"):
		os.mkdir("./grays")


def init_camera():
# initialize the camera and grab a reference to the raw camera capture
	camera = PiCamera()
	camera.resolution = tuple( conf['resolution'] )
	camera.framerate  = conf['fps']
	rawCapture = PiRGBArray(camera, size=tuple(conf['resolution']))
	return camera, rawCapture

def compute_results(image, model, model_details):
	# preprocess the input image
	input_data = cv2.resize(image, model_details['input_dims'])
	input_data = np.expand_dims(input_data,axis=0)
	input_data = input_data.astype(np.float32)
	# supply image to model
	model.set_tensor( model_details["input_index"], input_data )
	# process image
	time_start = time.perf_counter()
	model.invoke()
	time_end = time.perf_counter()
	comp_time = time_end - time_start
	# gather output
	output_data = model.get_tensor( model_details['output_index'] )
	confidences = np.squeeze(output_data)
	idx = np.argmax(confidences)

	return idx, confidences, comp_time

def export_image(image, folder, timestamp, label, confidence, comp_time):
	# replace above code to output frames always
	ts = timestamp.strftime("%Y_%m_%d_%H_%M_%S_%f")
	name = "capture_" + ts + '.jpg'
	path = os.path.join( "./", folder, name )

	# output image jpg
	cv2.imwrite(path,image)
	# write results to log
	# with open("./captures/captures.log" ,'a') as wrtr:
	with open( os.path.join( "./", folder, "captures.log" ), 'a' ) as wrtr:
		wrtr.write(f"{ts:26s} {label:6s} {confidence*100:0.2f} {comp_time:f}\n")

if __name__ == '__main__':
	# load configuration file
	conf = load_config()
	# load labels
	labels = load_labels( conf['label_file'] )
	# create model/Interpreter
	model, model_details = load_interpreter( model_path=conf['model'], nt=conf['num_threads'] )

	# load and start camera
	camera, rawCapture = init_camera()

	# create capture folders if not existing
	make_capture_dirs()

	# allow the camera to warmup, then initialize the average frame,
	# last uploaded timestamp, and frame motion counter
	print("[INFO] warming up...")
	time.sleep(conf["camera_warmup_time"])
	avg = None
	lastOutput = datetime.now()
	motionCounter = 0


	# Loop over frames directly from Rasberry Pi video stream
	# capture frames from camera
	for f in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
		# grab the raw NumPy array representing the image
		frame = f.array
		timestamp = datetime.now()
		motion_detected = False

		# resize the frame, convert it to grayscale, and blur it
		frame = imutils.resize(frame, width=500)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (21,21), 0)


		if avg is None:
			print("[INFO] starting background capture...")
			avg = gray.copy().astype("float")
			rawCapture.truncate(0)
			continue

		# Accumulate the weighted average between current frame and
		# previous frames, then compute the difference between the current
		# frame and running average
		cv2.accumulateWeighted(gray, avg, 0.5)
		frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))


		# threshold the delta image, dilate the threshold image to fill
		# in the holes, then find contours on threshold image
		thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		# loop over the contours
		for c in cnts:
			# if the contour is too small, ignore it
			if cv2.contourArea(c) < conf['min_area']:
				continue

			# compute the bounding box for the contour, draw it on the frame,
			# and update the text
			# (x,y,w,h) = cv2.boundingRect(c)
			# cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
			motion_detected = True

		# draw the text and timestamp on the frame
		# ts = timestamp.strftime("%A %d %B %Y %H:%M:%S")
		# cv2.putText(frame, f"Motion Detected", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
		# cv2.putText(frame, ts, (10,frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)


		# check to see if motion detected is significant
		# if so, process
		if motion_detected:
			# check to see if enough time has passed between uploads
			if (timestamp - lastOutput).seconds >= conf["min_upload_seconds"]:
				# increment the motion counter
				motionCounter += 1

				# check to see if the number of frames with consistent motion is high enough
				if motionCounter >= conf["min_motion_frames"]:
					# process image against model
					idx, confidences, comp_time = compute_results(frame, model, model_details)

					# export raw image
					export_image(frame, "captures", timestamp, labels[idx], confidences[idx], comp_time)
					# export gray difference
					export_image(frameDelta, "grays", timestamp, labels[idx], confidences[idx], comp_time)

					# push notification to smart home
					if labels[idx] != "None":

						msg = json.dumps({
						    "carrier":labels[idx],
						    "confidence": f'{confidences[idx]*100:7.2f}',
						    # "timestamp": timestamp.strftime("%Y_%m_%d_%H_%M_%S_%f")
							"timestamp": timestamp.isoformat()
						})
						client = mqtt.Client("RPI_Courier")
						client.connect(conf['broker_address'])
						client.publish("CourierNet/delivery",payload=msg)


					# update the last uploaded time stamp and reset the motion
					lastOutput = timestamp
					motionCounter = 0

		# otherwise, set motion counter back to zero
		else:
			motionCounter = 0



		# # check to see if the frames should be displayed to screen
		# if conf["show_video"]:
		# 	# display the security feed
		# 	cv2.imshow("Security Feed",frame)
		# 	key = cv2.waitKey(1) & 0xFF
		#
		# 	# if the `q` key is pressed, break from the loop
		# 	if key == ord('q'):
		# 		break

		# clear the stream in preparation for the next frame
		rawCapture.truncate(0)
