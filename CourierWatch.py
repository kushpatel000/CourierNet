

import os
import P3picam
import picamera
from datetime import datetime



if __name__ == '__main__':
	motionState = False
	output_dir  = './captures'

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	
	while True:
		motionState = P3picam.motion() 
		# print(motionState)

		if motionState:
			time = datetime.now()
			datestr = f"{time.year:04d}_{time.month:02d}_{time.day:02d}_{time.hour:02d}_{time.minute:02d}_{time.second:02d}"
			imagepath = os.path.join(output_dir,f"capture_{datestr}.jpg")
			
			with picamera.PiCamera() as camera:
				camera.resolution = (160,160)
				camera.capture(imagepath)

			# print(imagepath)