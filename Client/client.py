import pygame
import thread
#import easygui
from pykinect import nui
import numpy as np
from client_util import *

analyze_interval = 1000 # analyze image every certain interval (in milliseconds)
VIDEO_WINSIZE = (640, 480)
host,port = '127.0.0.1', 6666
screen = None
screen_lock = thread.allocate()
dosend = False
s=None

def video_frame_ready(frame):
	with screen_lock:
		frame.image.copy_bits(screen._pixels_address)
		arr3d = pygame.surfarray.pixels3d(screen)
		
		global dosend
		if dosend:
			data = connectAndSendArr3d(host, port, arr3d)
			print arr3d.shape
			dosend = False
			if data:
				for itm in data:
					
					box = tuple(num*100 for num in itm[1:])
					print "box",box
					pygame.draw.rect(screen,(0,0,0),box,0)
				

		pygame.display.update()

def main():
	# 's' key for save, 'q' key for quit
	"""Initialize and run the game"""
	pygame.init()
	global s
	s=0
	global arr
	arr = []
	pygame.time.set_timer(pygame.USEREVENT,analyze_interval)
	# Initialize PyGame
	global screen,dosend
	screen = pygame.display.set_mode(VIDEO_WINSIZE, 0, 32)

	pygame.display.set_caption("PyKinect Video Example")

	angle = 0
	with nui.Runtime() as kinect:
		kinect.video_frame_ready += video_frame_ready
		kinect.video_stream.open(nui.ImageStreamType.Video, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Color)		
		# Main game loop
		while (True):
			event = pygame.event.wait()
			if event.type == pygame.USEREVENT:
				dosend = True
			if event.type == pygame.KEYUP and event.key==pygame.K_q:
				break
			if (event == pygame.QUIT):
				break
			
if (__name__ == "__main__"):
	main()