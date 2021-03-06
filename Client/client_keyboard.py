import pygame
import thread
#import easygui
from pykinect import nui
import numpy as np
from client_util import *
import encode

VIDEO_WINSIZE = (640, 480)
host,port = '127.0.0.1', 7777
screen = None
screen_lock = thread.allocate()
dosave = False
dosend = False

s=None
def video_frame_ready(frame):
	with screen_lock:
		frame.image.copy_bits(screen._pixels_address)
		arr3d = pygame.surfarray.pixels3d(screen)
		
		global dosave,dosend
		if dosend:
			connectAndSendArr3d(host, port, arr3d)
			#//for i in range(3):
			#	np.savetxt('rgb'+str(i)+'.txt',255-arr3d[:,:,i])
			print arr3d.shape
			dosend = False
		if dosave:
			for i in range(3):
				np.savetxt('rgb'+str(i)+'.txt',arr3d[:,:,i].astype(np.uint8))
			print arr3d.shape
			dosave = False
		
		pygame.display.update()

def main():
	# 's' key for save, 'q' key for quit
	"""Initialize and run the game"""
	pygame.init()
	global s
	s=0
	global arr
	arr = []

	# Initialize PyGame
	global screen, dosave,dosend
	screen = pygame.display.set_mode(VIDEO_WINSIZE, 0, 32)

	pygame.display.set_caption("PyKinect Video Example")

	angle = 0
	with nui.Runtime() as kinect:
		kinect.video_frame_ready += video_frame_ready
		kinect.video_stream.open(nui.ImageStreamType.Video, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Color)		
		# Main game loop
		while (True):
			event = pygame.event.wait()
			if event.type == pygame.KEYUP and event.key==pygame.K_s:
				dosend = True
				print('Send:')
			if event.type == pygame.KEYUP and event.key==pygame.K_w:
				dosave = True
				print('Save:')
			if event.type == pygame.KEYUP and event.key==pygame.K_q:
				break
			if (event == pygame.QUIT):
				break
if (__name__ == "__main__"):
	main()