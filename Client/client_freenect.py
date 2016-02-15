import cv
#import easygui
from freenect import sync_get_depth as get_depth, sync_get_video as get_video
import frame_convert
from time import sleep
import threading
import numpy as np
from client_util import *

analyze_interval = 1000 # analyze image every certain interval (in milliseconds)
VIDEO_WINSIZE = (640, 480)
host,port = '127.0.0.1', 9001
send_lock = threading.Lock()
arr = []
depth = []

def display():
    global arr, depth, send_lock
    while (True):
        if send_lock.acquire():
            (depth,_),(rgb,_)=get_depth(),get_video()
            #arr = np.dstack((rgb[:,:,0].T,rgb[:,:,1].T,rgb[:,:,2].T))
            arr=rgb
            d3 = np.dstack((depth,depth,depth)).astype(np.uint8)
            da = np.hstack((d3,rgb))
            cv.ShowImage('both',cv.fromarray(np.array(da[::2,::2,::-1])))
            send_lock.release()
        cv.WaitKey(5)
    
# Main game loop
def main():
    global arr, depth, send_lock
    t = threading.Thread(target=display)
    t.daemon = True
    t.start()
    while (True):
        if send_lock.acquire(): 
            print arr.shape 
            print depth.shape 
            connectAndSendArr3d(host, port, arr)            
            send_lock.release()
            sleep(1)

if (__name__ == "__main__"):
	main()
