import cv2
import math
import numpy as py

#import easygui

# from freenect import sync_get_depth as get_depth, sync_get_video as get_video


# import frame_convert
from time import sleep
import threading
import numpy as np
from client_util import *

analyze_interval = 1000 # analyze image every certain interval (in milliseconds)
VIDEO_WINSIZE = (640, 480)
host,port = '127.0.0.1', 6666
# send_lock = threading.Lock()

depth = []

def get_map(height,width):
    FOV = 3.141592654; # FOV of the fisheye, eg: 180 degrees

    # Polar angles
    theta = 3.14159265 * (np.array(xrange(width))*1.0/width - 0.5); # -pi/2 to pi/2
    phi = 3.14159265 * (np.array(xrange(height))*1.0/height - 0.5); # -pi/2 to pi/2

    # Vector in 3D space
    psph_x = np.outer(np.cos(phi),np.sin(theta));
    psph_y = np.outer(np.cos(phi),np.cos(theta));
    psph_z = np.outer(np.sin(phi),np.ones(theta.shape));

    # Calculate fisheye angle and radius
    theta = np.arctan2(psph_z,psph_x);
    phi = np.arctan2(np.sqrt(psph_x*psph_x+psph_z*psph_z),psph_y);
    r = 0.9*width * phi / FOV;

    # Pixel in fisheye space
    pfish_x = 0.5 * width + r * np.cos(theta);
    pfish_y = 0.5 * width + r * np.sin(theta);

    map_x=pfish_x.astype('float32')
    map_y=pfish_y.astype('float32')
    return map_x,map_y

def process(img,map_x,map_y):
    H,W=map_x.shape[:2]
    dst = np.zeros((H,2*W,3),dtype=np.uint8)
    dst[:,0:W,:]=cv2.remap(img[:,0:W,:],map_x,map_y,cv2.INTER_LINEAR)
    dst[:,W:,:]=cv2.remap(img[:,W:,:],map_x,map_y,cv2.INTER_LINEAR)
    return dst

# def display():
#     global arr, depth, send_lock
#     while (True):
#         if send_lock.acquire():
#             (depth,_),(arr,_)=get_depth(),get_video()
#             d3 = np.dstack((depth,depth,depth)).astype(np.uint8)
#             da = np.hstack((d3,arr))
#             cv.ShowImage('both',cv.fromarray(np.array(da[::2,::2,::-1])))
#             send_lock.release()
#         cv.WaitKey(5)

# Main game loop
def main():
    # global arr, depth, send_lock
    # t = threading.Thread(target=display)
    # t.daemon = True
    # t.start()
    # while (True):
    #     if send_lock.acquire():
    #         print arr.shape # note that in client_util, arr is transposed
    #         print depth.shape
    #         connectAndSendArr3d(host, port, arr)
    #         send_lock.release()
    #         sleep(1)

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720);

    ret, frame = cap.read()
    H, W = frame.shape[:2]
    print H,W
    map_x, map_y = get_map(H, W/2)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #dst = process(frame,map_x,map_y)
        dst = frame

        resized = cv2.resize(dst,VIDEO_WINSIZE)

        so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        so.connect((host, port))
        data = connectAndSendArr3d(so, resized)
        so.close()
        height, width = dst.shape[:2]
        if data:
            for obj in data:
                a, x, y, w, h = obj
                print a[0], x, y, w, h
                cv2.putText(dst, str(a[0]),(10,500), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

                left  = max(1, int((x-w/2)*width))
                right = min(width, int((x+w/2)*width))
                top   = max(1, int((y-h/2)*height))
                bot   = min(height, int((y+h/2)*height))

                cv2.rectangle(dst, (left, top), (right, bot), (0,255,0), 3)
        cv2.imshow('frame',dst)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if (__name__ == "__main__"):
    main()