import numpy as np
import base64
import socket
import time

# This class does the following:
#   1. Convert from box representation (obj,x,y,w,h) to 3d representation (x,y,z,obj)
#   2. Send signal to Unity to trigger the 3D sound
#   3. Make sure there is a cool down period for each class of the object
# cam_type: "fisheye" or "logitech"
# note: the y direction for yolo is downwards, but for Unity it is upwards, so there is a minus sign
#(0, 'aeroplane'),
#(1, 'bicycle'),
#(2, 'bird'),
#(3, 'boat'),
#(4, 'bottle'),
#(5, 'bus'),
#(6, 'car'),
#(7, 'cat'),
#(8, 'chair'),
#(9, 'cow'),
#(10, 'diningtable'),
#(11, 'dog'),
#(12, 'horse'),
#(13, 'motorbike'),
#(14, 'person'),
#(15, 'pottedplant'),
#(16, 'sheep'), #
#(17, 'sofa'),
#(18, 'train'),
#(19, 'tvmonitor')
allowable_objs = set([1,4,6,8,10,14,17,19])
# seems that linux game has problem loading the first sound file: 00.wav, so the number is shifted...
allowable_objs = set( i-1 for i in allowable_objs )


NUM_CLASSES = 20
triggered = [time.time()]*NUM_CLASSES
COOL_DOWN_TIME = 1.5 # 1.5 second cool down time for each class
host = '127.0.0.1'
port = 5555
def estimate_distance(w,h):
    """Estimate distance from the box sizes
    w,h: box width and height represented as fractions
    """
    return 0.5 / np.sqrt(w*h)


class signal_analyzer(object):
  def __init__(self, cam_type):
    self.yolo = None #yolo.yolo_detector()
    self.cam_type = cam_type
    print cam_type+" camera is used"
    if cam_type=="fisheye":
      self.FOV = (180,180)
    if cam_type=="logitech":
      self.FOV = (72,54)
    if cam_type=="gopro":
      self.FOV = (122.6,94.4) # from website https://gopro.com/support/articles/hero3-field-of-view-fov-information
      self.FOV[0] = 170 # adjust the horzontal field because the display is not full
      pass

  def analyze(self,objs_data):
    # for each type of object, make sure time is passed cool down time before send to the unity game
    objs = set(obj[0] for obj in objs_data)
    for o_type in objs:
      if o_type not in allowable_objs:
        triggered[o_type] = time.time()
      if time.time() - triggered[o_type] > COOL_DOWN_TIME:
        # collection strings to send, then send at once
        sendstrs = []
        for obj in objs_data:
          o,prob,name,x,y,w,h = obj
          triggered[o] = time.time()
          thetax = (x-0.5) * self.FOV[0]
          thetay = (y-0.5) * self.FOV[1]
          dist = estimate_distance(w,h)
          xs = dist*np.cos( np.radians( thetay ) ) * np.sin( np.radians( thetax ) )
          ys = dist*np.sin( np.radians( thetay ) )
          zs = dist*np.cos( np.radians( thetay ) ) * np.cos( np.radians( thetax ) )
          sendstrs.append("%g,%g,%g,%d"%(xs,-ys,zs,o))
        print "send...",sendstrs,prob,name
        so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        so.connect((host, port))
        so.sendall("\n".join(sendstrs)) # use so.sendall to send to Unity game, not send_one_message
        so.close()

    return

if __name__ == "__main__":
  pass
  #clsr=yolo.yolo_detector()
  #scores, numbers, labels=clsr.get_class_by_file('cat.jpg')
  #print scores
  #print numbers
  #print labels
