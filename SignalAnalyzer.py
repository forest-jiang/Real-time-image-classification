import numpy as np
import base64
import socket
import time
import logging
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

names=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
  'diningtable', 'dog', 'horse', 'motorbike','person', 'pottedplant', 'sheep',
  'sofa', 'train', 'tvmonitor']


NUM_CLASSES = 20
triggered = [time.time()]*NUM_CLASSES
COOL_DOWN_TIME = 1.0 # 1.5 second cool down time for each class
host = '127.0.0.1'
port = 5555
def estimate_distance(w,h):
    """Estimate distance from the box sizes
    w,h: box width and height represented as fractions
    """
    return 0.5 / np.sqrt(w*h)


class signal_analyzer(object):
  def __init__(self, cam_type,correct_missed_wav=False):
    self.allowable_objs = set([1,4,6,8,10,14,15,17,19])
    self.allowable_objs = set([4,8,10,14,15,17,19])
    # seems that linux game has problem loading the first sound file: 00.wav, so the number is shifted...
    if correct_missed_wav:
        self.allowable_objs = set( i-1 for i in self.allowable_objs )
    self.yolo = None #yolo.yolo_detector()
    self.cam_type = cam_type
    print cam_type+" camera is used"
    if cam_type=="ricoh":
      self.FOV = (360,180)
    if cam_type=="logitech":
      self.FOV = (72,54)*2
    if cam_type=="gopro":
      self.FOV = (122.6,94.4) # from website https://gopro.com/support/articles/hero3-field-of-view-fov-information
      self.FOV = (170, 94.4)# adjust the horzontal field because the display is not full
      pass
    # logging code, not tested yet
    logging.info('Cam type is: '+cam_type)
    logging.info('FOV in deg is: '+repr(self.FOV))
    logging.info('Now local time is: '+ repr(time.localtime()))


  def analyze(self,objs_data):
    # for each type of object, make sure time is passed cool down time before send to the unity game
    objs = set(obj[0] for obj in objs_data)
    for o_type in objs:
      if o_type not in self.allowable_objs:
        print "ignore:", o_type,names[o_type]
        triggered[o_type] = time.time()
      if time.time() - triggered[o_type] > COOL_DOWN_TIME:
        # collection strings to send, then send at once
        sendstrs,sendnames,sendprobs = [],[],[]
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
          sendprobs.append(prob)
          sendnames.append(name)
        print "send...",sendstrs,prob,name, time.time()
        # logging code, not tested yet
        #logging.info('Time: '+repr(time.time()))
        #logging.info('Sending: names:'+ ','.join(sendnames) +
        #	','.join(map(str,sendprobs)) + '| str'+';'.join(sendstrs))

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


#TODO today:
# so 1. strech a little far on the prototype (nothing really...)
# 2. really think outloud, shoot video.
# 3. People looking at you.
# need: wireless earbuds.