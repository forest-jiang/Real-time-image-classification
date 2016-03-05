import numpy as np
import base64
import socket

# This class does the following:
#   1. Convert from box representation (obj,x,y,w,h) to 3d representation (x,y,z,obj)
#   2. Send signal to Unity to trigger the 3D sound
#   3. Make sure there is a cool down period for each class of the object
# cam_type: "fisheye" or "flat"

NUM_CLASSES = 20
triggered = [time.time()]*NUM_CLASSES
COOL_DOWN_TIME = 1.5 # 1.5 second cool down time for each class

def estimate_distance(w,h):
    """Estimate distance from the box sizes
    w,h: box width and height represented as fractions
    """
    return 0.5 / np.sqrt(w*h)

    
class signal_analyzer(object):
  def __init__(self, cam_type):
    self.yolo = None #yolo.yolo_detector()
    self.cam_type = cam_type
    # FOV: horzontal and vertical
    self.FOV = (180,180) if cam_type=="fisheye" else (10,10) ############ go pro one

  def analyze(self,objs_data):
    # for each type of object, make sure time is passed cool down time before send to the unity game
    objs = set(obj[0] for obj in objs_data)
    for o_type in objs:
      if time.time() - triggered[o_type] > COOL_DOWN_TIME:
        # collection strings to send, then send at once
        sendstrs = []
        for obj in objs_data:
          o,x,y,w,h = obj
          triggered[o] = time.time()
          thetax = (x-0.5) * self.FOV[0]
          thetay = (y-0.5) * self.FOV[1]
          dist = estimate_distance(w,h)
          xs = dist*np.cos( np.radians( thetay ) ) * np.sin( np.radians( thetax ) )
          ys = dist*np.sin( np.radians( thetay ) )
          zs = dist*np.cos( np.radians( thetay ) ) * np.cos( np.radians( thetax ) )
          sendstrs.append("%g,%g,%g,%d"%(xs,ys,zs,o))
        so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        so.connect((host, port))
        print "send...",sendstrs
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
