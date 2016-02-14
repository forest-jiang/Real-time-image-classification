import yolo_detector as yolo
import numpy as np
import base64

class image_analyzer(object):
  def __init__(self):
    self.yolo = yolo.yolo_detector()
  
  def analyze(self,img_data):
    if img_data[0] == ",":
      img_data = img_data[1:]
    
    objs = self.yolo.get_objs(img_data)
	
    
    return repr(objs)
    
if __name__ == "__main__":
  clsr=yolo.yolo_detector()
  #scores, numbers, labels=clsr.get_class_by_file('cat.jpg')
  #print scores
  #print numbers
  #print labels
