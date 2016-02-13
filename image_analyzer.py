import caffe_classifier as mc
import numpy as np
from PIL import Image
from io import BytesIO
import base64

class image_analyzer(object):
  def __init__(self):
    self.clsr = mc.caffe_classifier()
  def analyze(self,img_data):
    if img_data[0] == ",":
      img_data = img_data[1:]
    
    im = Image.open(BytesIO(base64.b64decode(img_data)))
    print np.array(im).shape, "should be H x W x 3"
    scores, numbers, labels=self.clsr.get_class(np.array(im)/255.0)
    r = '\n'.join([str(scores[i])+"  "+labels[i] for i in range(5)])    
    return r
    
if __name__ == "__main__":
  clsr=mc.caffe_classifier()
  scores, numbers, labels=clsr.get_class_by_file('cat.jpg')
  print scores
  print numbers
  print labels
