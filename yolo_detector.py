import numpy as np
import sys, os, time, random
from PIL import Image
from io import BytesIO
import base64

class yolo_detector(object):
    def __init__(self):
        self.counter = 0
        self.yolo_watch_path = "./tmp/"
        # load labels

        self.labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                       "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                       "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    def remove_tmp_files(self):
        now = time.time()
        for f in os.listdir(self.yolo_watch_path):
          f = os.path.join(self.yolo_watch_path, f)
          if os.stat(f).st_mtime < now - 60:
            if os.path.isfile(f):
              os.remove(f)
    def get_objs(self,img_data):
        self.counter+=1
        if self.counter > 60:
          self.remove_tmp_files()
        img_id = random.randint(1,1000000)
        img_name = self.yolo_watch_path+"%d.jpg" % img_id
        im = Image.open(BytesIO(base64.b64decode(img_data)))
        im.save(img_name)
        
        resultfile = self.yolo_watch_path + "%d.jpg.txt" %img_id
        while not os.path.isfile(resultfile):
          time.sleep(0.05)
        with open(resultfile) as f:
          ann = f.readlines()
        
        res = []
        #print "start parsing...", ann
        for entry in ann:
          #print "entry: ", repr(entry)
          classstr, boxstr = entry.split(';')
          classstrs = classstr.split(', ')
          classes = []
          for cstr in classstrs:
            #print "class str:", repr(cstr),repr(cstr.strip())
            if not cstr.strip(): continue
            iclass, probclass = [s for s in cstr.split(' ') if s.strip()]
            iclass, probclass = int(iclass), float(probclass)
            classes.append((iclass,self.labels[iclass],probclass))
          #print "box str:", boxstr
          x,y,w,h = [float(s) for s in boxstr.split(' ') if s.strip()]
          res.append((classes,x,y,w,h))
          return res
   
