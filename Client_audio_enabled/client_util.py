import socket
#import encode
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from shutil import copyfile


def play(data):
  fileid = np.random.randint(0,1000)
  unity_file= ".\location_sound_" + str(fileid)+".txt"
  #unity_file= "..\..\Pieces\BinauralSoundGenerator\win32\location_sound.txt"
  KINECT_HORIZONTAL_FIELD = 57 # field of view in degrees
  KINECT_VERTICAL_FIELD = 43 # field of view in degrees
  KINECT_HORIZONTAL_TAN = 0.54295 # tan(57/2 deg)
  KINECT_VERTICAL_TAN = 0.3939 # tan(43/2 deg)
  distance = 1  #z-depth in meters 
  z_factor = 1 #0.5
  if data:
    with open(unity_file+'tmp','w') as f:
      for ob in data:
        iclass = ob[0][0]
        x,y,w,h=ob[1],ob[2],ob[3],ob[4]
        
        x = (x-0.5) * 2 * distance * KINECT_HORIZONTAL_TAN # left right need to be reversed (because kinect image is reversed)
        y = (y-0.5) * 2 * distance * KINECT_VERTICAL_TAN
        z = distance * z_factor
        str_to_unity = ','.join(map(str,(x,y,z,iclass[0])))+'\n'
        #print str_to_unity
        f.write(str_to_unity)
    #copyfile(unity_file+'tmp', unity_file)
    print "file written:"
   
   
def sendData(so, string):
   #print len(string),string[:10000]
   if len(string) %1024 == 0:
       so.send(','+string)
   else:
       so.send(string) 
   print len(string),string[:10]+'  '+string[-10:]
   data = ''
   #print "waiting"
   data = so.recv(1024).decode()
   print data
   return eval(data)


#def connectAndSend(host, port, arr):
#	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#	host = host
#	port = port
#	s.connect((host,port))
#	sendData(s, base64.b64encode(arr))
#	s.close ()

def arr3d_img_string(arr):
	output = BytesIO()
	im = Image.fromarray(arr.astype(np.uint8).transpose((1,0,2)),'RGB')
	im.save(output,"JPEG")
	return base64.b64encode(output.getvalue())
	
def connectAndSendArr3d(host, port, arr):
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	host = host
	port = port
	s.connect((host,port))
	data=sendData(s, arr3d_img_string(arr))
	s.close ()
	play(data)
	return data

	

if __name__ == "__main__":
	connectAndSend('127.0.0.1', 13350, 'haha.jpg')
