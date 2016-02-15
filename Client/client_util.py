import socket
#import encode
import base64
from PIL import Image
from io import BytesIO
import numpy as np

def sendData(so, string):
   #print len(string),string[:10000]
   if len(string) %1024 == 0:
       so.send(','+string)
   else:
       so.send(string) 
   print len(string),string[:10]+'  '+string[-10:]
   data = ''
   print "waiting"
   data = so.recv(1024).decode()
   print (data)
   return eval(data)

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
	return data

if __name__ == "__main__":
	connectAndSend('127.0.0.1', 13350, 'haha.jpg')
