import socket
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import struct

def arr3d_img_string(arr):
	output = BytesIO()
	im = Image.fromarray(arr.astype(np.uint8).transpose((1,0,2)),'RGB')
	im.save(output,"JPEG")
	return base64.b64encode(output.getvalue())

def send_one_message(sock, data):
    length = len(data)
    sock.sendall(struct.pack('!I', length))
    sock.sendall(data)

def connectAndSendArr3d(so, arr):
	data = arr3d_img_string(arr)
	send_one_message(so, data)
	return_data = so.recv(10000)
	return eval(return_data) if return_data else None
