import socket
import numpy as np
import time
import struct

# Send to 3D Sound generator signal to simulate a sound source moving around the user's head

def send_one_message(sock, data):
    length = len(data)
    sock.sendall(struct.pack('!I', length))
    sock.sendall(data)


if __name__ == "__main__":
    host,port ='127.0.0.1', 5555
    t = time.time()
    # the sound source start from the right, moving around the use counterclock wise
    for i in range(36):
        time.sleep(0.8)
        x,z = np.cos( np.radians(i*10) ) , np.sin( np.radians(i*10) )
        so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        so.connect((host, port))
        so.sendall("%g,0,%g,1"%(x,z)) # use so.sendall to send to Unity game, not send_one_message
        so.close()
    print time.time()-t


#time.sleep(0.3)
#connectAndSend('127.0.0.1', 5554, '0,0,1,0')
#time.sleep(1)
#connectAndSend('127.0.0.1', 5554, '-1,0,1,0\n1,0,1,1')
#
    