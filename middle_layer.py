import socket
import struct
import time
import os, sys, logging

import SignalAnalyzer

cam_type = "logitech"
#cam_type = "gopro"
#cam_type = "ricoh"


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


def recv_one_message(sock):
    lengthbuf = recvall(sock, 4)
    length, = struct.unpack('!I', lengthbuf)
    return recvall(sock, length)


# Exampe of message received from Yolo:
# car 6 0.266080 0.505038 0.504032 0.962252 0.904232;
# tvmonitor 19 0.206442 0.091263 0.508923 0.181008 0.689912;tvmonitor 19 0.887895 0.484474 0.673417 0.707561 0.655528;
# format: name, class id, probability, x, y, w, h
thres = 0.2
correct_missed00 = False # sometimes the SoundGenerator failed to load the first wav file...

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host = '127.0.0.1'
    port = 5554
    s.bind((host, port))
    s.listen(5)
    sa = SignalAnalyzer.signal_analyzer(cam_type,correct_missed00)
    count = 0
    print "listener ready"
    c, addr = s.accept()
    #logging.basicConfig(filename='log'+repr(time.time()),level=logging.DEBUG )
    while True:
        count += 1
        #print count # debug use
        #totalReceive = recv_one_message(c)
        totalReceive = c.recv(4096)

        assert(totalReceive[-1]==';')
        objects = []
        for objstr in totalReceive.split(';')[:-1]:
            name,objid,prob,x,y,w,h = objstr.split(' ')
            prob,x,y,w,h = map(float,(prob,x,y,w,h))
            if correct_missed00:
                objid = int(objid)-1
            else:
                objid = int(objid)

            if prob >= thres:
                objects.append((objid,prob,name,x,y,w,h))
        #print totalReceive, objects, "analyze..."
        sa.analyze(objects)
#        sendback = "got"#
#        print sendback
        #c.sendall(sendback)
        #c.close()
    c.close()
except Exception:
    print sys.exc_info()[0]
finally:
    print "error, quit"
    s.close()



