import socket
import sys
import struct
import SignalAnalyzer

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

try:
    sa = SignalAnalyzer.signal_analyzer()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host = '127.0.0.1'
    port = 5554
    s.bind((host, port))
    s.listen(5)
    print "listener ready"
    count = 0
    while True:
        c, addr = s.accept()
        totalReceive = recv_one_message(c)
        sendback = sa.analyze(totalReceive)
        count += 1
        print count,totalReceive,sendback
        c.sendall(sendback)
        c.close()

finally:
    print "error, quit"
    s.close()



