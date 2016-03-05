import socket
import sys
import image_analyzer
import struct
IA = image_analyzer.image_analyzer()

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
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host = '127.0.0.1'
    port = 6666
    s.bind((host, port))
    s.listen(5)

    while True:
        c, addr = s.accept()
        totalReceive = recv_one_message(c)
        c.sendall(IA.analyze(totalReceive))
        c.close()

finally:
    print "error, quit"
    s.close()
