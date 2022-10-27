import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("192.168.13.118", 6688))
server.listen(5)

print("waiting for connect...")
connect, (host, port) = server.accept()
print("the client %s:%s has connected." % (host, port))

while True:
    connect.sendall(b"your words has received.")
