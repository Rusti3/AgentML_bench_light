"""Private loopback TCP bridge to the gateway's Unix socket, with no IP network."""
import os
import select
import socket
import socketserver
import subprocess
import sys
import threading


class Bridge(socketserver.BaseRequestHandler):
    def handle(self):
        remote=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        try:
            remote.connect('/llm/model.sock')
            peers={remote:self.request,self.request:remote}
            while True:
                ready,_,_=select.select(list(peers),[],[],1250)
                if not ready:break
                for src in ready:
                    block=src.recv(65536)
                    if not block:return
                    peers[src].sendall(block)
        finally:remote.close()


class Server(socketserver.ThreadingMixIn,socketserver.TCPServer):
    allow_reuse_address=True
    daemon_threads=True


if __name__=='__main__':
    if os.environ.get('BENCH_LLM_GATEWAY')=='1':
        server=Server(('127.0.0.1',8090),Bridge)
        threading.Thread(target=server.serve_forever,daemon=True).start()
    child=subprocess.Popen(sys.argv[1:],stdin=subprocess.DEVNULL)
    raise SystemExit(child.wait())
