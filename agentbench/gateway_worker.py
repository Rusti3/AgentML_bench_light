"""HTTP over an AF_UNIX socket: only configured model API routes are forwarded."""
import http.server
import json
import os
from pathlib import Path
import socketserver
import urllib.parse

import httpx

ALLOWED={'/v1/responses','/responses','/v1/chat/completions','/chat/completions','/v1/models','/models'}


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):self.forward()
    def do_POST(self):self.forward()
    def log_message(self,*args):pass

    def forward(self):
        parsed=urllib.parse.urlsplit(self.path)
        if parsed.path not in ALLOWED or parsed.query or parsed.fragment or parsed.scheme or parsed.netloc:
            self.send_error(403,'Only configured model routes are allowed');return
        if self.command=='GET' and not parsed.path.endswith('/models'):
            self.send_error(405);return
        if self.command=='POST' and parsed.path.endswith('/models'):
            self.send_error(405);return
        length=int(self.headers.get('Content-Length','0'))
        if length>16*1024*1024:self.send_error(413);return
        body=self.rfile.read(length)
        if self.command=='POST':
            try:
                content=json.loads(body)
                if not isinstance(content,dict):raise ValueError()
            except Exception:self.send_error(400,'JSON object required');return
        base=os.environ['BENCH_UPSTREAM_BASE_URL'].rstrip('/')
        route=parsed.path
        if base.endswith('/v1') and route.startswith('/v1/'):route=route[3:]
        headers={'Content-Type':'application/json','Accept-Encoding':'identity'}
        key=os.environ.get('BENCH_UPSTREAM_API_KEY')
        if key:headers['Authorization']='Bearer '+key
        headers.update(json.loads(os.environ.get('BENCH_UPSTREAM_HEADERS','{}')))
        try:
            with httpx.Client(timeout=1200,follow_redirects=False) as client:
                with client.stream(self.command,base+route,content=body,headers=headers) as response:
                    self.send_response(response.status_code)
                    self.send_header('Content-Type',response.headers.get('content-type','application/json'))
                    self.send_header('Connection','close');self.end_headers()
                    for block in response.iter_bytes():self.wfile.write(block);self.wfile.flush()
        except Exception:
            # Do not expose upstream URLs, credentials or response bodies in infrastructure logs.
            try:self.send_error(502,'Configured upstream unavailable')
            except Exception:pass


class Server(socketserver.ThreadingMixIn,socketserver.UnixStreamServer):
    daemon_threads=True


if __name__=='__main__':
    path=Path('/ipc/model.sock');path.unlink(missing_ok=True)
    server=Server(str(path),Handler)
    path.chmod(0o666)
    print('MODEL_GATEWAY_READY',flush=True)
    server.serve_forever()
