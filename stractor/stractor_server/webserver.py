import os, json
from http.server import BaseHTTPRequestHandler, HTTPServer
from routes.default import routes

from response.staticHandler import StaticHandler
from response.templateHandler import TemplateHandler
from response.badRequestHandler import BadRequestHandler


class WebServer(HTTPServer):
    def __init__(self, db, *kargs):
        super().__init__(*kargs)
        self.db = db
        print(f"[Server init] Server loaded. DB: {len(self.db)} samples")


class WebServerRequestHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        return

    def do_POST(self):
        split_path = os.path.splitext((self.path))
        request_extension = split_path[1]

        status = 404
        ctype = 'text/plain'
        contents = "404 not found"

        if (request_extension is "" or request_extension is ".html"):
            if (self.path in routes):
                content_length = int(self.headers['Content-Length'])
                data = self.rfile.read(content_length).decode()
                data = json.loads(data)
                idx = int(data['id'])

                if(idx in self.server.db):
                    status = 200
                    contents = self.server.db[idx]['hash']

        self.send_response(status)
        self.send_header('Content-type', ctype)
        self.end_headers()
        self.wfile.write(contents.encode('utf-8'))

    def do_GET(self):
        split_path = os.path.splitext(self.path)
        request_extension = split_path[1]

        if(request_extension is "" or request_extension is ".html"):
            if(self.path in routes):
                handler = TemplateHandler()
                handler.find(routes[self.path])
            else:
                handler = BadRequestHandler()
        elif request_extension is ".py":
            handler = BadRequestHandler()
        else:
            handler = StaticHandler()
            handler.find(self.path)

        self.respond({'handler': handler})

    def handle_http(self, handler):
        status_code = handler.getStatus()
        self.send_response(status_code)

        if(status_code is 200):
            content = handler.getContents()     # The actual binary read is done from RequestHandler() class
                                                # self.content is actually a file handler, in rb or r mode depending on the type of the file
            self.send_header('Content-type', handler.getContentType())
        else:
            content = "404 Not Found"

        self.end_headers()

        if isinstance(content, (bytes, bytearray)):
            return content

        return bytes(content, 'utf-8')

        # send a file:
        # with open(filepath, 'rb') as f:
        #   self.wfile.write(f.read())

    def respond(self, opts):
        response = self.handle_http(opts['handler'])
        self.wfile.write(response)
