from http.server import HTTPServer, SimpleHTTPRequestHandler

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

PORT = 8000
httpd = HTTPServer(('localhost', PORT), CORSRequestHandler)
print(f"Serving at http://localhost:{PORT}")
httpd.serve_forever()
