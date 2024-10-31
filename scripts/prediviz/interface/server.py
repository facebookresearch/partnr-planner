# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import argparse
import http.server
import json
import os
import socketserver
import threading
import time
import webbrowser

SCRIPT_DIR = os.path.relpath(os.path.dirname(os.path.abspath(__file__)), os.getcwd())

HTML_FILE = os.path.join(SCRIPT_DIR, "interface.html")
SAVE_FILE = os.path.join(SCRIPT_DIR, "annotations.json")
JSON_FILE = os.path.join(SCRIPT_DIR, "sample_episodes.json")


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/sample_episodes.json":
            try:
                with open(JSON_FILE, "r") as f:
                    json_data = f.read()
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json_data.encode("utf-8"))
            except Exception as e:
                print(f"Error loading JSON: {e}")
                self.send_response(500)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Failed to load JSON file")
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/save_annotations":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                with open(SAVE_FILE, "w") as f:
                    json.dump(data, f, indent=2)
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Annotations saved successfully")
            except Exception as e:
                print(f"Error saving annotations: {e}")
                self.send_response(500)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Failed to save annotations")
        else:
            self.send_error(404, "File not found")


def start_server(port):
    handler = CustomHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at port {port}")
        httpd.serve_forever()


def open_browser(port):
    time.sleep(1)
    webbrowser.open(f"http://localhost:{port}/{HTML_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a simple HTTP server.")
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Port to run the server on (default: 8888)",
    )
    args = parser.parse_args()

    server_thread = threading.Thread(target=start_server, args=(args.port,))
    server_thread.start()

    open_browser(args.port)
