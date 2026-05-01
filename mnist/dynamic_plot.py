from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import webbrowser

# HTML content
html_content = b""


# Simple HTTP server class
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress logging of HTTP requests

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html_content)

    def do_POST(self):
        global html_content
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        html_content = post_data
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"HTML content updated successfully.")


def start_http_server():
    """Start a simple HTTP server."""
    server_address = ("localhost", 8000)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    httpd.serve_forever()


def update_html_content(new_html_content):
    """Update the HTML content."""
    global html_content
    html_content = new_html_content


def update_html_content():
    """Update the HTML content from the file and trigger browser refresh."""
    global html_content
    with open("updated_plot.html", "rb") as file:
        html_content = file.read()
    # JavaScript to refresh the page
    js_refresh = b"<script>window.location.reload(true);</script>"
    html_content += js_refresh


# server_thread = threading.Thread(target=start_http_server)
# server_thread.daemon = True
# server_thread.start()

# Open the browser window to display the HTML


if __name__ == "__main__":
    # Start HTTP server in a separate thread
    server_thread = threading.Thread(target=start_http_server)
    server_thread.daemon = True
    server_thread.start()

    # Open the browser window to display the HTML
    webbrowser.open("http://localhost:8000")

    update_html_content()

    # Example: periodically update the HTML content
    import time

    while True:
        # Example: dynamically generate or fetch the HTML content
        update_html_content()
        time.sleep(100)  # Wait for some time before updating (e.g., 10 seconds)


# from http.server import HTTPServer, BaseHTTPRequestHandler
# import threading
# import webbrowser
# import time

# # HTML content
# html_content = b""
# html_lock = threading.Lock()

# # Simple HTTP server class
# class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
#     def do_GET(self):
#         self.send_response(200)
#         self.send_header('Content-type', 'text/html')
#         self.end_headers()
#         with html_lock:
#             self.wfile.write(html_content)

#     def do_POST(self):
#         content_length = int(self.headers['Content-Length'])
#         post_data = self.rfile.read(content_length)
#         with html_lock:
#             global html_content
#             html_content = post_data
#         self.send_response(200)
#         self.end_headers()
#         self.wfile.write(b'HTML content updated successfully.')

# def start_http_server():
#     """Start a simple HTTP server."""
#     server_address = ('localhost', 8000)
#     httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
#     httpd.serve_forever()

# def update_html_content():
#     """Update the HTML content from the file."""
#     while True:
#         with open('updated_plot.html', 'rb') as file:
#             new_content = file.read()
#         with html_lock:
#             global html_content
#             html_content = new_content
#         time.sleep(10)  # Wait for some time before updating (e.g., 10 seconds)

# # Start HTTP server in a separate thread
# server_thread = threading.Thread(target=start_http_server)
# server_thread.daemon = True
# server_thread.start()

# # Start thread to update HTML content periodically
# update_thread = threading.Thread(target=update_html_content)
# update_thread.daemon = True
# update_thread.start()
# if __name__ == "__main__":
#     # Start HTTP server in a separate thread
#     server_thread = threading.Thread(target=start_http_server)
#     server_thread.daemon = True
#     server_thread.start()

#     # Start thread to update HTML content periodically
#     update_thread = threading.Thread(target=update_html_content)
#     update_thread.daemon = True
#     update_thread.start()

#     # Open the browser window to display the HTML
#     webbrowser.open('http://localhost:8000')

#     # Keep the main thread running to allow interaction with the HTML
#     while True:
#         time.sleep(1)
