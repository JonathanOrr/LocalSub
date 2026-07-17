#!/usr/bin/env python3
from webapp.app import app

if __name__ == "__main__":
    print("subtranslate web UI: http://127.0.0.1:5000")
    app.run(debug=False, threaded=True, port=5000)
