"""Start the LLM web app and open it in the browser."""

import threading
import webbrowser
import uvicorn

URL = "http://127.0.0.1:8000"

if __name__ == "__main__":
    print("Starting your LLM web app...")
    print(f"Opening browser at {URL}")

    threading.Timer(
        2.0,
        lambda: webbrowser.open(URL)
    ).start()

    uvicorn.run(
        "web.app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )