import os
import subprocess

def set_environment_variables():
    port = input("Enter the port number (default 5000): ") or "5000"
    debug = input("Enable debug mode? (y/n): ").lower() in ['y', 'yes']
    host = input("Enter the host (default 127.0.0.1): ") or "127.0.0.1"

    os.environ["FLASK_APP"] = "application.py" # Change this to your main Flask app file
    os.environ["FLASK_ENV"] = "development"
    os.environ["FLASK_RUN_PORT"] = port
    os.environ["FLASK_RUN_HOST"] = host
    os.environ["FLASK_DEBUG"] = "1" if debug else "0"

def run_flask_app():
    subprocess.run(["flask", "run"])

if __name__ == "__main__":
    set_environment_variables()
    run_flask_app()
