print("Starting wsgi.py...")

from app import app as application

if __name__ == "__main__":
    # This only runs for development.
    application.run()
