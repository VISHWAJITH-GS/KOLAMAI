"""
Kolam AI - Application Runner
This script launches the Flask web application.
"""

from app import app

if __name__ == "__main__":
	# You can set debug and host/port here or via environment variables
	app.run(debug=True, host="0.0.0.0", port=5000)
