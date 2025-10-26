# app.py

from flask import Flask, render_template

# Initialize the Flask application
app = Flask(__name__)

# --- Main Route ---
# This route will serve our index.html page
@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- API Route (for later) ---
# We will add a @app.route('/predict') route here in the next step
# to handle the webcam data.

# --- Run the Server ---
if __name__ == "__main__":
    # We'll run on port 5000 in debug mode
    app.run(debug=True, port=5000)