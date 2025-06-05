def main():
    import os
    import streamlit

    # Path to the Streamlit app file
    app_path = os.path.join(os.path.dirname(__file__), "app.py")

    # Run the Streamlit app
    streamlit.run(app_path)