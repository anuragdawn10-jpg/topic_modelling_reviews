"""
Deployment entry point.

Some platforms (and Streamlit Community Cloud's default settings) look for
`app.py` specifically. This just runs the actual app defined in
streamlit_app.py, so there's a single source of truth for the app logic.

Run with:
    streamlit run app.py
"""
from streamlit_app import main

if __name__ == "__main__":
    main()
