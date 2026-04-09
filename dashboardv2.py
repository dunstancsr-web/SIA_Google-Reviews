import sys
import os
import streamlit as st



# Add the 'app' directory to sys.path so imports inside dashboardv2 still work if any are relative to it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Execute the actual dashboard module so the app continues to work in the meantime
app_path = os.path.join(os.path.dirname(__file__), 'app', 'dashboardv2.py')
with open(app_path) as f:
    code = f.read()
    exec(code, globals())
