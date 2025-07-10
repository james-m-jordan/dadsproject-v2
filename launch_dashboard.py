#!/usr/bin/env python3
"""
Launch script for the Publishing Analytics Dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages if not already installed"""
    try:
        import streamlit
        import pandas
        import plotly
        import sklearn
        print("✓ All required packages are installed")
    except ImportError as e:
        print(f"Installing missing packages: {e}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Publishing Analytics Dashboard...")
    print("📊 Dashboard will open in your default web browser")
    print("🛑 Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Launch streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", "publishing_dashboard.py"])

if __name__ == "__main__":
    print("📚 Publishing Analytics Dashboard Launcher")
    print("=" * 50)
    
    # Install requirements
    install_requirements()
    
    # Launch dashboard
    launch_dashboard()