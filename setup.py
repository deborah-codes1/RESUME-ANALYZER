#!/usr/bin/env python3
"""
Run this once before starting the app:
    python setup.py
"""
import subprocess, sys

print("Installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

print("\nDownloading spaCy language model...")
subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

print("\nSetup complete! Start the app with:")
print("   python app.py")
