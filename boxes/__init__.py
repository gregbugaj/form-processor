"""
Name : __init__.py
boxes module

This import path is important to allow importing correctly as package
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
