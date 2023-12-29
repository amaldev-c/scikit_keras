"""Adds the 'fire_ext_model' package root folder to the list of package paths"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
