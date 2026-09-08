"""Launcher for robot_studio (works from any cwd)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robot_studio.app import main

if __name__ == "__main__":
    main()