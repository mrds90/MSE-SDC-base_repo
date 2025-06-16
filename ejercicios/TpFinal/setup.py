#!/usr/bin/env python3

import os
import platform
import subprocess
import sys

def install_system_deps():
    system = platform.system()
    print(f"Detected OS: {system}")
    
    if system == "Linux":
        print("Installing system dependencies: doxygen and graphviz...")
        subprocess.run(["sudo", "apt", "update"], check=True)
        subprocess.run(["sudo", "apt", "install", "-y", "doxygen", "graphviz"], check=True)
    elif system == "Windows":
        print("⚠️ On Windows, please install Doxygen and Graphviz manually:")
        print("  - Doxygen: https://www.doxygen.nl/download.html")
        print("  - Graphviz: https://graphviz.org/download/")
    elif system == "Darwin":
        print("Installing system dependencies: doxygen and graphviz using Homebrew...")
        # Check if brew is installed
        try:
            subprocess.run(["brew", "--version"], check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print("❌ Homebrew is not installed. Please install Homebrew first: https://brew.sh/")
            sys.exit(1)
        subprocess.run(["brew", "update"], check=True)
        subprocess.run(["brew", "install", "doxygen", "graphviz"], check=True)
    else:
        print("Unsupported OS for automatic system package installation.")

def install_python_requirements(requirements_path):
    if not os.path.exists(requirements_path):
        print(f"❌ Error: '{requirements_path}' not found!")
        sys.exit(1)

    print(f"Installing Python dependencies from '{requirements_path}'...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], check=True)

if __name__ == "__main__":
    # Cambiá esto si cambia la ubicación relativa
    requirements_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../requirement.txt"))

    install_system_deps()
    install_python_requirements(requirements_path)