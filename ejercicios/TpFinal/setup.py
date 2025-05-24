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