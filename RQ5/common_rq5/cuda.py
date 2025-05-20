"""
This file provides some debugging information to determine the version of cuda 
pytorch is using.
"""

import torch
import subprocess

def get_nvidia_smi_output():
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error running nvidia-smi: {e.stderr}"

# Example usage
output = get_nvidia_smi_output()
print(output)

print(f"torch.__version__ = {torch.__version__}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"torch.version.cuda = {torch.version.cuda}")