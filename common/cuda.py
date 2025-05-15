"""
This file provides some debugging information to determine the version of cuda 
pytorch is using.
"""

import torch

print(f"torch.__version__ = {torch.__version__}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"torch.version.cuda = {torch.version.cuda}")