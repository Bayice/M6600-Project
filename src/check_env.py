"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import torch
import torchaudio
import transformers

print("PyTorch:", torch.__version__)
print("Torchaudio:", torchaudio.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA used by PyTorch:", torch.version.cuda)
else:
    print("No CUDA detected. CPU can still run inference, but slower.")