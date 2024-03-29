
"""
This file is useful for checking if your GPU is being utilised by torch.
"""

import torch
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')