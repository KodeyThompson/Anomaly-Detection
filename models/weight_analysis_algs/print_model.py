import sys
import torch

model = torch.load(sys.argv[1])
print(model)
