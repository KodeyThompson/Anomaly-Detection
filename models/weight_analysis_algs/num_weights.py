import torch
import sys

model = torch.load(sys.argv[1])

total_params = 0

for key in model.keys():
	if sys.argv[2] == '1':
		space = 50 - len(key)
		print(f'{key} shape:', ' ' * space, {model[key].size()})
	for index, dim in enumerate(model[key].size()):
		if index == 0:
			new_params = dim
		else:
			new_params *= dim
	if len(model[key].size()) > 0:
		total_params += new_params

print(f'This model has {total_params} total parameters')
