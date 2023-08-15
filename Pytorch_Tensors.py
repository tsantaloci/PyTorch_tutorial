import torch
import numpy as np

data = [[1,2], [3,4]]
x_data = torch.tensor(data)
print(x_data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

shape = (3, 6,)
'''
first index in the tuple is number of columns
second index in the tuple is number of rows
'''
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")