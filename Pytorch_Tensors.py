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

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

if torch.cuda.is_available():
    ### Checks if gpu is available ###
    tensor = tensor.to('cuda')
    print(f"Device tensor is stored on: {tensor.device}")
tensor = torch.ones(4,4)
tensor[0:,1] = 0
print(tensor)

t1 = torch.cat([tensor,tensor,tensor],dim = 1)
print(t1)

# This computes the element-wise product
print(f"tensor.mul(tensor) \n{tensor.mul(tensor)} \n")
print(f"tensor * tensor \n {tensor*tensor}")

print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)}\n")
# Alternative syntax
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")
"""
the _ is an in place operation and saves some memory but is problematic when computing derivatives because of an immediate loss of history so their use is discoouraged
"""
print(tensor, "\n")
tensor.add_(5)
print(tensor)
"""
These share memory location
"""
t = torch.ones(6)
n = t.numpy()
print(f"t: {t}")
print(f"n: {n}")
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


#### NumPy array to Tensor ####
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")