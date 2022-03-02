# importing the numpy module
import numpy as np

# Making fist 1-D vector v1
v1 = np.array([3, 7, 4, 0, 0, 0])
print("First vector sequence is: ", v1)

# Making second 1-D vector v2
v2 = np.ones(v1.shape[0])
print("Second vector sequence is: ", v2)

print("\nprinting linear convolution  result between v1 and v2 using 'same' mode:")
print(np.convolve(v1, v2, mode='same'))

