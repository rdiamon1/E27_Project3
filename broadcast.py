import numpy as np

A = np.arange(12).reshape(4,3)

rowconstants = np.array([1,2,3,4])
colconstants = np.array([10,20,30])

print A

print A * rowconstants.reshape((4,1))

print rowconstants.shape
print rowconstants.T.shape
print rowconstants[:,None].shape

print A * colconstants

print rowconstants[:,None]*colconstants
