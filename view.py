import numpy as np

# Path to your .npy file
file_path = "data/nmed_rn34_ham10k_vectors.npy"

# Load the .npy file
data = np.load(file_path)

# Display basic information
print("Shape of the array:", data.shape)
print("Data type:", data.dtype)

# Display the first few elements
print("First 5 elements:")
print(data[:5])
