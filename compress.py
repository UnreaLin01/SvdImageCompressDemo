from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the webp image
imagePath = "image/cat.webp"
image = Image.open(imagePath)

# Convert original image to array
originalImageArray = np.array(image)

# Resize the image
image = image.resize((50, 50))

# Convert the image to grayscale
image = image.convert('L')

threshold = 128
image = image.point(lambda p: p > threshold and 255)

# Convert image to array
imageArray = np.array(image)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot original image
axs[0].imshow(originalImageArray)
axs[0].set_title("Original Image")

# Plot processed image
axs[1].imshow(imageArray, cmap="gray")
axs[1].set_title("Processed Image")

# Display the images
plt.show()

# Left sigular vectors
# Sigular matrix
# Right sigular vectors

print("Image Rank:", np.linalg.matrix_rank(imageArray))

U, S, VT = np.linalg.svd(imageArray)

# Visualize U, S, and VT
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot U (left singular vectors)
axs[0].imshow(U, cmap="gray")
axs[0].set_title("U (Left Singular Vectors)")

# Plot S (singular values)
print()
axs[1].imshow(np.diag(S), cmap="gray", vmin=0, vmax=np.max(S)/13)
axs[1].set_title("S (Singular Values)")

# Plot VT (right singular vectors)
axs[2].imshow(VT, cmap="gray")
axs[2].set_title("VT (Right Singular Vectors)")

# Display the SVD components
plt.show()

def diplay_n_component(U, S, VT, n, fig_n):
    U_n = np.full((50, 50), np.max(U))
    U_n[:, n-1] = U[:, n-1]
    S_n = np.zeros((50, 50))
    S_n[n-1, n-1] = S[n-1]
    V_n = np.full((50, 50), np.max(VT))
    V_n[n-1, :] = VT[n-1, :]
    N_approximation = np.outer(U[:, n-1], VT[n-1, :]) * S[n-1]

    # Plot U_n
    axs[fig_n, 0].imshow(U_n, cmap="gray", vmin=np.min(U), vmax=np.max(U))
    axs[fig_n, 0].set_title(f"$U_{{{n}}}$")

    # Plot S_n
    axs[fig_n, 1].imshow(S_n, cmap="gray", vmin=0, vmax=S[n-1])
    axs[fig_n, 1].set_title(f"$S_{{{n}}}$")

    # Plot V_n^T
    axs[fig_n, 2].imshow(V_n, cmap="gray", vmin=np.min(VT), vmax=np.max(VT))
    axs[fig_n, 2].set_title(f"${{V_{{{n}}}}}^T$")
    # Plot N_approximation
    axs[fig_n, 3].imshow(N_approximation, cmap="gray")
    axs[fig_n, 3].set_title(f"$U_{{{n}}} * S_{{{n}}} * {{V_{{{n}}}}}^T$")

fig, axs = plt.subplots(3, 4, figsize=(15, 6))

diplay_n_component(U, S, VT, 1, 0)
diplay_n_component(U, S, VT, 2, 1)
diplay_n_component(U, S, VT, 3, 2)
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 8, figsize=(15, 6))

def comp(U, S, VT, n, row, col):
    N_approximation = np.outer(U[:, n-1], VT[n-1, :]) * S[n-1]

    # Plot N_approximation
    axs[row, col].imshow(N_approximation, cmap="gray")
    axs[row, col].set_title(f"$U_{{{n}}} * S_{{{n}}} * {{V_{{{n}}}}}^T$")

def best(U, S, VT, n, row, col):
    N_approximation = np.zeros((50,50))

    for i in range(1, n+1):
        N_approximation += np.outer(U[:, i-1], VT[i-1, :]) * S[i-1]

    # Plot N_approximation
    axs[row, col].imshow(N_approximation, cmap="gray")
    axs[row, col].set_title(f"Best Rank-{n}")

for i in range(8):
    comp(U, S, VT, i+1, 0, i)
    best(U, S, VT, i+1, 1, i)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 8, figsize=(15, 6))

for i in range(8):
    comp(U, S, VT, i+9, 0, i)
    best(U, S, VT, i+9, 1, i)


plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 8, figsize=(15, 6))

for i in range(8):
    comp(U, S, VT, i+17, 0, i)
    best(U, S, VT, i+17, 1, i)


plt.tight_layout()
plt.show()


# Calculate the rank of the original matrix
original_rank = np.linalg.matrix_rank(imageArray)

# Calculate the data for the new graphs
ranks = np.arange(1, 51)
original_space = 50 * 50
space_needed = ranks * (50 + 50 + 1)
space_ratio_percentage = (space_needed / original_space) * 100
information_storage = [np.sum(S[:n]) / np.sum(S) for n in ranks]

# Plot space comparison graph as percentage
plt.figure(figsize=(10, 5))
plt.plot(ranks, space_ratio_percentage, label='Space Needed / Original Space (%)')
plt.axvline(x=original_rank, color='red', linestyle='--', label=f'Original Rank = {original_rank}')
plt.xlabel('Best Rank-n')
plt.ylabel('Space Ratio (%)')
plt.title('Space Needed Compared to Original Space (%)')
plt.legend()
plt.grid(True)

# Annotate the point where the space is the same for the original rank approximation
same_space_percentage = (original_rank * (50 + 50 + 1) / original_space) * 100
plt.text(original_rank + 1, same_space_percentage, 
         f'{same_space_percentage:.2f}% Space', 
         color='red', fontsize=12)

plt.show()
