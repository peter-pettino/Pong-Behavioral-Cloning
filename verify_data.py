import numpy as np
import matplotlib.pyplot as plt

# Load the processed data
data = np.load('data/processed/pong_dataset.npz')
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']

print("Verifying the processed data...")
print(f"\nTraining set:")
print(f"  Number of samples: {X_train.shape[0]}")
print(f"  Shape of a single image: {X_train[0].shape}")
print(f"  Data type: {X_train.dtype}")
print(f"  Min pixel value: {X_train.min()}")
print(f"  Max pixel value: {X_train.max()}")
print(f"  Actions shape: {y_train.shape}")
print(f"  Unique actions: {np.unique(y_train)}")

print(f"\nTest set:")
print(f"  Number of samples: {X_test.shape[0]}")
print(f"  Shape of a single image: {X_test[0].shape}")
print(f"  Data type: {X_test.dtype}")
print(f"  Min pixel value: {X_test.min()}")
print(f"  Max pixel value: {X_test.max()}")
print(f"  Actions shape: {y_test.shape}")
print(f"  Unique actions: {np.unique(y_test)}")

# Action distribution
print(f"\nAction distribution in training set:")
unique, counts = np.unique(y_train, return_counts=True)
for action, count in zip(unique, counts):
    print(f"  Action {action}: {count} samples ({100*count/len(y_train):.2f}%)")

# Visualize a few random samples
print("\nDisplaying sample images...")
plt.figure(figsize=(15, 3))
num_samples = 10
random_indices = np.random.choice(len(X_train), num_samples, replace=False)

for i, idx in enumerate(random_indices):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(np.squeeze(X_train[idx]), cmap='gray')
    plt.title(f'Action: {y_train[idx]}', fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.savefig('data/sample_images.png', dpi=150, bbox_inches='tight')
print("Sample images saved to 'data/sample_images.png'")
plt.show()
