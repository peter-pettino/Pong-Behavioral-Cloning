# Data processing for PongNoFrameskip-v4 dataset (Mac optimized version)
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_DIR = 'PongNoFrameskip-v4'
PROCESSED_DATA_DIR = 'data/processed'
IMG_SIZE = (84, 84)

# Ensure the processed data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Build a global cache of (folder, filename) -> full path mapping for fast lookups
def build_image_path_cache(base_dir):
	"""Build a dictionary mapping (folder_name, filename) tuples to their full paths."""
	print("Building image path cache...")
	cache = {}
	for root, dirs, files in os.walk(base_dir):
		folder_name = os.path.basename(root)
		for filename in files:
			if filename.endswith('.png'):
				# Store the full path using (folder, filename) as key
				cache[(folder_name, filename)] = os.path.join(root, filename)
	print(f"Cached {len(cache)} image paths")
	return cache

def preprocess_image(img_path, base_dir, path_cache):
	"""Load and preprocess an image using the path cache for fast lookups."""
	# Try direct path first (if it somehow exists)
	if os.path.exists(img_path):
		img = Image.open(img_path).convert('L').resize(IMG_SIZE)
		img = np.array(img, dtype=np.float32) / 255.0
		return img
	
	# Extract the folder name and filename from the Windows path
	# Windows path format: C:\...\PongNoFrameskip-v4\PongNoFrameskip-v4-recorded_images-X\Y.png
	# First normalize path separators to forward slashes
	normalized_path = img_path.replace('\\', '/')
	parts = normalized_path.split('/')
	
	# Extract filename (last part after splitting)
	img_filename = parts[-1] if parts else ''
	
	# Find the folder name (looks like PongNoFrameskip-v4-recorded_images-X)
	folder_name = None
	for part in parts:
		if part.startswith('PongNoFrameskip-v4-recorded_images-'):
			folder_name = part
			break
	
	# Look up in cache using (folder, filename) tuple
	if folder_name and (folder_name, img_filename) in path_cache:
		img = Image.open(path_cache[(folder_name, img_filename)]).convert('L').resize(IMG_SIZE)
		img = np.array(img, dtype=np.float32) / 255.0
		return img
	
	raise FileNotFoundError(f"Image not found: {folder_name}/{img_filename}")

def load_npz_data(npz_path, base_dir, path_cache):
	import warnings
	data = np.load(npz_path, allow_pickle=True)
	obs_paths = data['obs']
	images = []
	idxs = []
	for i, obs in enumerate(obs_paths):
		try:
			img = preprocess_image(obs, base_dir, path_cache)
			images.append(img)
			idxs.append(i)
		except FileNotFoundError:
			warnings.warn(f"Image not found, skipping: {obs}")
	# Only keep data for found images
	def filter_arr(arr):
		arr = np.array(arr)
		if arr.shape[0] == len(obs_paths):
			return arr[idxs]
		return arr  # episode_returns may be per-episode, not per-frame
	return {
		'images': np.stack(images) if images else np.empty((0, *IMG_SIZE)),
		'model_selected_actions': filter_arr(data['model selected actions']),
		'taken_actions': filter_arr(data['taken actions']),
		'rewards': filter_arr(data['rewards']),
		'episode_returns': data['episode_returns'],
		'episode_starts': filter_arr(data['episode_starts']),
		'repeated': filter_arr(data['repeated'])
	}

def collect_dataset(data_dir):
	# Build the image path cache once at the start
	path_cache = build_image_path_cache(data_dir)
	
	all_data = {
		'images': [],
		'model_selected_actions': [],
		'taken_actions': [],
		'rewards': [],
		'episode_returns': [],
		'episode_starts': [],
		'repeated': []
	}
	
	# Get all .npz files and sort them for consistent processing
	npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
	total_files = len(npz_files)
	
	for idx, fname in enumerate(npz_files, 1):
		npz_path = os.path.join(data_dir, fname)
		print(f'Processing {idx}/{total_files}: {fname}')
		d = load_npz_data(npz_path, data_dir, path_cache)
		for k in all_data:
			all_data[k].append(d[k])
	
	# Concatenate all arrays
	print("\nConcatenating all data...")
	for k in all_data:
		all_data[k] = np.concatenate(all_data[k], axis=0)
	return all_data

def main():
	print("Starting data processing (Mac optimized version)...")
	dataset = collect_dataset(DATA_DIR)
	
	# Convert to numpy arrays
	X = dataset['images']
	y = dataset['model_selected_actions']
	
	# Flatten y if it has an extra dimension
	if len(y.shape) > 1:
		y = y.flatten()
	
	# Add a channel dimension for the CNN (batch, height, width) -> (batch, height, width, channels)
	X = np.expand_dims(X, axis=-1)
	
	print(f"\nTotal samples collected: {X.shape[0]}")
	print(f"Image shape: {X.shape[1:]}")
	print(f"Actions shape: {y.shape}")
	print(f"Unique actions: {np.unique(y)}")
	
	# Split into train/test (80/20) with stratification to maintain action distribution
	try:
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
		)
		print("Applied stratified split to maintain action distribution")
	except ValueError:
		# If stratification fails (e.g., too few samples per class), do regular split
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.2, random_state=42, shuffle=True
		)
		print("Applied regular split (stratification not possible)")
	
	# Save processed data to the processed directory
	processed_file_path = os.path.join(PROCESSED_DATA_DIR, 'pong_dataset.npz')
	np.savez_compressed(
		processed_file_path,
		X_train=X_train,
		y_train=y_train,
		X_test=X_test,
		y_test=y_test
	)
	
	print(f"\nDataset processed and saved to {processed_file_path}")
	print(f"Training data shape: {X_train.shape}")
	print(f"Test data shape: {X_test.shape}")
	print(f"Training labels shape: {y_train.shape}")
	print(f"Test labels shape: {y_test.shape}")

if __name__ == '__main__':
	main()
