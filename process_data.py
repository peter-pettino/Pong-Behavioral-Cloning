# Data processing for PongNoFrameskip-v4 dataset
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

DATA_DIR = 'PongNoFrameskip-v4'
PROCESSED_DATA_DIR = 'data/processed'
IMG_SIZE = (84, 84)

# Ensure the processed data directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def preprocess_image(img_path, base_dir):
	# Try to load image, convert to grayscale, resize, and normalize to [0, 1]
	if os.path.exists(img_path):
		img = Image.open(img_path).convert('L').resize(IMG_SIZE)
		img = np.array(img, dtype=np.float32) / 255.0
		return img
	# If not found, try to extract the filename and search in local subfolders
	img_filename = os.path.basename(img_path)
	# Try to find the subfolder from the original path (if present)
	parts = img_path.replace('\\', '/').split('/')
	subfolder = None
	for p in parts:
		if p.startswith('PongNoFrameskip-v4-recorded_images-'):
			subfolder = p
			break
	if subfolder:
		local_img_path = os.path.join(base_dir, subfolder, img_filename)
		if os.path.exists(local_img_path):
			img = Image.open(local_img_path).convert('L').resize(IMG_SIZE)
			img = np.array(img, dtype=np.float32) / 255.0
			return img
	# Fallback: search all subfolders for the image
	for root, dirs, files in os.walk(base_dir):
		if img_filename in files:
			img = Image.open(os.path.join(root, img_filename)).convert('L').resize(IMG_SIZE)
			img = np.array(img, dtype=np.float32) / 255.0
			return img
	raise FileNotFoundError(f"Image not found: {img_path}")

def load_npz_data(npz_path, base_dir):
	import warnings
	data = np.load(npz_path, allow_pickle=True)
	obs_paths = data['obs']
	images = []
	idxs = []
	for i, obs in enumerate(obs_paths):
		try:
			img = preprocess_image(obs, base_dir)
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
	all_data = {
		'images': [],
		'model_selected_actions': [],
		'taken_actions': [],
		'rewards': [],
		'episode_returns': [],
		'episode_starts': [],
		'repeated': []
	}
	for fname in os.listdir(data_dir):
		if fname.endswith('.npz'):
			npz_path = os.path.join(data_dir, fname)
			print(f'Processing {npz_path}')
			d = load_npz_data(npz_path, data_dir)
			for k in all_data:
				all_data[k].append(d[k])
	# Concatenate all arrays
	for k in all_data:
		all_data[k] = np.concatenate(all_data[k], axis=0)
	return all_data

def main():
	print("Starting data processing...")
	dataset = collect_dataset(DATA_DIR)
	
	# Convert to numpy arrays
	X = dataset['images']
	y = dataset['model_selected_actions']
	
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