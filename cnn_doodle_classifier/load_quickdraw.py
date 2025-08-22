'''
Purpose: Loads and preprocesses the Quick, Draw! dataset (.npy files for your 10 categories: Cat, Dog, Bird, Fish, Tree, Flower, Car, House, Sun, Moon).
Output: Returns NumPy arrays for training and validation data (train_data, train_labels, val_data, val_labels) with shapes like [N, 28, 28] for images and [N] for labels.
Role: Provides raw data that model.py will use to create PyTorch Dataset and DataLoader objects for training.
'''

import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

def load_quickdraw_data(categories, data_dir = "/data/Projects/cnn_projects/cnn_doodle_classifier/datasets", samples_per_category = 5000):
    """
    Load Quick, Draw! dataset for given categories and return train/validation splits.
    
    Args:
        categories (list): List of category names (e.g., ["cat", "dog", ...]).
        data_dir (str): Directory containing .npy files.
        samples_per_category (int): Number of samples to load per category.
    
    Returns:
        tuple: (train_data, train_labels), (val_data, val_labels)
    """
    data = []
    labels = []
    
    for idx, category in enumerate(categories):
        file_path = f"{data_dir}/{category}.npy"
        category_data = np.load(file_path)[:samples_per_category]
        category_data = category_data.reshape(-1, 28, 28)
        category_labels = np.full(len(category_data), idx)
        
        data.append(category_data)
        labels.append(category_labels)
        
    data = np.concatenate(data, axis = 0)
    labels = np.concatenate(labels, axis = 0)
    
    train_data, vala_data, train_labels, val_labels = train_test_split(data, labels, test_size = 0.2, random_state=42)
    
    return (train_data, train_labels), (vala_data, val_labels)