import torch #type:ignore
import torch.nn as nn #type:ignore
import torch.optim as optim #type:ignore
import torchvision #type:ignore
import torchvision.transforms as transforms #type:ignore
from torch.utils.data import DataLoader, Dataset #type:ignore
import numpy as np #type:ignore
from load_quickdraw import load_quickdraw_data

'''
Purpose: Defines the QuickDrawDataset (to wrap the data into a PyTorch-compatible format), the DoodleCNN model, and the training loop.
Interaction: Imports the load_quickdraw_data function from load_quickdraw.py to get the data, then feeds it into QuickDrawDataset and DataLoader for training the CNN.

Flow:

model.py calls load_quickdraw_data to get preprocessed data.
The data is passed to QuickDrawDataset to create a PyTorch dataset.
DataLoader batches the dataset for training.
The CNN model processes the batched data, computes loss, and updates weights.

How the files interact in the code:

Import: model.py imports load_quickdraw_data from load_quickdraw.py.
Data Loading: In main(), load_quickdraw_data is called with the 10 categories, returning NumPy arrays.
Dataset Creation: The arrays are passed to QuickDrawDataset to create PyTorch datasets.
Training: DataLoader feeds batched data to DoodleCNN for training.
'''
class QuickDrawDataset(Dataset):
    def __init__(self, data, labels, transform = None):
        """
        Custom Dataset for Quick, Draw! images.
        
        Args:
            data (np.ndarray): Shape [N, 28, 28], grayscale images.
            labels (np.ndarray): Shape [N], integer labels (0 to 9).
            transform: PyTorch transforms to apply.
        """
        self.data = data # Numpy array of shape (N, 28, 28)
        self.labels = labels # Numpy array of shape (N,)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # get image and label 
        image = self.data[idx]
        label = self.labels[idx]
        # image = image.astype(np.float32)/255
        # image = image[:,:, np.newaxis] # Adds a channel dimension, making the shape [28, 28, 1]. This prepares the image for ToTensor
        
        if self.transform:
            image = self.transform(image)
        #     # Applies the transform (e.g., ToTensor, Normalize) if provided, converting the image to a tensor [1, 28, 28]
            
        return image, label #Returns a tuple of the transformed image (tensor) and label (scalar)
    
class DoodleCNN(nn.Module):
    def __init__(self, num_classes = 10):
        super(DoodleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 28, 3, padding = 1) # nn.Conv2d(in_channels, out_channels, kernel_size, ...)
        self.pool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(28, 56, 3, padding = 1)
        self.fc1 = nn.Linear(56 * 7 * 7, 128)  # Flatten , then to 10 classes  
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)  # Fix: Assign the result of view
        x = self.fc1(x)    
        x = self.relu(x)
        x = self.fc2(x)    
        
        return x

def main():
    # Define categories
    categories = ["cat", "dog", "bird", "fish", "tree", "flower", "car", "house", "sun", "moon"]
    
    # Load data
    (train_data, train_labels), (val_data, val_labels) = load_quickdraw_data(categories)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts [28, 28, 1] to [1, 28, 28]
        transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale
    ])
    
    # Create train and validation datasets 
    train_dataset = QuickDrawDataset(train_data, train_labels, transform=transform)
    val_dataset = QuickDrawDataset(val_data, val_labels, transform=transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoodleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()