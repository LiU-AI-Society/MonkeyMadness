import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torch
import matplotlib.pyplot as plt
import torchvision
import pandas as pd


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, amount_of_classes=2, data_percentage=1.0):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        self.nr_classes = amount_of_classes
        # Traverse the directory to collect image paths and labels
        for class_idx, class_name in enumerate(os.listdir(root_dir)):
            if class_name == '.DS_Store':
                continue
            if len(self.class_to_idx) >= amount_of_classes:
                break
            
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = class_idx
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.endswith(".jpg") or img_path.endswith(".jpeg"):  # Handle only JPEGs
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
            self.class_names.append(class_name)
        # Use only a portion of the dataset based on data_percentage
        total_images = len(self.image_paths)
        selected_size = int(total_images * data_percentage)
        
        # Randomly select a subset of the data
        if data_percentage < 1.0:
            selected_indices = random.sample(range(total_images), selected_size)
            self.image_paths = [self.image_paths[i] for i in selected_indices]
            self.labels = [self.labels[i] for i in selected_indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        target = torch.zeros(self.nr_classes)
        target[label] = 1

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, target
    
    def visualize(self, nr_images=5):
        """
        Visualize a specified number of images from the dataset with their true labels.
        
        Args:
        - nr_images (int): Number of images to visualize.
        """
        images = []
        labels = []

        # Randomly select 'nr_images' indices
        indices = random.sample(range(len(self.image_paths)), nr_images)

        for idx in indices:
            image, target = self.__getitem__(idx)
            images.append(image)
            
            # Convert the one-hot target tensor back to class index
            label_idx = torch.argmax(target).item()
            labels.append(label_idx)

        # Create a grid of images
        grid_img = torchvision.utils.make_grid(images, nrow=nr_images, normalize=True)

        # Plot the images
        plt.figure(figsize=(12, 6))
        plt.imshow(grid_img.permute(1, 2, 0))  # permute to (height, width, channels)
        plt.axis('off')

        # Add titles below the images showing the true labels
        label_names = [list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(lbl)] for lbl in labels]
        plt.title(" | ".join(label_names))  # Titles are class names for the images
        
        plt.show()

    def visualize_all_classes(self):
        """
        Visualize one random image from each class in the dataset.
        """
        images = []
        labels = []

        # Get one random image from each class
        for class_name, class_idx in self.class_to_idx.items():
            class_images = [i for i, label in enumerate(self.labels) if label == class_idx]
            if class_images:
                random_idx = random.choice(class_images)
                image, target = self.__getitem__(random_idx)
                images.append(image)

                # Convert the one-hot target tensor back to class index
                label_idx = torch.argmax(target).item()
                labels.append(label_idx)

        # Create a grid of images
        grid_img = torchvision.utils.make_grid(images, nrow=len(images), normalize=True)

        # Plot the images
        plt.figure(figsize=(16, 4))
        plt.imshow(grid_img.permute(1, 2, 0))  # permute to (height, width, channels)
        plt.axis('off')

        # Add titles below the images showing the class names
        label_names = [list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(lbl)] for lbl in labels]
        plt.title(" | ".join(label_names))  # Titles are class names for the images
        
        plt.show()


class MonkeyImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, amount_of_classes=10, data_percentage=1.0):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        self.nr_classes = amount_of_classes
        
        # Load class labels from the text file
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'monkey_labels.txt'), header=None, skiprows=1)
        self.labels_df.columns = ['Label', 'Latin Name', 'Common Name', 'Train Images', 'Validation Images']
        
        # Create a mapping from class names to indices
        for class_idx, row in self.labels_df.iterrows():
            class_name = row['Label'].rstrip()
            if class_idx >= amount_of_classes:  # Limit the number of classes
                break
            self.class_to_idx[class_name] = class_idx
            class_dir = os.path.join(root_dir, class_name)
            class_dir = f'{class_dir}/'
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.endswith(('.jpg', '.jpeg', '.png')):  # Handle JPEGs and PNGs
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)
            self.class_names.append(row['Common Name'].rstrip())  # Class index as label

        # Use only a portion of the dataset based on data_percentage
        total_images = len(self.image_paths)
        selected_size = int(total_images * data_percentage)

        # Randomly select a subset of the data
        if data_percentage < 1.0:
            selected_indices = random.sample(range(total_images), selected_size)
            self.image_paths = [self.image_paths[i] for i in selected_indices]
            self.labels = [self.labels[i] for i in selected_indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Create a one-hot encoded target tensor
        target = torch.zeros(self.nr_classes)
        target[label] = 1

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, target
    
    def visualize(self, nr_images=5):
        """Visualize a specified number of images from the dataset with their true labels."""
        images = []
        labels = []

        # Randomly select 'nr_images' indices
        indices = random.sample(range(len(self.image_paths)), nr_images)

        for idx in indices:
            image, target = self.__getitem__(idx)
            images.append(image)
            
            # Convert the one-hot target tensor back to class index
            label_idx = torch.argmax(target).item()
            labels.append(label_idx)

        # Create a grid of images
        grid_img = torchvision.utils.make_grid(images, nrow=nr_images, normalize=True)

        # Plot the images
        plt.figure(figsize=(12, 6))
        plt.imshow(grid_img.permute(1, 2, 0))  # permute to (height, width, channels)
        plt.axis('off')

        # Add titles below the images showing the common names
        common_names = [self.labels_df.iloc[label_idx]['Common Name'].rstrip() for label_idx in labels]
        plt.title(" | ".join(common_names))  # Titles are common names for the images
        
        plt.show()

    def visualize_all_classes(self):
        """Visualize one random image from each class in the dataset."""
        images = []
        labels = []
        common_names = []  # List to store common names

        # Get one random image from each class
        for class_name, class_idx in self.class_to_idx.items():
            class_images = [i for i, label in enumerate(self.labels) if label == class_idx]
            if class_images:
                random_idx = random.choice(class_images)
                image, target = self.__getitem__(random_idx)
                images.append(image)

                # Convert the one-hot target tensor back to class index
                label_idx = torch.argmax(target).item()
                labels.append(label_idx)

                # Fetch the common name from the DataFrame
                common_name = self.labels_df.loc[label_idx, 'Common Name'].rstrip()  # Adjust index based on your DataFrame structure
                common_names.append(common_name)

        # Create a grid of images
        grid_img = torchvision.utils.make_grid(images, nrow=len(images), normalize=True)

        # Plot the images
        plt.figure(figsize=(16, 4))
        plt.imshow(grid_img.permute(1, 2, 0))  # permute to (height, width, channels)
        plt.axis('off')

        # Add titles below the images showing the class names and common names
        for i in range(len(labels)):
            class_name = list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(labels[i])]
            common_name = common_names[i]
            # Calculate position for the text (x position and y position above the image)
            x_position = i * (grid_img.size(2) // len(labels)) + (grid_img.size(2) // (2 * len(labels)))  # Centered above each image
            y_position = -10  # Position text above the image
            
            plt.text(x_position, y_position, f"{common_name}", ha='center', va='bottom', fontsize=10, color='black')

        plt.show()



if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((500, 500)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Example: Using 50% of the data and limiting to 5 classes
    data = CustomImageDataset('data/raw-img', transform=transform, amount_of_classes=5, data_percentage=0.5)

    data.visualize(10)

