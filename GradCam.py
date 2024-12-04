import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
import os


# Function to register hook and get gradients from a specific layer (conv4)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        # Register hooks
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, class_idx, input_tensor):
        self.model.eval()
        
        # Ensure the input tensor requires gradients
        #input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)
        output = F.softmax(output, dim=1)

        # Get the class score for the target class
        class_score = output[:, class_idx]

        # Backward pass
        self.model.zero_grad()
        class_score.backward(retain_graph=True)

        # Compute Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().detach().numpy(), 0)
        heatmap /= np.max(heatmap).squeeze()
        return heatmap
    

def save_grad_cam_heatmap(cam, img, file_path, str_pred=None, str_target=None, batch_accuracy=None, epoch=None):
    # Resize the heatmap to match the input image size
    heatmap = cv2.resize(cam, (img.shape[0], img.shape[1]))

    # Normalize and convert heatmap to 8-bit color image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply the JET color map
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)      # Convert BGR (OpenCV) to RGB (Matplotlib)

    # Process the input image
   # Detach and transpose to (H, W, C)
    img = (img - img.min()) / (img.max() - img.min())       # Normalize the image to range [0, 1]
    img = np.uint8(255 * img)                               # Convert to uint8 for overlay


    # Combine the heatmap with the image
    overlay = cv2.addWeighted(heatmap, 0.3, img, 0.7, 0)

    # Create a figure and axes for displaying the image and colorbar
    fig, (ax_img, original_im, ax_colorbar) = plt.subplots(1, 3, figsize=(12, 6), gridspec_kw={'width_ratios': [4, 4, 0.1]})

    # Display the image with heatmap overlay
    ax_img.imshow(overlay)
    ax_img.axis('off')  # Hide axis for the overlay image

    # Display the original image
    original_im.imshow(img)
    original_im.axis('off')  # Hide axis for the original image



    # Add title with prediction, ground truth, batch accuracy, and epoch information
    title = "Where does the model look to classify?"
    

    # Set the title with a background for contrast
    ax_img.set_title(title, color='white', fontsize=14, weight='bold',
                     bbox=dict(facecolor='black', alpha=0.7))

    # Display the colorbar for the overlay (heatmap)
    norm = plt.Normalize(vmin=0, vmax=255)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    cbar = plt.colorbar(sm, cax=ax_colorbar)
    cbar.set_label('Heatmap Intensity', rotation=270, labelpad=15)

    # Save the figure as a PNG file
    #plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    #plt.close(fig)
    plt.show()




def get_one_example_per_class(test_loader, num_classes):
    examples = {}
    for inputs, targets in test_loader:

        for i, target in enumerate(targets):
            if target.item().argmax(1) not in examples:
                examples[target.item().argmax(1)] = inputs[i]
            if len(examples) == num_classes:
                return examples
    return examples    

def generate_and_save_grad_cams(model, grad_cam, test_loader, classes, save_dir, device, index):
    # Ensure the model is in evaluation mode and on the correct device
    model.to(device)
    model.eval()

    # Get one example per class
    examples = test_loader.dataset[index]

    # Generate Grad-CAM for each example
    example = examples[0]
    print(examples[1])
    class_idx = examples[1].argmax()
    print(class_idx)
    
    input_tensor = example.unsqueeze(0).to(device)  # Add batch dimension and move to device
    target_class = class_idx
    pred_label = target_class  # For visualization, use the same class as prediction

    # Generate Grad-CAM
    cam = grad_cam.generate_cam(class_idx=target_class, input_tensor=input_tensor)
    img = input_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format

    str_target = classes[target_class]
    str_pred = classes[pred_label]


    # Save Grad-CAM visualization
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = f"{save_dir}/grad_cam_class_{target_class}.png"
    save_grad_cam_heatmap(cam, img, save_path, str_pred=str_pred, str_target=str_target)


if __name__ == "__main__":
    from torchvision import transforms, utils
    from Dataset import CustomImageDataset, MonkeyImageDataset

    import torch.nn as nn
    import torch.nn.functional as F
    import torch
    from torchsummary import summary
    from train import train, training_info
    from test_model import test
    import torch.optim
    from torch.utils.data import DataLoader, random_split
    import torch.nn as nn
    import torchvision.models as models
    import datetime
    current_datetime = datetime.datetime.now()
    NUM_OF_CLASSES = 10
    IMAGE_SIZE = (64,64) # DO NOT ALTER THIS PARAMETER

    DATA_PERCENTAGE = 0.7
    transform = transforms.Compose([
        #Randomly flip the images vertically
        #transforms.RandomVerticalFlip(p=0.2),  # Randomly flip the image vertically with 20% probability
        #transforms.RandomHorizontalFlip(p=0.2),  # Randomly flip the image horizontally with 20% probability
        #transforms.RandomRotation(degrees=15),  # Rotate the image randomly within a 15-degree range
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, etc.
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE[0], IMAGE_SIZE[1])), 

        #for imagenet
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = MonkeyImageDataset('Monkey/training/training', transform, NUM_OF_CLASSES, data_percentage = DATA_PERCENTAGE )
    #dataset.visualize(5)

    class MonkeyNET(nn.Module):
        def __init__(self, num_classes=10, input_size=(500, 500)):
            super(MonkeyNET, self).__init__()

            # First convolutional layer: 3 input channels (RGB),  output channels, kernel size 5, padding 2 to preserve size
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
            # Calculate the size of the fully connected layer dynamically
            self.fc_input_size = self._get_fc_input_size(input_size)
            self.fc = nn.Linear(self.fc_input_size, num_classes) # Adjusted for the final size after pooling
            self.fc2 = nn.Linear(num_classes, num_classes)


        def _get_fc_input_size(self, input_size):
            x = torch.zeros(1, 3, *input_size) # Create a dummy input tensor
            #x = self.attention(x)
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, kernel_size=(4, 4), stride=4)

            x = F.relu(self.conv2(x))

            x = F.max_pool2d(x, kernel_size=(4, 4), stride=4)



            return x.numel() # Total number of elements after conv layers

        def forward(self, x):
            # First conv -> ReLU -> Max Pooling
            x = F.relu(self.conv1(x))


            x = F.max_pool2d(x, kernel_size=(4, 4), stride=4)

            x = F.relu(self.conv2(x))

            x = F.max_pool2d(x, kernel_size=(4, 4), stride=4)

            # Flatten the tensor for fully connected layer
            x = x.view(x.size(0), -1) 

            # Fully connected layer -> ReLU
            x = self.fc(x)

            x = F.relu(x)

            x = self.fc2(x)

            return x
        
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)  # 80% for training
    val_size = dataset_size - train_size   # 20% for validation
    criterion = nn.CrossEntropyLoss() #This is the loss function

    model = MonkeyNET(num_classes=NUM_OF_CLASSES, input_size=IMAGE_SIZE).to('mps')


    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    grad_cam = GradCAM(model, model.conv2)

    generate_and_save_grad_cams(model, grad_cam, val_loader, ["p", "a", "p", "a","p", "a","p", "a","p", "a"], save_dir="grads", device='mps', index=np.random.randint(len(val_loader)))