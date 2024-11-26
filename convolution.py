import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F


def perform_convolution(kernel, image):
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0)
    kernel = kernel.expand(1, 3, 3, 3)

    convolved_image = F.conv2d(image_tensor, kernel, padding=1)
    convolved_image = convolved_image.squeeze().detach().numpy()

    return convolved_image


def plot_images(image, vertical_conv, horizontal_conv):
    plt.figure(figsize=(18, 6))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Vertical line convolution
    plt.subplot(1, 3, 2)
    plt.imshow(vertical_conv, cmap="gray")
    plt.title("Vertical Line Convolution")
    plt.axis("off")

    # Horizontal line convolution
    plt.subplot(1, 3, 3)
    plt.imshow(horizontal_conv, cmap="gray")
    plt.title("Horizontal Line Convolution")
    plt.axis("off")

    plt.show()
