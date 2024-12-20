![MonkeyMadness](https://github.com/user-attachments/assets/bfcf152a-8cce-458d-8f32-576f164ea3f7)
# MonkeyMadness



# What is a convolution? 


The convolution is like sliding a small "window" (called a kernel or filter) over an image to look for patterns.

Here's how it works step-by-step:

Kernel: Think of this as a small grid of numbers (e.g., 3x3).
Slide and Multiply: Place the kernel on part of the image. Multiply the numbers in the kernel with the corresponding numbers in the image under it.
Sum Up: Add the results of the multiplication together. This gives one number for that position.
Move the Kernel: Slide the kernel to the next part of the image and repeat.
The result is a new image (called a feature map) that highlights certain patterns like edges or textures.



# Supervised Learning

In this task, you will use supervised learning to classify images of monkeys. Supervised learning is one of the three main areas of machine learning.  

Supervised learning is like teaching the computer to recognize monkeys by using flashcards.  

1. **Labeled Data**: You show the computer images of monkeys and tell it which species each one is (e.g., "This is a chimpanzee").  
2. **Training**: The computer studies these examples to learn patterns.  
3. **Comparison**: It makes a prediction of the image and then compares it to the labeled data. It will be penalized based on how "wrong" it is.  
4. **Prediction**: Once trained, it can look at a new image and guess the species of the monkey based on what it learned.  

It’s called “supervised” because the model learns under guidance (the labeled data).  

## There are some main ingredients:

1. **Data**  
    - The data has to be labeled, i.e., someone has to manually note down what monkey is present in the image.  

2. **Model**  
    - It is the model that makes predictions. It does so by looking at the input and making a guess about what monkey is present. The model consists of mathematical operations and weights (these are adjustable). In this task we will use so called Convolutional Neural Network which are very good at handling image data.

3. **Loss Function**  
    - Somehow we need to tell the computer how wrong the guess is.  
    - The model will output probabilities for each monkey class. Let’s say it sees an image of an orangutan, then it could perhaps output the following:  

        ```markdown
        Chimpanzee: 0.70 (70%)  
        Orangutan: 0.20 (20%)  
        Other monkeys: 0.10 (10%)  
        ```

        The correct label is orangutan (100% probability), but the model guessed only 20% for this class.

        Using a loss function like Cross-Entropy Loss, the score is calculated to show how wrong the prediction is
        The closer the prediction is to 1 (100%), the smaller the loss. 
        The goal is to adjust the model to make higher confidence predictions for the correct class.
4. **Optimizer**
    - Somehow we need to adjust the model to perform better. This is done by calculating, based on the loss how the model should be adjusted.


# BUDGET: 100 Billion Nerual nuggets

## Shop
| Upgrade                           | Cost         |
|-----------------------------------------|--------------|
| Info Infusion             | 1 B NN per % added data      |
| Resize Rocket                                       | 10 B NN for 2x increase in size       |
| Whirl and swirl                         | 10 B NN      |
| Foggy Lens                                      | 5 B NN       |
| Flip trick                              | 10 B NN      |
| Need for shift: Tokyo data drift        | 5 B NN       |
| Cutout mask                             | 5 B NN      |
| Ensemble enchanter                      | 60 B NN      |
| Neural boost                      | 20 B NN  per layer. Buy 3 for the price of 2    |
| Drop shield                             | 10 B NN      |
| Weigth decay                                      | 10 B NN      |
| Herr Nilsson's friend                   | Free            |
| Wisdom extractor                        | 40 B NN            |
| Speed boost                        | 10 B NN            |
| Focus Lens                       | 15 B NN            |
| Skip Connection                       | 15 B NN       |
| Hypothesis hustle | 20 B NN           |


# Things to try out for yourself
- Batch size
- Epochs 
- Learning rate
- Kernel & Maxpooling sizes
- Optimizers
- Batch normalization



# Instructions on how to implement things from the shop

</details>

<details>
<summary><strong> Implement Augmentations </strong> </summary>


  In order to implement augmentations for the data one needs to change the cell called "DATASET"  
  
  ```python
transform = transforms.Compose([ 
      #Put augmentations here preferably
      transforms.ToTensor(), 
      transforms.Resize((IMAGE_SIZE[0], IMAGE_SIZE[1]))
      ])
  ```

### Whirl and Swirl

```python
transforms.RandomRotation(degrees=15),  # Rotate the image randomly within a 15-degree range
```

### Flip trick

```python
transforms.RandomHorizontalFlip(p=0.2),  # Randomly flip the image horizontally with 20% probability
transforms.RandomVerticalFlip(p=0.2),  # Randomly flip the image vertically with 20% probability
```

### Need for shift: Tokyo data drift

```python
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)) # Randomly shift image by up to 10% of its size

```

### Foggy Lens  

```python
transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Apply Gaussian blur with a random kernel size

```

### Missing Pieces (Cutout Chaos) 

```python
transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),  # Randomly erases parts of the image
```
  </details>     </details>


</details>




</details>




</details>

<details>
<summary><strong> Implement Info infusion </strong> </summary>

Change the variable ```DATA_PERCENTAGE``` to 1 in the dataset code block


</details>


</details>

<details>
<summary><strong> Implement Ensemble Enchanter </strong> </summary>

This one is a bit more tricky....

Ensembling combines predictions from multiple models to improve accuracy and stability. By merging outputs (through averaging, voting, or stacking), ensembles reduce individual model errors, leading to more robust and reliable predictions, especially on complex tasks.

You could implement ensembling by creating a class that takes a list of models and combines their predictions. This class could run each model independently, then combine their outputs through  majority voting.

You could start by adding the following code to the MODEL block.

```python
class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        # Store the list of models
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # Get predictions from each model and store them in a list        
        outputs = [F.softmax(model(x), dim=1) for model in self.models]

        # Take the average of them
        output = torch.mean(torch.stack(outputs), dim=0)
        
        return output
```

The next step is to train a few models, preferably with different hyperparameters (to introduce some variability) and add them to a list.

For this you need to create different optimizers for each model, since these hold the model parameters in them. So if you want three models in your ensemble it could look like this: (add these lines to the TRAINING code block)


```python

model1 = MonkeyNET(num_classes=NUM_OF_CLASSES, input_size=IMAGE_SIZE)
optimizer1 = torch.optim.SGD(model1.parameters(), lr=LR)

model2 = MonkeyNET(num_classes=NUM_OF_CLASSES, input_size=IMAGE_SIZE)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=LR)


model3 = MonkeyNET(num_classes=NUM_OF_CLASSES, input_size=IMAGE_SIZE)
optimizer3 = torch.optim.SGD(model1.parameters(), lr=LR)


model1.to(device)
model1, t_loss, t_acc, v_loss, v_acc = train(model1, train_loader, val_loader, optimizer1, criterion, device, start_epoch=START_EPOCH, num_epochs=EPOCHS, model_name=MODEL_NAME, unique_id=ID)


model2.to(device)
model2, t_loss, t_acc, v_loss, v_acc = train(model2, train_loader, val_loader, optimizer2, criterion, device, start_epoch=START_EPOCH, num_epochs=EPOCHS, model_name=MODEL_NAME, unique_id=ID)


model3.to(device)
model3, t_loss, t_acc, v_loss, v_acc = train(model3, train_loader, val_loader, optimizer3, criterion, device, start_epoch=START_EPOCH, num_epochs=EPOCHS, model_name=MODEL_NAME, unique_id=ID)


models = [model1, model2, model3]
```

Then run the ensemble on the validation data!


```

ensemble = Ensemble(models)

acc = test(model=ensemble, testloader=val_loader, device=device, model_name="ensemble", unique_id=ID)


```

Then to save it (do this it has trained in the TRAINING BLOCK): 

```
# Export the model to ONNX format
            dummy_input = torch.randn(1, 3, IMAGE_SIZE(0), IMAGE_SIZE(0))  # Example shape (batch_size, channels, height, width)

            torch.onnx.export(
                ensemble,
                dummy_input,
                "saved_models/MEGA_ENSEMBLE.onnx",
                verbose=True,
                input_names=["input"],
                output_names=["output"],
                opset_version=11,  # Use appropriate ONNX opset version
            )
```


</details>


<details>
<summary><strong> Implement Neural turbo boost </strong> </summary>

Why add more layers? Adding a layer to a Convolutional Neural Network (CNN) increases the model’s depth, allowing it to learn more complex features from the input data. New layers, like convolutional, pooling, or fully connected layers, enhance the network's ability to capture patterns such as edges, textures, or object parts. Adding layers can improve model performance but also increases computational requirements and the risk of overfitting.

For this you should change the MonkeyNET. You need to add the convolutional layer to the constructor, the forward method and the get_fc_input_size method. (This is done in the MODEL code block)

For the constructor add a Conv2d as following:

```python

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=10, input_size=(500, 500)):
        super(ClassificationModel, self).__init__()
        
        # First convolutional layer: 3 input channels (RGB), 32 output channels, kernel size 5, padding 2 to preserve size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)

```

To note is that the in_channels must match with the previous layers out_channel

For the forward method:

```python

def forward(self, x):
        # First conv -> ReLU -> Max Pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Second conv -> ReLu -> Max Pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
```

Make sure that you add the activation (ReLu) and maxpooling. Here you can experiment with the pooling parameters if you add more layers

For the get_fc_input_size() (This method calculates how large the fully connected layer input should be):

```python

def _get_fc_input_size(self, input_size):
        x = torch.zeros(1, 3, *input_size)  # Create a dummy input tensor
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        return x.numel()  # Total number of elements after conv layers

```
</details>


<details>
<summary><strong> Weight decay </strong> </summary>
What is weight decay? Weight decay is a regularization technique used to prevent overfitting in machine learning models by adding a penalty to the model's loss function based on the size of its weights. It works by slightly reducing the weights during training, encouraging simpler models with smaller weights, which often generalize better to new data. This technique is especially useful in neural networks, where complex models can easily overfit to the training data.

The following lines should be added to the TRAINING code block
```python
weight_decay = 1e-4  # Adjust weight decay as needed
optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=weight_decay)
```

Here you can experiment with the weight decay parameter. It controls how much it should penalize large weights.

</details>



</details>

<details>
<summary><strong> Implement Rate Rollercoaster </strong> </summary>

Why have learning rate scheduler?A learning rate scheduler is used to adjust the learning rate during training to improve the performance of a machine learning model. By modifying the learning rate, the scheduler helps to balance the trade-off between convergence speed and stability. A high learning rate can lead to unstable training and overshooting the optimal solution, while a low learning rate can result in slow convergence. Learning rate schedulers can implement strategies such as gradually decreasing the learning rate over time or adjusting it based on performance metrics, allowing the model to escape local minima and achieve better overall accuracy. This dynamic approach enhances training efficiency and often leads to improved model performance.




</details>


</details>

<details>
<summary><strong> Implement Speed Boost </strong> </summary>

Why momentum? Momentum is an optimization technique that helps accelerate gradients vectors in the right directions, thus leading to faster converging. It works by adding a fraction of the previous update to the current update, which helps to smooth out the updates and reduces oscillation, especially in areas with noisy gradients. This technique mimics the physical concept of momentum, where the optimizer retains a memory of past gradients to guide its current direction.

Add momemntum to the optimizer: (TRAINING code block)

```python
momentum = 0.9
optimizer1 = torch.optim.SGD(model1.parameters(), lr=LR, momentum=momentum)
```
</details>


</details>

<details>
<summary><strong> Herr Nilsson's friend</strong> </summary>
If you want to get the title of Herr Nilssons friend you might want to give an extra reward to the model when it makes corrects predictions for the squirrel monkey class. You can do this by: (in the TRAINING block)

```python
class_weights = torch.ones(NUM_OF_CLASSES)
class_weights[7] = 5
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

Remeber that you still want high recall so you still want to predict some of the other monkeys correctly. You will have to experiment with the weighting factor to get a reasoable result.
</details>






<details>
<summary><strong> Implement Drop shield </strong> </summary>




Dropout is a regularization technique that helps prevent overfitting by randomly "dropping out" a proportion of neurons during training. These changes are supposed to be implemented in the MODEL block.

#### 1. Add a dropout_rate parameter to __init__ (default 0.5)
```python
def __init__(self, num_classes=10, input_size=(500, 500), dropout_rate=0.5):
```
#### 2. Add one  dropout layer in the init method

```python
# Dropout layers
        self.dropout1 = nn.Dropout2d(p=dropout_rate)  # Spatial dropout for convolutional layers
```
#### 3. Apply dropout after activation functions but before pooling layers in the forwards method

One convolutional block should then look like this
```python
# First conv -> ReLU -> Dropout -> Max Pooling
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)  # Apply spatial dropout
        x = F.max_pool2d(x, kernel_size=2, stride=2)
```

### You can experiment with how many of these dropout layers you want 


#### 4. Use dropout
To use this model, you can instantiate it with different dropout rates:
```python
# Default dropout rate (0.5)
model = MonkeyNET(num_classes=10)

# Custom dropout rate
model = MonkeyNET(num_classes=10, dropout_rate=0.3)
```

</details>


</details>

<details>
<summary><strong> Implement Wisdom Extractor  </strong> </summary>

# How Knowledge Distillation Works

## 1. Train a Teacher Model
- Start with a large, powerful model (e.g., ResNet, VGG, or a pretrained network).
- Train it on the dataset until it achieves high accuracy. This model becomes the "teacher."
- This model has been trained on millions of images and have learned how to represent images well
- We "finetuned" the model to the monkeys and it achieves approximately 97 % accuracy.

## 2. Generate Soft Labels
- The teacher model produces outputs (logits or probabilities) for each input image.
- These outputs are "soft labels" because they include information about all classes, not just the predicted one (e.g., probabilities for every monkey, not just the most likely one).

## 3. Train the Student Model
- Use a smaller, simpler model (e.g., MonkeyNET) as the "student."
- Train it using two loss functions:
  - **Hard Label Loss:** Cross-entropy between the true labels and the student’s predictions.
  - **Soft Label Loss:** Cross-entropy or KL divergence between the teacher's soft labels and the student’s predictions.
- A weighting factor (**α**) balances these two losses.
- A temperature (**T**) is applied to soften the teacher’s logits, making the soft labels smoother and more informative.

## 4. Outcome
- The student learns not only the final predictions but also the teacher’s nuanced knowledge about class relationships (e.g., "This monkey looks like species A but also resembles species B").

# How to implement it:

## 1. Define the loss DistillationLoss (can be done in the TRAINING code block)
```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, true_labels):
        # Hard label loss (ground truth)
        hard_loss = self.criterion(student_logits, true_labels)

        # Soft label loss (teacher knowledge)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_loss = F.kl_div(student_probs, teacher_probs, reduction="mean") * (self.temperature ** 2)

        # Combine losses
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

```

# Next load the pretrained model: (IN THE TRAINING BLOCK)
```python
teacher_model = models.resnet18(pretrained=False)  # Set pretrained=False since you're loading a custom-trained model
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, NUM_OF_CLASSES)
teacher_model.load_state_dict(torch.load('Pretrained/ResNet18_cuda.pt'))

# set in eval mode:
teacher_model.eval()
teacher_model.to(device)
```

# Next, declare your criterion as the Distiliation loss and send both to the training function: (also in the TRAINING block)
```python
criterion = DistillationLoss()

model, t_loss, t_acc, v_loss, v_acc = train(model, train_loader, val_loader, optimizer, criterion, device, start_epoch=0, num_epochs=EPOCHS, model_name=MODEL_NAME, unique_id=ID, teacher_model = teacher_model)
```


</details>
<details>
<summary><strong> Implement Focus Lens  </strong> </summary>

# **Spatial Attention**

## **What is Spatial Attention?**
Spatial attention is a mechanism in deep learning models designed to emphasize the most important spatial regions in an input feature map. It guides the model to focus on relevant areas, enhancing performance in tasks that require spatial understanding, such as object detection, segmentation, and image recognition.

## **How Does it Work?**
Spatial attention operates on the spatial dimensions (height and width) of a feature map. It identifies where in the feature map the model should focus by creating a spatial attention map, which assigns importance scores to each spatial location.

## **Steps in Spatial Attention**

1. **Pooling Across Channels:**
   - **Average Pooling:** Captures overall spatial context by taking the mean across all channels.
   - **Max Pooling:** Highlights the most prominent features across channels.

2. **Concatenation:**
   - The outputs of average and max pooling are combined along the channel dimension.

3. **Convolution:**
   - A convolution operation (typically with a 7x7 kernel) processes the concatenated output to capture local spatial relationships.

4. **Attention Map Generation:**
   - A sigmoid activation function is applied to generate the spatial attention map, which scales values to the range [0, 1].

5. **Feature Refinement:**
   - The spatial attention map is multiplied element-wise with the input feature map to highlight important regions and suppress irrelevant ones.

# How to implement:

1. **Start by defining the attention mechanism (add these line to the MODEL code block**

```python

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average and Max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate pooled outputs
        combined = torch.cat([avg_out, max_out], dim=1)
        # Convolve and apply sigmoid
        attention_map = self.sigmoid(self.conv(combined))
        return x * attention_map

```

2. **Add it to MonkeyNET after the last convolution (MODEL code block)**

```python

class MonkeyNET(nn.Module):
    def __init__(self, num_classes=10, input_size=(500, 500)):
        super(MonkeyNET, self).__init__()
        
        # First convolutional layer: 3 input channels (RGB), 32 output channels, kernel size 5, padding 2 to preserve size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)

        self.attention = SpatialAttention()
        
        # Calculate the size of the fully connected layer dynamically
        self.fc_input_size = self._get_fc_input_size(input_size)
        self.fc1 = nn.Linear(self.fc_input_size, 10)  # Adjusted for the final size after pooling
        
        # Prediction layer
        
    def _get_fc_input_size(self, input_size):
        x = torch.zeros(1, 3, *input_size)  # Create a dummy input tensor
        x = F.relu(self.conv1(x))
        x = self.attention(x)
        x = F.max_pool2d(x, kernel_size=(8, 8), stride=8)
        return x.numel()  # Total number of elements after conv layers
    
    def forward(self, x):
        # First conv -> ReLU -> Max Pooling
        x = F.relu(self.conv1(x))

        x = self.attention(x)

        x = F.max_pool2d(x, kernel_size=(8, 8), stride=8)

        # Flatten the tensor for fully connected layer
        x = x.view(x.size(0), -1)  # Output: (batch_size, 128 * 16 * 16) for 500x500 input

        # Fully connected layer -> ReLU
        x = self.fc1(x)

       
        return x

```






</details>
<details>
<summary><strong> Skip Connections</strong> </summary>
Residual connections (ResNet architecture) address the vanishing gradient problem in deep neural networks by:

-Allowing direct information flow between layers
-Enabling training of much deeper networks
-Creating "shortcut" paths for gradient flow
-Helping to mitigate the vanishing gradient problem

## Implementation
1. Create a Residual Block Class (perhaps in the modelblock over the current model)
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # Main convolutional path
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += self.shortcut(residual)  # Add shortcut connection
        return F.relu(out)
```

2. Modify Your Model to Use Residual Blocks
```python
class MonkeyNET(nn.Module):
    def __init__(self, num_classes=10, input_size=(500, 500)):
        super(MonkeyNET, self).__init__()
        # Initial layers with residual blocks
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.res_block1 = ResidualBlock(32, 32)
        self.res_block2 = ResidualBlock(32, 32)
        
        # Rest of your model remains similar
        # ...
```


## Key Benefits
- Enables training of deeper networks
- Mitigates vanishing gradient problem
- Improves gradient flow through the network
- Often leads to better performance with increased depth

## Considerations 

- Works best with networks deeper than traditional architectures
- Batch normalization often used in conjunction
- Can be applied to various network architectures

##  Experimental

- Start with 1-2 residual blocks
- Gradually increase network depth
- Monitor validation performance
- Adjust block complexity as needed


<details>
<summary><strong> Example of Skip connections and 2 added layers  </strong> </summary>
```python
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x
        
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        
        # Add shortcut connection
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class MonkeyNET(nn.Module):
    def __init__(self, num_classes=10, input_size=(500, 500)):
        super(MonkeyNET, self).__init__()

        # 4 convolutional layers with progressively increasing channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.res_block1 = ResidualBlock(32, 32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.res_block2 = ResidualBlock(64, 64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.res_block3 = ResidualBlock(128, 128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.res_block4 = ResidualBlock(256, 256)

        # Calculate the size of the fully connected layer dynamically
        self.fc_input_size = self._get_fc_input_size(input_size)
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def _get_fc_input_size(self, input_size):
        x = torch.zeros(1, 3, *input_size)
        
        x = F.relu(self.conv1(x))
        x = self.res_block1(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)
        
        x = F.relu(self.conv2(x))
        x = self.res_block2(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)
        
        x = F.relu(self.conv3(x))
        x = self.res_block3(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)
        
        x = F.relu(self.conv4(x))
        x = self.res_block4(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        return x.numel()

    def forward(self, x):
        # First conv -> Residual Block -> Max Pooling
        x = F.relu(self.conv1(x))
        x = self.res_block1(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Second conv -> Residual Block -> Max Pooling
        x = F.relu(self.conv2(x))
        x = self.res_block2(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Third conv -> Residual Block -> Max Pooling
        x = F.relu(self.conv3(x))
        x = self.res_block3(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Fourth conv -> Residual Block -> Max Pooling
        x = F.relu(self.conv4(x))
        x = self.res_block4(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Flatten the tensor for fully connected layer
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        return x
```


</details>

</details>
