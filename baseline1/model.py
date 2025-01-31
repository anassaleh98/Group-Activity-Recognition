import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_preparation import train_dataset, trainloader, valloader
from utils import set_seed

# seed 
set_seed(42)

# Step 5: Load Pretrained ResNet50 Model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Modify the final fully connected layer to match the number of classes
num_classes = 8
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 8)

# Freeze Layers (layer1)
for layer in [model.layer1]:
    for param in layer.parameters():
        param.requires_grad = False


# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Step 6: Define Loss Function and Optimizer
# Calculate class weights as inverse frequency
label_counts = Counter([label.item() for _, label in train_dataset])  
total_samples = len(train_dataset)
# Calculate normalized weights for the 8 classes
class_weights = [total_samples / label_counts[i] if i in label_counts else 0 for i in range(num_classes)]

# Convert weights to a tensor and move to the appropriate device
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Define the Loss Function with Weights
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),  lr=0.001)

# Step 6b: Add a learning rate scheduler to decay the learning rate by 0.1 every 20 epochs
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


# Step 7: Train the model
num_epochs = 35

def train():
    for epoch in range(num_epochs): 
        model.train()  
        running_loss = 0.0
        total_correct = 0  # To accumulate correct predictions across all epochs
        total_samples = 0  # To accumulate total samples across all epochs
        all_targets = []
        all_predictions = []
        
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)  # Move data to device (GPU)
            
            optimizer.zero_grad()  # Zero the gradients
            output = model(data)  # Forward pass
            
            loss = criterion(output, target)  # Calculate the loss
            loss.backward()  # Backward pass to calculate gradients
            optimizer.step()  # Update model parameters
            
            running_loss += loss.item()  # Accumulate loss
            
            
            # Calculate total accuracy
            _, predicted = output.max(1)  # Get the index of the max log-probability
            total_samples += target.size(0)  # Increment the total number of samples
            total_correct += predicted.eq(target).sum().item()  # Increment the number of correct predictions
                        
            # Collect all targets and predictions for F1 score calculation
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            
        # Step the learning rate scheduler 
        scheduler.step()
        
        # Calculate average loss and accuracy
        avg_loss = running_loss / len(trainloader)
        epoch_accuracy = 100. * total_correct / total_samples
        
        # Calculate F1 score
        epoch_f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, F1 Score: {epoch_f1:.4f}")
        
        # Validation 
        if (epoch+1) == num_epochs :
            evaluate(valloader, "Validation")


# Step 8: Evaluate the model
def evaluate(dataloader, desc):
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            eval_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            # Collect all targets and predictions for F1 score calculation
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    avg_loss = eval_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    # Calculate F1 score
    eval_f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    print(f"{desc} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {eval_f1:.4f}")


# step 9 :confusion_matrix
def create_confusion_matrix(dataloader): 
    # Define class names
    class_names = ['r_set', 'r_spike', 'r-pass', 'r_winpoint', 
                   'l_winpoint', 'l-pass', 'l-spike', 'l_set']
    
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            # Collect true and predicted labels
            all_true_labels.extend(target.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    
    # Convert to percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix (%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print Classification Report
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_pred_labels, 
                              target_names=class_names, 
                              digits=3))    