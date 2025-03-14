import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_preparation import  train_dataset
from utils import set_seed

# seed 
set_seed()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 5: VideoClassifier model
class VideoClassifier(nn.Module):
    def __init__(self, num_classes=8, lstm_hidden=512, lstm_layers=1):
        super(VideoClassifier, self).__init__()
        
        # Load Pretrained ResNet50
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
                
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=2048, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True)
        
        # Fully Connected Layer for Classification
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(lstm_hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, frames, C, H, W = x.shape  # [Batch, Frames, Channels, Height, Width]
        
        # Reshape for ResNet
        x = x.view(batch_size * frames, C, H, W)  # Merge batch and time for ResNet
        
        # Feature Extraction
        features = self.feature_extractor(x)  # [Batch*Frames, 2048, 1, 1]
        features = features.view(batch_size, frames, -1)  # Reshape to [Batch, Frames, 2048]
        
        # LSTM
        lstm_out, _ = self.lstm(features)  # [Batch, Frames, Hidden]
        
        # Use the last time step's output
        last_out = lstm_out[:, -1, :]  # [Batch, Hidden]
        
        # Fully Connected Layer
        output = self.fc(last_out)  # [Batch, Num_classes]
        
        return output

# Step 6: Define Loss Function
def calculate_class_weights(train_dataset):
    label_counts = Counter([label.item() for _, label in train_dataset])  
    total_samples = len(train_dataset)
    # Calculate normalized weights for the 8 classes
    class_weights = [total_samples / label_counts[i] if i in label_counts else 0 for i in range(8)]
    
    # Convert weights to a tensor and move to the appropriate device
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    return weights


# Calculate class weights
class_weights = calculate_class_weights(train_dataset)
# Define Loss
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Step 7: Train the model
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    # Define Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),  lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    accumulation_steps = 32  # Effective batch: 4*32=128
    model.to(device)

    for epoch in range(epochs):
        # Training loop
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        all_true_labels = []
        all_pred_labels = []
        optimizer.zero_grad()  # Reset gradients at the start of each epoch

        for step, (videos, labels) in enumerate(train_loader):  # Assume train_loader provides [batch, frames, C, H, W]
            videos, labels = videos.to(device), labels.to(device)  # Move data to GPU  
            
            outputs = model(videos)  # Forward pass
        
            loss = criterion(outputs, labels) / accumulation_steps  # Scale loss

            loss.backward()  # Backpropagation (gradients accumulate
            
            if (step + 1) % accumulation_steps == 0 or step == len(train_loader) - 1: 
                optimizer.step()  # Update model weights
                optimizer.zero_grad()  # Reset gradients
            
            # Statistics
            epoch_loss += loss.item()* accumulation_steps  # Restore scaled loss
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Collect true and predicted labels for F1 score calculation
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())
        
        # Step the learning rate scheduler 
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total
        train_f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')  # F1 score for training
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, F1 Score: {train_f1:.4f}")

        # Validation 
        if (epoch+1) == epochs :
            print("Validation ", end="")
            validate_model(model, val_loader)


# Step 8: Evaluate the model 
def validate_model(model, val_loader):
    model.eval()  # Set to evaluation mode
    epoch_loss, correct, total = 0, 0, 0
    all_true_labels, all_pred_labels = [], []
    
    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)  # Move to GPU
            
            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)  # Compute loss
            
            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())

    avg_epoch_loss = epoch_loss / len(val_loader)
    validation_accuracy = 100.0 * correct / total
    val_f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')  
    print(f"Loss: {avg_epoch_loss:.4f}, Accuracy: {validation_accuracy:.2f}%, F1 Score: {val_f1:.4f}")

# step 9 :confusion_matrix
def create_confusion_matrix(model, dataloader): 
    # Define class names
    class_names = ['r_set', 'r_spike', 'r-pass', 'r_winpoint', 
                   'l_winpoint', 'l-pass', 'l-spike', 'l_set']
    
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            
            
            # Get predictions
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)
            
            # Collect true and predicted labels
            all_true_labels.extend(labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    
    # Convert to percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
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

