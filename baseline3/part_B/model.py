import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_preparation import  trainloader
from utils import set_seed

# seed 
set_seed()

# device
device = torch.device("cuda")

# Step 5: Pretrained Feature Extractor
def get_feature_extractor(model_load_path):
    model = resnet50(weights=None)
    # Modify to match the trained model's structure (9 classes)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 9)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_load_path, map_location=device))

    # Remove the final fully connected layer for feature extraction
    feature_extractor = nn.Sequential(
        *list(model.children())[:-1],  # Excludes the FC layer
         torch.nn.Flatten()  # Flatten output to (batch_size, 2048))  
    )
    return feature_extractor.to(device).eval()


# Step 6:Define the Classifier

class FrameClassifier(nn.Module):
    def __init__(self, input_dim=2048, num_classes=8):
        super(FrameClassifier, self).__init__()
        
        # Add batch normalization layers
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, num_classes)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.fc(x)



# Step 7: Define Loss Function
def calculate_class_weights(dataloader):
    class_counts = torch.zeros(8)  # 8 classes 
    total_samples = 0
    
    # Count samples for each class
    for _, labels in dataloader:
        for label in labels:
            class_counts[label] += 1
            total_samples += 1
    
    # Calculate weights
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    
    return class_weights


# Calculate class weights
class_weights = calculate_class_weights(trainloader)
class_weights = class_weights.to(device)
loss_criterion = nn.CrossEntropyLoss(weight=class_weights)


# Step 8: Train the model
def train_model(model, feature_extractor, train_dataloader, val_dataloader, epochs=10, lr=1e-3):

    criterion = loss_criterion

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    model.to(device)

    for epoch in range(epochs):
        # Training loop
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        all_true_labels = []
        all_pred_labels = []

        for player_images, frame_labels in train_dataloader:
            player_images = player_images.to(device)  # Shape: [batch_size, num_players, channels, height, width]
            frame_labels = frame_labels.to(device)

            # Flatten the player images: [batch_size * num_players, channels, height, width]
            batch_size, num_players, channels, height, width = player_images.shape
            player_images = player_images.view(batch_size * num_players, channels, height, width)

            # Extract features for the whole batch
            with torch.no_grad():
                player_features = feature_extractor(player_images)  # Shape: [batch_size * num_players, 2048]

            # Reshape the player features back to [batch_size, num_players, 2048]
            player_features = player_features.view(batch_size, num_players, -1)

            # Apply max over the players in the batch to get frame-level features
            frame_features = torch.max(player_features, dim=1).values  # Shape: [batch_size, 2048]

            # Forward pass
            outputs = model(frame_features)
            loss = criterion(outputs, frame_labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == frame_labels).sum().item()
            total += frame_labels.size(0)

            # Collect true and predicted labels for F1 score calculation
            all_true_labels.extend(frame_labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())

        scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_accuracy = 100.0 * correct / total
        train_f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')  # F1 score for training
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, F1 Score: {train_f1:.4f}")

        # Validation 
        if (epoch+1) == epochs :
            print("Validation ", end="")
            validate_model(model, feature_extractor, val_dataloader)

        #if (epoch+1) == epochs :
        #   create_confusion_matrix(model, feature_extractor, val_dataloader)


# Step 9: Evaluate the model        
def validate_model(model, feature_extractor, val_dataloader):
    model.eval()
    correct, total = 0, 0
    epoch_loss = 0
    all_true_labels = []
    all_pred_labels = []
    
    criterion = loss_criterion
    
    with torch.no_grad():
        for player_images, frame_labels in val_dataloader:
            player_images = player_images.to(device)  # Shape: [batch_size, num_players, channels, height, width]
            frame_labels = frame_labels.to(device)

            # Flatten the player images: [batch_size * num_players, channels, height, width]
            batch_size, num_players, channels, height, width = player_images.shape
            player_images = player_images.view(batch_size * num_players, channels, height, width)

            # Extract features for the whole batch
            player_features = feature_extractor(player_images)  # Shape: [batch_size * num_players, 2048]

            # Reshape the player features back to [batch_size, num_players, 2048]
            player_features = player_features.view(batch_size, num_players, -1)

            # Apply max over the players in the batch to get frame-level features
            frame_features = torch.max(player_features, dim=1).values  # Shape: [batch_size, 2048]

            # Forward pass
            outputs = model(frame_features)
            loss = criterion(outputs, frame_labels)

            # Statistics
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == frame_labels).sum().item()
            total += frame_labels.size(0)

            # Collect true and predicted labels for F1 score calculation
            all_true_labels.extend(frame_labels.cpu().numpy())
            all_pred_labels.extend(predicted.cpu().numpy())

    avg_epoch_loss = epoch_loss / len(val_dataloader)
    validation_accuracy = 100.0 * correct / total
    val_f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')  
    print(f"Loss: {avg_epoch_loss:.4f}, Accuracy: {validation_accuracy:.2f}%, F1 Score: {val_f1:.4f}")


# step 10 :confusion_matrix
def create_confusion_matrix(model, feature_extractor, dataloader): 
    # Define class names
    class_names = ['r_set', 'r_spike', 'r-pass', 'r_winpoint',
                   'l_winpoint', 'l-pass', 'l-spike', 'l_set']
    
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for player_images, frame_labels in dataloader:
            player_images = player_images.to(device)
            frame_labels = frame_labels.to(device)
            
            # Flatten the player images
            batch_size, num_players, channels, height, width = player_images.shape
            player_images = player_images.view(batch_size * num_players, channels, height, width)
            
            # Extract features
            player_features = feature_extractor(player_images)
            player_features = player_features.view(batch_size, num_players, -1)
            frame_features = torch.max(player_features, dim=1).values
            
            # Get predictions
            outputs = model(frame_features)
            _, predicted = torch.max(outputs, 1)
            
            # Collect true and predicted labels
            all_true_labels.extend(frame_labels.cpu().numpy())
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