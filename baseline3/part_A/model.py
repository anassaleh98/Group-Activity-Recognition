import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data_preparation import train_data, train_dataset, trainloader, valloader
from utils import set_seed

# seed 
set_seed()

# Step 5: Load Pretrained ResNet50 Model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_player_classes = 9
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 9)

# Freeze Layers (layer1,2)
for layer in [model.layer1, model.layer2]:
    for param in layer.parameters():
        param.requires_grad = False


device = torch.device("cuda")
model.to(device)


# Step 6: Define Loss Function and Optimizer
label_counts = Counter([label for frame in train_data for player in frame['players'] for label in [player['action_class']]])
total_samples = sum(label_counts.values())
class_weights = [total_samples / label_counts[i] if i in label_counts else 1e-3 for i in range(num_player_classes)]
weights_player = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion_player = nn.CrossEntropyLoss(weight=weights_player)

def custom_loss_fn(output, labels):
    mask = labels != -1
    output = output[mask]
    labels = labels[mask]
    if len(labels) == 0:
        return torch.tensor(0.0, requires_grad=True, device=output.device)
    return criterion_player(output, labels)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)


# Step 7: Train the model
num_epochs = 4

def train():
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        total_correct_player = 0
        total_samples_player = 0
        train_preds = []
        train_labels = []
        
        for data in trainloader:
            player_images_batch, player_labels_batch = data
            all_player_images = [img for frame_imgs in player_images_batch for img in frame_imgs]
            all_player_labels = [lbl for frame_lbls in player_labels_batch for lbl in frame_lbls]
            player_images = torch.stack(all_player_images).to(device)
            player_labels = torch.stack(all_player_labels).to(device)
            
            optimizer.zero_grad()
            output = model(player_images)
            loss_player = custom_loss_fn(output, player_labels)
            loss_player.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss_player.item()
            _, predicted_player = output.max(1)
            
            # Only consider non-ignored labels (-1)
            mask = player_labels != -1
            total_samples_player += mask.sum().item()
            total_correct_player += ((predicted_player == player_labels) & mask).sum().item()
            
            # Store predictions and labels for F1 score
            train_preds.extend(predicted_player[mask].cpu().numpy())
            train_labels.extend(player_labels[mask].cpu().numpy())
    
        
        scheduler.step()

        epoch_accuracy_player = 100. * total_correct_player / total_samples_player
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Training - Loss: {running_loss/len(trainloader):.4f}, Accuracy: {epoch_accuracy_player:.2f}%, F1: {train_f1:.4f}")
        
        # Validation 
        evaluate(valloader,"Validation")
        

# Step 8: Evaluate the model
def evaluate(dataloader, desc):
    model.eval()
    eval_loss = 0.0
    correct_player = 0
    samples_player = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in dataloader:
            player_images_batch, player_labels_batch = data
            all_player_images = [img for frame_imgs in player_images_batch for img in frame_imgs]
            all_player_labels = [lbl for frame_lbls in player_labels_batch for lbl in frame_lbls]
            player_images = torch.stack(all_player_images).to(device)
            player_labels = torch.stack(all_player_labels).to(device)
            
            output = model(player_images)
            loss_player = custom_loss_fn(output, player_labels)
            eval_loss += loss_player.item()
            
            _, predicted_player = output.max(1)
            
            # Only consider non-ignored labels (-1)
            mask = player_labels != -1
            samples_player += mask.sum().item()
            correct_player += ((predicted_player == player_labels) & mask).sum().item()
            
            # Store predictions and labels for F1 score
            all_preds.extend(predicted_player[mask].cpu().numpy())
            all_labels.extend(player_labels[mask].cpu().numpy())
    
    accuracy_player = 100. * correct_player / samples_player
    avg_loss = eval_loss / len(dataloader)
    eval_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"{desc} Loss: {avg_loss:.4f}, Accuracy: {accuracy_player:.2f}%, F1 Score: {eval_f1:.4f}")  
