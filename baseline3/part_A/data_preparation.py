import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from utils import set_seed

#seed 
set_seed()

# Step 1: Data Extraction Functions
def crop_image(image_path, X, Y, W, H):
    """Crop a player region from the target frame."""
    image = Image.open(image_path)
    width, height = image.size

    x1 = max(0, X - W // 2)
    y1 = max(0, Y - H // 2)
    x2 = min(width, X + W // 2)
    y2 = min(height, Y + H // 2)

    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image

def extract_target_image_and_label(video_id, dataset_path):
    """Extract target frames and player labels for a specific video."""
    video_path = os.path.join(dataset_path, str(video_id))
    annotation_file = os.path.join(video_path, 'annotations.txt')

    frame_annotations = []

    if not os.path.exists(annotation_file):
        return frame_annotations

    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            frame_image = parts[0]
            frame_id = os.path.splitext(frame_image)[0]
            target_image_path = os.path.join(video_path, str(frame_id), frame_image)

            players = []
            for i in range(2, len(parts), 5):
                if i + 4 < len(parts):
                    X, Y, W, H = map(int, parts[i:i+4])
                    action_class = parts[i + 4]
                    players.append({
                        'action_class': action_class,
                        'X': X,
                        'Y': Y,
                        'W': W,
                        'H': H
                    })

            frame_annotations.append({
                'frame_id': frame_id,
                'frame_image_path': target_image_path,
                'players': players
            })

    return frame_annotations

def prepare_dataset(video_ids, dataset_path, shuffle=True):
    dataset = []
    for video_id in video_ids:
        video_annotations = extract_target_image_and_label(video_id, dataset_path)
        dataset.extend(video_annotations)

    if shuffle:
        random.shuffle(dataset)
    
    return dataset

# Step 2: Dataset Class
class VideoFrameDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.player_class_mapping = {
            'waiting': 0, 'setting': 1, 'digging': 2, 'falling': 3,
            'spiking': 4, 'blocking': 5, 'jumping': 6, 'moving': 7, 'standing': 8
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_data = self.data[idx]
        frame_image_path = frame_data['frame_image_path']
        players = frame_data['players']

        player_images = []
        player_labels = []

        for player in players:
            cropped_image = crop_image(
                frame_image_path, player['X'], player['Y'], player['W'], player['H']
            )

            if self.transform:
                cropped_image = self.transform(cropped_image)

            label = torch.tensor(self.player_class_mapping[player['action_class']], dtype=torch.long)

            player_images.append(cropped_image)
            player_labels.append(label)

        while len(player_images) < 12:
            dummy_image = torch.zeros((3, 224, 224))
            dummy_label = torch.tensor(-1, dtype=torch.long)
            player_images.append(dummy_image)
            player_labels.append(dummy_label)

        player_images = player_images[:12]
        player_labels = player_labels[:12]

        return player_images, player_labels


def collate_fn(batch):
    player_images_batch = []
    player_labels_batch = []

    for player_images, player_labels in batch:
        player_images_batch.append(player_images)
        player_labels_batch.append(player_labels)

    return player_images_batch, player_labels_batch

# Dataset path
dataset_path = '/kaggle/input/volleyball/volleyball_/videos'

train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

train_data = prepare_dataset(train_videos, dataset_path, shuffle=True)
val_data = prepare_dataset(val_videos, dataset_path, shuffle=False)
test_data = prepare_dataset(test_videos, dataset_path, shuffle=False)


# Step 3: Data Augmentation and Transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Step 4: Load the Data

# Create Dataset objects for training, validation, and testing
train_dataset = VideoFrameDataset(train_data, transform=train_transform)
val_dataset = VideoFrameDataset(val_data, transform=val_transform)
test_dataset = VideoFrameDataset(test_data, transform=val_transform)

batch_size = 4
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
