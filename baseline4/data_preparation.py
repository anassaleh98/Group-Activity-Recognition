import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils import set_seed


# seed 
set_seed()
def extract_target_image_and_label(video_id, dataset_path):
    """
    Extracts all videos (9 frames paths) and corresponding frame activity class from the annotations.txt file.
    """
    video_path = os.path.join(dataset_path, str(video_id))  # Video directory path
    annotation_file = os.path.join(video_path, 'annotations.txt')  # Path to annotation file
    
    # List to store (tuple(frame_paths, activity_class))
    videos_and_labels = []
    
    # Read annotations.txt
    with open(annotation_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Extract the frame ID (image name) and frame activity class
            frame_image = parts[0]  # Image name (e.g., '48075.jpg')
            frame_activity_class = parts[1]  # Frame Activity Class (e.g., 'r_winpoint')
            
            # Remove .jpg from frame_image to get the frame ID
            frame_id = os.path.splitext(frame_image)[0]
            
            # video : 4 frames before targrt and 4 frames after it
            video_frames_paths = []
            for fr_id in range(int(frame_id)-4, int(frame_id)+5):
                frame_path = os.path.join(video_path, str(frame_id), f"{fr_id}.jpg")
                video_frames_paths.append(frame_path)
            
            video_tuple = (video_frames_paths, frame_activity_class)
            
            videos_and_labels.append(video_tuple)
    
    return videos_and_labels


def prepare_dataset(video_ids, dataset_path):
    """
    Prepares the dataset by extracting the videos frames paths and their labels for the given video IDs.
    Returns a List of lists of tuples (list of frames pathes, label).
    """
    dataset = []
    
    for video_id in video_ids:
        # Extract video(frames paths) and label for this video
        videos_and_labels = extract_target_image_and_label(video_id, dataset_path)
        dataset.extend(videos_and_labels)  # Add them to the dataset list
    
    return dataset

# Step 2: Dataset Class
class VideoFrameDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data: List of lists of tuples (list of frames pathes, label)
            transform: Optional transformations to be applied on a sample
        """
        self.data = data
        self.transform = transform

        # Class mapping: 9 activity classes 
        self.class_mapping = {'r_set': 0, 'r_spike': 1, 'r-pass': 2, 'r_winpoint': 3,
                              'l_winpoint': 4, 'l-pass': 5, 'l-spike': 6, 'l_set':7} 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get frames paths  and labels
        video_frames_paths, label = self.data[idx]

        video_frames = []

        for frame_path in video_frames_paths:
            # Open the farme
            frame = Image.open(frame_path).convert('RGB')

            # Apply transformations
            if self.transform:
                frame = self.transform(frame)

            video_frames.append(frame)

        video_frames = torch.stack(video_frames)  # Shape: (9, 3, 224, 224)
        label_tensor = torch.tensor(self.class_mapping[label], dtype=torch.long)

        return video_frames, label_tensor
    
# Step 3: Data Augmentation and Transformations
train_transform = transforms.Compose([
    transforms.Resize(256),            # Resize shorter side to 256     
    transforms.RandomResizedCrop(224),  # Randomly crop and resize to 224x224
    transforms.RandomRotation(degrees=5),                   # Randomly rotate images within ±5 degrees
    transforms.ToTensor(),                                   # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # Normalize using ImageNet mean and std values
                         std=[0.229, 0.224, 0.225]),         # (mean and std are the same used during ResNet pre-training)
])

# Define the validation transform without augmentations
val_transform = transforms.Compose([
    transforms.Resize(256),            
    transforms.CenterCrop(224),       
    transforms.ToTensor(),                                   
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        
                         std=[0.229, 0.224, 0.225]),         
])  


# Step 4: Load the Data

# Video IDs for each dataset split
train_videos = [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54]
val_videos = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
test_videos = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]

# Dataset path 
dataset_path = r'E:\Machine Learining\Dr Mostafa Saad\ML\Slides\15 Deep Learning Project for CNN+LSTM\Data\videos'

# Prepare datasets for train, validation, and test splits
train_data = prepare_dataset(train_videos, dataset_path)
val_data = prepare_dataset(val_videos, dataset_path)
test_data = prepare_dataset(test_videos, dataset_path)

# Create Dataset objects for training, validation, and testing
train_dataset = VideoFrameDataset(train_data, transform=train_transform)
val_dataset = VideoFrameDataset(val_data, transform=val_transform)
test_dataset = VideoFrameDataset(test_data, transform=val_transform)

batch_size = 4

# Create DataLoader objects to load batches of data
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


      