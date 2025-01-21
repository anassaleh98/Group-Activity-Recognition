## [A Hierarchical Deep Temporal Model for Group Activity Recognition. Mostafa S. Ibrahim, Srikanth Muralidharan, Zhiwei Deng, Arash Vahdat, Greg Mori. IEEE Computer Vision and Pattern Recognition 2016](http://www.cs.sfu.ca/~mori/research/papers/ibrahim-cvpr16.pdf)

## Contents
0. [Key Updates](#key-updates)
0. [Dataset](#dataset)
0. [Ablation Study](#ablation-study)

## Key Updates
- Introduced a ....
- Improved accuracy for group activity recognition.

## Dataset

This dataset was collected using publicly available YouTube volleyball videos and annotated with 4,830 frames handpicked from 55 videos, with 9 player action labels and 8 team activity labels.

### Example
<img src="https://github.com/mostafa-saad/deep-activity-rec/blob/master/img/dataset1.jpg" alt="Figure 3" height="400" >

**Figure**: A frame labeled as Left Spike with bounding boxes around each team player annotated in the dataset.

<img src="https://github.com/mostafa-saad/deep-activity-rec/blob/master/img/dataset2.jpg" alt="Figure 4" height="400" >

**Figure**: For each visible player, an action label is annotated.

The list of action and activity labels and related statistics are tabulated in the following tables:

| Group Activity Class | No. of Instances |
|----------------------|------------------|
| Right set            | 644              |
| Right spike          | 623              |
| Right pass           | 801              |
| Right winpoint       | 295              |
| Left winpoint        | 367              |
| Left pass            | 826              |
| Left spike           | 642              |
| Left set             | 633              |

| Action Classes | No. of Instances |
|----------------|------------------|
| Waiting        | 3601             |
| Setting        | 1332             |
| Digging        | 2333             |
| Falling        | 1241             |
| Spiking        | 1216             |
| Blocking       | 2458             |
| Jumping        | 341              |
| Moving         | 5121             |
| Standing       | 38696            |

**Further information**:
- The dataset contains 55 videos. Each video has a folder with a unique ID (0, 1...54).
  - **Train Videos**: 1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54
  - **Validation Videos**: 0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51
  - **Test Videos**: 4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47
- Inside each video directory, annotated frames are organized as subdirectories (e.g., `volleyball/39/29885`).
  - For example, video 39, frame ID 29885.
- Each frame directory contains 41 images (20 images before the target frame, **target frame**, 20 frames after the target frame).
  - Example for frame ID: 29885 → Window = {29865, 29866.....29885, 29886....29905}.
- Each video directory includes an `annotations.txt` file with frame annotations.
- An annotation line format: `{Frame ID} {Frame Activity Class} {Player Annotation}  {Player Annotation} ...`
  - Player Annotation: a tight bounding box around each player.
- Player Annotation format: `{Action Class} X Y W H`
- Videos with a resolution of 1920x1080 are: 2, 37, 38, 39, 40, 41, 44, 45 (8 total). All others are 1280x720.

**Downloading the dataset**:  
The dataset can be accessed from the following sources:

- **Kaggle**: [Download from Kaggle](https://www.kaggle.com/datasets/ahmedmohamed365/volleyball)  
- **GitHub**: [Download from GitHub](https://github.com/mostafa-saad/deep-activity-rec#dataset)

## Ablation Study
Details about ablation studies.
