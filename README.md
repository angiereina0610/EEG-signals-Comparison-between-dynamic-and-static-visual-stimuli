# EEG-signals-Comparison-between-dynamic-and-static-visual-stimuli
# EEG-signals-Comparison-between-dynamic-and-static-visual-stimuli
## Introduction

Emotional processing in the human brain can vary significantly depending on the type of stimulus, especially when comparing **static images** and **dynamic video content**. Understanding these differences is essential for advancing affective computing and for developing EEG-based emotion-decoding systems that remain robust across different perceptual contexts.

This project analyzes the **EmoEEG-MC: Multi-Context Emotional EEG Dataset**, which provides EEG recordings collected while participants viewed images and videos designed to evoke specific emotional states. Because the raw dataset contains several inconsistencies—such as missing trials, split EEG segments, and misaligned stimulus orders—this repository includes a complete pipeline to **reconstruct, clean, and reorganize the full dataset**.

The workflow integrates:

- Preprocessing of raw EEG recordings  
- Reconstruction of the correct experimental sequence for each participant  
- Differentiation between image-based and video-based trials  
- Extraction of time-domain and frequency-domain features (PSD & Differential Entropy)  
- Generation of **2D topographic maps** from 64 EEG channels  
- Training of **CNN models** for cross-context emotion classification  

This project not only restructures the dataset into a usable format (*allData.csv*, PSD/DE tensors, and topographic maps), but also makes it possible to study emotional differences between **static** and **dynamic** visual stimuli through deep-learning-based decoding.
<img width="1566" height="376" alt="image" src="https://github.com/user-attachments/assets/466f74d4-5cd0-432f-b57e-1523d184576e" />
## 2. Data Preprocessing and Reordering

There are **two main scripts** in this project:

### 2.1 Data Preparation Script
This script performs:

1. **Emotion mapping and labels**:  
   - Defines the emotions to be used.  
   - Loads the original labels for each participant.  

2. **Dataset reordering**:  
   - Iterates through each participant’s data.  
   - Applies filters to handle signals split into two parts.  
   - Determines which stimuli are **images** and which are **videos**.  
     > Each stimulus block contains 3 images or 3 videos, and the order is verified for each participant.  
3. **Exclusion of participants with errors**:  
   - Participant 22 is excluded due to dataset inconsistencies.  

4. **Generation of `allData.csv`**:  
   Includes:
   - `trial_numbers` and video number.  
   - Scores for 6 emotions (Arousal, Valence, Familiarity, Liking).  
   - Stimulus order label (first 21 images, then 21 videos).  
   - Participant ID and original labels.  
   - Final label assigned based on the highest emotion score.

---

## 3. Feature Extraction and Topographic Maps

### 3.1 Frequency Features
Two main types of features are extracted from the EEG signals:

- **Differential Entropy (DE)**: Computed across multiple frequency bands to capture signal complexity and information content.  
- **Power Spectral Density (PSD)**: Represents the distribution of power across different frequency bands.

### 3.2 Topographic Maps
EEG topographic maps are generated to visualize spatial patterns of brain activity:

1. **Channel Montage Setup**:  
   The positions of 64 EEG channels are mapped using a standard montage:

```python
montage = mne.channels.make_standard_montage('brainproducts-RNP-BA-128')
pos = montage.get_positions()
posCh = np.empty([64,3])
for idx, ch in enumerate(ch_names):
    posCh[idx] = pos["ch_pos"].get(ch, [np.nan, np.nan, np.nan])
```
Here we can observe an example of two topographical representations for the emotion disgust, one with "DE" and the other with "PSD".
<img width="1449" height="378" alt="image" src="https://github.com/user-attachments/assets/dd8b6743-d297-4f49-8c7c-60d5a30531cd" /> 
## 4. CNN Model and Results

### 4.1 CNN Model
The CNN model is trained using the 2D topographic maps generated from EEG signals. Key details:

- **Inputs**: Topographic maps for each emotion, separated by stimulus type (images vs videos).  
- **Features**: Concatenation of frequency-domain features:
  - Differential Entropy (DE) across bands.  
  - Power Spectral Density (PSD) across bands.  
- **Temporal Windowing**: Maps have a shape `(batch_size, 30, 16, 16, 10)`, where 30 represents the temporal window.  
- **Architecture**: Standard 2D CNN layers are applied to capture spatial patterns, followed by fully connected layers for emotion classification.  

### 4.2 Training Procedure
- Data is split by participant and stimulus type to avoid leakage.  
- Normalization and interpolation ensure consistent input shapes.  
- Training optimizes cross-entropy loss for **multiclass emotion prediction**.

### 4.3 Results and Analysis
Two types of analyses are performed:

1. **Multiclass Emotions**:  
   - Each emotion (Fear, Inspiration, Joy, Sadness, Tenderness) is compared separately between images and videos using Wilcoxon signed-rank tests.  
   - The results are stored in the folder **`resul`**, which contains:
     - Accuracy for images and videos.  
     - Wilcoxon statistics and p-values.  
     - Significant differences and the best stimulus type per emotion.

2. **Binary Emotions (Positive vs Negative)**:  
   - Positive and Negative categories are also compared across stimulus types.  
   - The results are stored in the folder **`binary_resuld`**, containing the same information:
     - Accuracy for images and videos.  
     - Wilcoxon statistics, p-values, significance, and best stimulus type.
