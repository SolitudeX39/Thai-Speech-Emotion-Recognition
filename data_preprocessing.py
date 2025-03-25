import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

target_sr = 22050
frame_length = 2048
hop_length = 512
n_mels = 128
max_files_per_class = 150  # amount of files per class

def create_feature_dataframe(base_dir):
    max_frames = 0
    feature_data = []
    
    # find max_freq
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            files = sorted([f for f in os.listdir(class_dir) if f.endswith('.flac')])[:max_files_per_class]
            for file in files:
                y, _ = librosa.load(os.path.join(class_dir, file), sr=target_sr)
                n_frames = (len(y) - frame_length) // hop_length + 1
                max_frames = max(max_frames, n_frames)

    # feature extraction
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            files = sorted([f for f in os.listdir(class_dir) if f.endswith('.flac')])[:max_files_per_class]
            for file in files:
                file_path = os.path.join(class_dir, file)
                y, sr = librosa.load(file_path, sr=target_sr)
                
                
                zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_length, 
                                                    hop_length=hop_length, n_mels=n_mels)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # combine feature
                combined = np.vstack((zcr, mel_db))
                
                # apply padding
                if combined.shape[1] < max_frames:
                    pad_width = max_frames - combined.shape[1]
                    combined = np.pad(combined, ((0,0), (0,pad_width)), mode='constant')
                
                
                feature_data.append({
                    'feature': combined.T.flatten().tobytes(),
                    'label': class_name
                })
    
    return pd.DataFrame(feature_data), combined.shape


df, feature_shape = create_feature_dataframe('Balanced_audio')


df.to_csv('audio_dataset_class.csv', index=False)
np.save('metadata.npy', {
    'max_frames': feature_shape[1],
    'n_features': feature_shape[0],
    'classes': df['label'].unique().tolist(),
    'max_files_per_class': max_files_per_class
})

print(f'Dataset size: {len(df)} samples')
print(f'Estimated size: {df.memory_usage(deep=True).sum()/1024**3:.2f} GB')