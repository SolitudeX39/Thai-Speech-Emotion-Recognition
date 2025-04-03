import librosa
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# ตั้งค่าพารามิเตอร์
target_sr = 22050
frame_length = 2048
hop_length = 512
n_mels = 128
max_files_per_class = 500

def create_feature_dataframe(base_dir):
    max_frames = 0
    feature_data = []
    n_features = None
    
    # ขั้นตอนที่ 1: คำนวณ max_frames
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            files = sorted([f for f in os.listdir(class_dir) if f.endswith('.flac')])[:max_files_per_class]
            for file in files:
                y, _ = librosa.load(os.path.join(class_dir, file), sr=target_sr)
                n_frames = (len(y) - frame_length) // hop_length + 1
                max_frames = max(max_frames, n_frames)

    # ขั้นตอนที่ 2: สกัดและบันทึกฟีเจอร์
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            files = sorted([f for f in os.listdir(class_dir) if f.endswith('.flac')])[:max_files_per_class]
            for file in files:
                file_path = os.path.join(class_dir, file)
                y, sr = librosa.load(file_path, sr=target_sr)
                
                # คำนวณฟีเจอร์
                zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_length, 
                                                   hop_length=hop_length, n_mels=n_mels)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                
                # รวมฟีเจอร์และปรับขนาด
                combined = np.vstack((zcr, mel_db))
                
                # ทำ padding
                if combined.shape[1] < max_frames:
                    pad_width = max_frames - combined.shape[1]
                    combined = np.pad(combined, ((0,0), (0,pad_width)), mode='constant')
                
                # แปลงเป็น list 2D และ transpose
                feature_2d = combined.T.astype(np.float32).tolist()
                if n_features is None:
                    n_features = combined.shape[0]
                    
                feature_data.append({
                    'feature': feature_2d,
                    'label': class_name
                })
    
    return pd.DataFrame(feature_data), (max_frames, n_features)

# สร้าง DataFrame
df, (max_frames, n_features) = create_feature_dataframe('Balanced_audio')

# บันทึกข้อมูล
df.to_parquet('audio_dataset_2d.parquet', index=False)

# บันทึก metadata
metadata = {
    'max_frames': max_frames,
    'n_features': n_features,
    'classes': df['label'].unique().tolist(),
    'max_files_per_class': max_files_per_class
}
np.save('metadata.npy', metadata)

# run time approx 12 mins