import librosa
import numpy as np
import pandas as pd
import os

# ตั้งค่าพารามิเตอร์
target_sr = 22050
frame_length = 2048
hop_length = 512
n_mels = 128
max_files_per_class = 678

def create_feature_dataframe(base_dir):
    max_frames = 0
    feature_data = []
    
    # รวบรวมไฟล์ทั้งหมด
    file_list = []
    for class_name in os.listdir(base_dir):
        class_dir = os.path.join(base_dir, class_name)
        if os.path.isdir(class_dir):
            files = sorted([f for f in os.listdir(class_dir) if f.endswith('.flac')])[:max_files_per_class]
            for file in files:
                file_list.append((class_name, os.path.join(class_dir, file)))
    
    # คำนวณ max_frames
    for class_name, file_path in file_list:
        y, _ = librosa.load(file_path, sr=target_sr)
        n_frames = (len(y) - frame_length) // hop_length + 1
        max_frames = max(max_frames, n_frames)
    
    # สกัด Features และบันทึก
    for class_name, file_path in file_list:
        y, sr = librosa.load(file_path, sr=target_sr)
        
        # คำนวณ Features
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
        
        # รวม Features และปรับขนาด (stack)
        combined = np.vstack((zcr, mel_db, mfcc, spectral_centroid, spectral_rolloff, rms))
        combined = librosa.util.fix_length(combined, size=max_frames, axis=1)
        
        # บันทึกเป็น Bytes และ Shape
        feature_bytes = combined.T.astype(np.float32).tobytes()
        feature_shape = combined.T.shape
        
        feature_data.append({
            'feature_bytes': feature_bytes,
            'feature_shape': feature_shape,
            'label': class_name
        })
    
    return pd.DataFrame(feature_data), max_frames, combined.shape[0]

# สร้าง DataFrame
df, max_frames, n_features = create_feature_dataframe('Balanced_audio')

# บันทึกข้อมูล
df.to_parquet('audio_dataset_bytes_6f.parquet', index=False)

# บันทึก Metadata
metadata = {
    'max_frames': max_frames,
    'n_features': n_features,
    'classes': df['label'].unique().tolist()
}
np.save('metadata_6f.npy', metadata)
