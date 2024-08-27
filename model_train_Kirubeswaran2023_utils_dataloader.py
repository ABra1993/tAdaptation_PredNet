import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

class VideoDataset(Dataset):
    def __init__(self, static, video_path, sequence_length, w, h, c):

        print('Static images: ', static)
        print(video_path)

        self.cap = cv2.VideoCapture(video_path)
        self.sequence_length = sequence_length
        self.w = w
        self.h = h
        self.c = c
        self.static = static

        # Load all frames
        self.frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.w, self.h))  # Resize frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            frame = frame / 255.0  # Normalize
            frame = torch.Tensor(frame).permute(2, 0, 1)  # Reshape to (C, H, W)
            self.frames.append(frame)
        self.cap.release()

        self.total_frames = len(self.frames)
        print('Total number of frames:', self.total_frames)

    def __len__(self):
        
        # large number here since we are using random sampling
        return int(1e6)

    def __getitem__(self, idx):
            
        if self.static:

            # randomly choose a frame
            frame_idx = random.randint(0, self.total_frames - 1)
            frame = self.frames[frame_idx]
            
            # repeat the frame to form a sequence
            sequence = frame.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)
            return sequence
        
        else:

            # randomly choose the starting position for the sequence
            start_idx = random.randint(0, self.total_frames - self.sequence_length)
            sequence = torch.stack(self.frames[start_idx:start_idx + self.sequence_length])
            return sequence

