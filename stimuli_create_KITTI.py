import hickle as hkl
import numpy as np
import cv2
import h5py
import os
import matplotlib.pyplot as plt

# Define the path to your HDF5 file and the output MP4 file
hdf5_file_path = '/home/amber/OneDrive/code/prednet_Lotter2017/kitti_data/X_train.hkl'
output_mp4_file_path = '/home/amber/OneDrive/datasets/KITTI.avi'  # Using .avi extension with XVID codec

# Verify that the output path is not empty and is writable
if not output_mp4_file_path:
    raise ValueError("The output file path is empty.")
if not os.path.isdir(os.path.dirname(output_mp4_file_path)) and os.path.dirname(output_mp4_file_path) != '':
    raise ValueError("The directory for the output file does not exist or is not writable.")

# Open the HDF5 file and read the frames
with h5py.File(hdf5_file_path, 'r') as hf:
    # List all datasets in the file for debugging
    print("Datasets in the file:", list(hf.keys()))
    
    # Read the frames from the dataset
    frames = hf['data_0'][:]  # Adjust according to your .hkl file structure

# Get the shape of the frames
num_frames, height, width, channels = frames.shape
print(f"Number of frames: {num_frames}, Height: {height}, Width: {width}, Channels: {channels}")

# Check if the dimensions are valid
if width <= 0 or height <= 0:
    raise ValueError("Invalid frame dimensions.")
if channels != 3:
    raise ValueError("Frames must have 3 channels (RGB).")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Fallback codec for .avi files; MJPG
# fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
fps = 20  # Adjust frames per second as needed
video_writer = cv2.VideoWriter(output_mp4_file_path, fourcc, fps, (width, height))

# Check if the video writer is opened successfully
if not video_writer.isOpened():
    raise Exception("Error: Could not open the video writer.")

print("Video writer opened successfully.")

fig, axs = plt.subplots(1, 2)

# Write each frame to the video file
for i in range(num_frames):
    frame = frames[i]

    # if i == 0:
    #     axs[0].imshow(frame)
    
    # Debug: Print the shape of the current frame
    # print(f"Frame {i} shape: {frame.shape}")
    
    # Ensure the frame is in the correct format (uint8)
    # frame = (frame * 255).astype(np.uint8)
    
    # # Convert frame to BGR if it is not already (assuming frames are in RGB format)
    # if channels == 3:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # if i == 0:
    #     axs[1].imshow(frame)
    #     plt.show()
    
    # Write the frame to the video file and add an assertion to check the frame size
    assert frame.shape[1] == width and frame.shape[0] == height and frame.shape[2] == channels, \
        f"Frame {i} dimensions are incorrect: {frame.shape}"
    
    video_writer.write(frame)
    if i % 100 == 0:
        print(f"Processed frame {i + 1} / {num_frames}")

# Release the VideoWriter object
video_writer.release()

print("Video created successfully.")