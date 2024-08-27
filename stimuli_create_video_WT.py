import cv2
import numpy as np

def resize_video(input_file, output_file, target_width, target_height, target_fps):
    
    #  open the input video file
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error: Failed to open input video file.")
        return

    # get the original frame rate of the input video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_fps = np.ceil(original_fps)
    if original_fps <= 0:
        print("Error: Failed to get the FPS of the input video.")
        cap.release()
        return

    # create VideoWriter object to save the resized video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_file, fourcc, target_fps, (target_width, target_height))

    # Check if VideoWriter object is opened successfully
    if not out.isOpened():
        print("Error: Failed to open output video file.")
        return

    # calculate frame skipping interval
    frame_skip_interval = int(original_fps/target_fps)
    if frame_skip_interval < 1:
        frame_skip_interval = 1
    print('Original: ', original_fps, ' fps')
    print('Target: ', target_fps, ' fps')
    print(frame_skip_interval, ' interval')

    # Read frames from the input video and resize them
    count = 0 # video size (~)
    while count < 110000:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip_interval == 0:
            
            # resize the frame to the target width and height
            resized_frame = cv2.resize(frame, (target_width, target_height))
            
            # write the resized frame to the output video file
            out.write(resized_frame)

        # increment count
        count+=1

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()
    print("Resized video saved successfully.")

# define input and output file paths
# datasets = ['WT_AMS', 'WT_VEN', 'WT_WL']
datasets = ['WT_WL']

fpss = [3, 6, 12]

# create videos
for fps in fpss:
    for dataset in datasets:

        print(dataset)

        input_file = '/home/amber/OneDrive/datasets/' + dataset + '.mp4'
        output_file = '/home/amber/OneDrive/datasets/train/' + dataset + '_fps' + str(fps) + '.mp4'

        # Define target width and height for resizing
        target_width = 160
        target_height = 128

        # Resize the video and save the resized version
        resize_video(input_file, output_file, target_width, target_height, fps)