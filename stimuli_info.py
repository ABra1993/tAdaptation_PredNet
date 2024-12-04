import cv2
import datetime

datasets = ['WT_AMS', 'WT_VEN', 'WT_WL']

for dataset in datasets:

    print(30*'-')
    print(dataset)
    print(30*'-')

    # import file
    input_file = '/datasets/train/' + dataset + '.mp4'
    cap = cv2.VideoCapture(input_file)

    # count the number of frames 
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = cap.get(cv2.CAP_PROP_FPS) 
    
    # calculate duration of the video 
    seconds = round(frames / fps) 
    video_time = datetime.timedelta(seconds=seconds) 

    # print info
    print(f"number of frames: {int(frames)}") 
    print(f"fps: {fps}") 
    print(f"duration in seconds: {seconds}") 
    print(f"video time: {video_time}") 