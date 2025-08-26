try:
    import cv2
except:
    import pip
    pip.main(['install', 'opencv-python'])
    import cv2

import os

# capture each frame's data in video
def read_video(video_path):
    # captures video content
    cap=cv2.VideoCapture(video_path)
    frames=[]
    # appends all frames to frame list
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

# saves video to directory
def save_video(output_video_frames, output_video_path):
    # create video directory
    video_dirname = os.path.dirname(output_video_path)
    if not os.path.exists(video_dirname):
        os.mkdir(video_dirname)

    # specifiy video encode/decode format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # initializes output format
    out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    # writes overlay to every video frame
    for frame in output_video_frames:
        out.write(frame)
    out.release()