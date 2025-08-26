from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd

# goes back in directory to access utils file
import sys
sys.path.append('../')
from utils import read_stub, save_stub

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    # detects frames via the model attribute 
    def detect_frames(self, frames):
        # number of frames read at once
        batch_size = 20
        detections = []
        # runs model on batched frames
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            batch_detections = self.model.predict(batch_frames, conf=0.5)
            detections += batch_detections
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        # read previously cached stubs
        tracks = read_stub(stub_path, read_from_stub)
        # if stub has same number of frames, use previous stub (avoids unneeded looping)
        if tracks is not None and len(tracks) == len(frames):
            return tracks
        
        # gets detection predictions from model
        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            # gets model predicted names 
            cls_names = detection.names
            cls_names_inv = {v:k for k, v in cls_names.items()}

            # convert ultralytics to supervisiion format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # key = player-id, value = player bounding_box
            tracks.append({})

            # define ball likelihood
            chosen_bbox = None
            max_confidence = 0

            for frame_detection in detection_supervision:
                # gets bounding_box, class_id, and current confidence of the ball's position
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]

                # if object class is identified as the ball with higher confidence, that is assigned the new ball location
                if cls_id == cls_names_inv['Ball'] and max_confidence < confidence:
                    chosen_bbox = bbox
                    max_confidence = confidence
            
            # if there is a ball, it's position is added to dictionary log
            if chosen_bbox is not None:
                tracks[frame_num][1] = {'bbox' : chosen_bbox}

        # saves check point for next loop
        save_stub(stub_path, tracks)

        return tracks
    
    def remove_wrong_detections(self, ball_positions):
        # define values of last ball location
        max_allowed_distance = 25
        last_good_frame_index = -1

        for i in range(len(ball_positions)):
            # gets current bbox (or no bbox position depending on the value)
            curr_bbox = ball_positions[i].get(1, {}).get('bbox', [])

            # if there's no current bbox, skip cycle
            if len(curr_bbox) == 0:
                continue

            # assign the initial bbox value
            if last_good_frame_index == -1:
                last_good_frame_index = i
                continue

            # gets last good bbox and how many frames that was from the last good bbox
            last_good_box = ball_positions[last_good_frame_index].get(1, {}).get('bbox', [])
            frame_gap = i - last_good_frame_index
            adjusted_max_distance = max_allowed_distance * frame_gap

            # if last good bbox - curr bbox > adjusted_max_dis, clear previous ball positions to initialize a new value
            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(curr_bbox[:2]) > adjusted_max_distance):
                ball_positions[i] = {}
            # otherwise curr index is valid, assign as last for next iteration
            else:
                last_good_frame_index = i

            
        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        # convert list to df
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        # interpolate frames misisng a ball detection
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        # reformat back to python list
        ball_positions = [{1:{'bbox' : x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
