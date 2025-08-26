from ultralytics import YOLO
import supervision as sv

# goes back in directory to access utils file
import sys
sys.path.append('../')
from utils import read_stub, save_stub

# used to define a player object in video
class PlayerTracker:
    # initialize model and tracker to be used 
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

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

            # tracks objects over set of frames
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            # key = player-id, value = player bounding_box
            tracks.append({})

            for frame_detection in detection_with_tracks:
                # gets bounding_box, class_id, tracker_id
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = int(frame_detection[4])

                # if object class is identified as a player, add to tracks dictionary
                if cls_id == cls_names_inv['Player']:
                    tracks[frame_num][track_id] = {'bbox' : bbox}

        # saves check point for next loop
        save_stub(stub_path, tracks)

        return tracks
