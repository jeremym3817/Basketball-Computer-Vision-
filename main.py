from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import (
    PlayerTracksDrawer,
    BallTracksDrawer,
    TeamBallControlDrawer
)
from team_assigner import TeamAssigner
from ball_acquisition import BallAcquisitionDetector

# pip install ultralytics supervision matplotlib opencv-python pickle5 transformers

def main(input_video, output_path):
    # Read Video
    video_frames = read_video(input_video)

    # Initialize Tracker
    player_tracker = PlayerTracker('models/player_detector.pt')
    ball_tracker = BallTracker('models/ball_detector.pt')

    # Run Trackers, Saves stubs for future runs
    player_tracks = player_tracker.get_object_tracks(video_frames, 
                                                     read_from_stub=True, 
                                                     stub_path='stubs/player_track_stubs.pkl')
    
    ball_tracks = ball_tracker.get_object_tracks(video_frames, 
                                                 read_from_stub=True, 
                                                 stub_path='stubs/ball_track_stubs.pkl')

    # Remove Incorrect Ball Detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)

    # Interpolate Ball Tracks
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    player_assignment = team_assigner.get_player_teams_across_frames(video_frames, player_tracks, 
                                                                read_from_stub=True, stub_path='stubs/player_assignment_stub.pkl')

    # Ball Acquisition
    ball_acquisition_detector = BallAcquisitionDetector()
    ball_acquisition = ball_acquisition_detector.detect_ball_possession(player_tracks, ball_tracks)

    # Draw Output
    # Initialize Drawers
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    team_ball_control_drawer = TeamBallControlDrawer()

    # Draw Object Tracks
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracks, player_assignment, ball_acquisition)
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    # Draw Stat Overlay
    output_video_frames = team_ball_control_drawer.draw(output_video_frames, player_assignment, ball_acquisition)

    # Save Video
    save_video(output_video_frames, output_path)

if __name__ == '__main__':
    #main('Input-Videos/video_1.mp4', 'output_videos/output_video1_v2.avi')
    main('Input-Videos/celtics_test.mp4', 'output_videos/output_celtics_test.avi')















#model = YOLO('models\player_detector.pt')

#results = model.predict('Input-Videos/video_1.mp4', save=True)
#results = model.track('Input-Videos/video_1.mp4', save=True)

#print(results)
#print('*************')
#for box in results[0].boxes:
#    print(box)