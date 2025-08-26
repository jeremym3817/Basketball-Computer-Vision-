import numpy as np
import cv2

class TeamBallControlDrawer:
    def __init__(self):
        pass

    def draw(self, video_frames, player_assignment, ball_acquisition):
        # get team control for each frame
        team_ball_control = self.get_team_ball_control(player_assignment, ball_acquisition)

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            # ignore first frame
            if frame_num == 0:
                continue
            
            # draw on frame
            frame_drawn = self.draw_frame(frame, frame_num, team_ball_control)
            output_video_frames.append(frame_drawn)
        
        return output_video_frames
    
    def draw_frame(self, frame, frame_num, team_ball_control):
        # copy frame and define consts
        overlay = frame.copy()
        font_scale = 0.7
        font_thickness = 2

        # get overlay position
        frame_height, frame_width = overlay.shape[:2]
        rect_x1 = int(frame_width * .60)
        rect_y1 = int(frame_height * .75)
        rect_x2 = int(frame_width * .99)
        rect_y2 = int(frame_height * .90)

        # get text position
        text_x = int(frame_width * 0.63)
        text_y1 = int(frame_height * 0.80)
        text_y2 = int(frame_height * 0.88)

        # draw rectangle overlay and make transparent
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # increment to next frame and filter by which team has the most time of possession
        team_ball_control_till_frame = team_ball_control[: frame_num + 1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]

        # get percentages
        team_1_pct = team_1_num_frames / (team_ball_control_till_frame.shape[0])
        team_2_pct = team_2_num_frames / (team_ball_control_till_frame.shape[0])

        # draw text overlay
        cv2.putText(frame, f'Team 1 Ball Control: {team_1_pct * 100: .2f}%', (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(frame, f'Team 2 Ball Control: {team_2_pct * 100: .2f}%', (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        return frame

    def get_team_ball_control(self, player_assignment, ball_acquisition):
        team_ball_control = []
    
        for player_assignment_frame, ball_acquisition_frame in zip(player_assignment, ball_acquisition):
            # if neither team has ball or a player with the ball is not in frame, -1
            if ball_acquisition_frame == -1 or ball_acquisition_frame not in player_assignment_frame:
                team_ball_control.append(-1)
                continue

            # assign team's possess on a given frame 
            if player_assignment_frame[ball_acquisition_frame] == 1:
                team_ball_control.append(1)
            else:
                team_ball_control.append(2)

        # convert to numpy
        team_ball_control = np.array(team_ball_control)
        return team_ball_control

