from .utils import draw_ellipse, draw_triangle

class PlayerTracksDrawer:

    def __init__(self, team_1_color=[0, 0, 255], team_2_color=[255, 0, 0]):
        self.default_player_id = 1
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    # draws different overlays onto video
    def draw(self, video_frames, tracks, player_assignment, ball_acquisition):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # gets dictionary of objects form each frame
            player_dict = tracks[frame_num]

            # get assignment for curr frame
            player_assignment_for_frame = player_assignment[frame_num]

            # get player id with ball
            player_id_with_ball = ball_acquisition[frame_num]

            # draw players tracks
            for track_id, player in player_dict.items():
                # get the team assigned to each player in frame and give them a color based on that
                team_id = player_assignment_for_frame.get(track_id, self.default_player_id)

                if team_id == 1:
                    color = self.team_1_color
                else:
                    color = self.team_2_color

                frame = draw_ellipse(frame, player['bbox'], color, track_id)

                # if player has the ball, identify them with possession
                if track_id == player_id_with_ball:
                    frame = draw_ellipse(frame, player['bbox'], (0, 0, 0), track_id)
                    #frame = draw_triangle(frame, player['bbox'], (0, 0, 255))

            # adds overlay onto the frame
            output_video_frames.append(frame)

        return output_video_frames