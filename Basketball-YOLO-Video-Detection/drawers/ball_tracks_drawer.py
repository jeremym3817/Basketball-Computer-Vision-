from .utils import draw_triangle

class BallTracksDrawer:

    def __init__(self):
        self.ball_pointer_color = (0, 255, 0)

    # draws different overlays onto video
    def draw(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # gets dictionary of objects form each frame
            ball_dict = tracks[frame_num]

            # draw ball tracks
            for _, ball in ball_dict.items():
                # if bbox is not found on frame, continue with previous location
                bbox = ball['bbox']
                if bbox is None:
                    continue

                frame = draw_triangle(frame, bbox, self.ball_pointer_color)

            # adds overlay onto the frame
            output_video_frames.append(frame)

        return output_video_frames