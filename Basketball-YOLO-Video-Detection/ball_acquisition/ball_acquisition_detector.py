import sys
sys.path.append('../')
from utils import measure_distance, get_center_of_bbox

class BallAcquisitionDetector:
    def __init__(self):
        # max distance ball can be from player to be considered acquired
        self.possession_threshold = 50
        # min frames ball needs to overlap with player to be considered acquired
        self.min_frames = 11
        # percentage of bbox overlap between ball and player
        self.containment_threshold = 0.8

    def get_key_basketball_player_assignment_points(self, player_bbox, ball_center):
        # obtain player and ball points to measure distance
        ball_center_x = ball_center[0]
        ball_center_y = ball_center[1]

        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1

        output_points = []

        # if ball is within the vertical space of a player, append player bbox points to ball
        if ball_center_y > y1 and ball_center_y < y2:
            output_points.append((x1, ball_center_y))
            output_points.append((x2, ball_center_y))

        # if ball is within the horizontal space of a player, append player bbox points to ball
        if ball_center_x > x1 and ball_center_x < x2:
            output_points.append((ball_center_x, y1))
            output_points.append((ball_center_x, y2))

        output_points += [
            (x1, y1), # top left corner
            (x2, y1), # top right corner
            (x1, y2), # bottom left corner
            (x2, y2), # bottom right corner
            (x1 + width // 2, y1), # top center
            (x1 + width // 2, y2), # bottom center
            (x1, y1 + height // 2), # left center
            (x2, y1 + height // 2) # right center
        ]

        return output_points
    
    def find_min_distance_to_ball(self, ball_center, player_bbox):
        # get important points for ball distance
        key_points = self.get_key_basketball_player_assignment_points(player_bbox, ball_center)

        # find the closest player to ball connection
        return min(measure_distance(ball_center, key_point) for key_point in key_points)
    
    def calc_ball_containment_ratio(self, player_bbox, ball_bbox):
        # get player and ball points/area
        px1, py1, px2, py2 = player_bbox
        bx1, by1, bx2, by2 = ball_bbox

        player_area = (px2 - px1) * (py2 - py1)
        ball_area = (bx2 - bx1) * (by2 - by1)

        # find intersection between player and ball
        intersection_x1 = max(px1, bx1)
        intersection_y1 = max(py1, by1)
        intersection_x2 = min(px2, bx2)
        intersection_y2 = min(py2, by2)

        # if ball and player are not intersecting, return zero
        if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
            return 0

        # find intersection area and return what percentage is overlapped with player bbox
        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)

        return intersection_area / ball_area

    def find_best_possession_candidate(self, ball_center, player_tracks_frame, ball_bbox):
        # one list for containment overlap, one for distance from ball
        # priortize containment then distance for possession likelihood
        high_containment_players = []
        regular_distance_players = []

        for player_id, player_info in player_tracks_frame.items():
            # get player bbox, skip is not availible this frame
            player_bbox = player_info.get('bbox', [])
            if not player_bbox:
                continue

            # get player-ball containment/distance 
            containment = self.calc_ball_containment_ratio(player_bbox, ball_bbox)
            min_distance = self.find_min_distance_to_ball(ball_center, player_bbox)

            # populate lists with containment or distance
            if containment > self.containment_threshold:
                high_containment_players.append((player_id, containment))
            else:
                regular_distance_players.append((player_id, min_distance))

        # look through list of high containment players, get id of biggest overlap
        if high_containment_players:
            best_candidate = max(high_containment_players, key=lambda x : x[1])
            return best_candidate[0]
        
        # look through list of close distance players, get id of smallest distance
        if regular_distance_players:
            best_candidate = min(regular_distance_players, key=lambda x : x[1])
            if best_candidate[1] < self.possession_threshold:
                return best_candidate[0]

        # no one has possession 
        return -1
    
    def detect_ball_possession(self, player_tracks, ball_tracks):
        # initialize amount of frames to detect possession, who has possession, and dictionary to match frame to possession
        num_frames = len(ball_tracks)
        poss_list = [-1] * num_frames
        consec_poss_count = {}

        for frame_num in range(num_frames):
            # get ball_info (skip if no info on current frame)
            ball_info = ball_tracks[frame_num].get(1, {})
            if not ball_info:
                continue

            # then get ball bbox (skip if no bbox)
            ball_bbox = ball_info.get('bbox', [])
            if not ball_bbox:
                continue
            # get ball center
            ball_center = get_center_of_bbox(ball_bbox)
            # find best player id with possession
            best_player_id = self.find_best_possession_candidate(ball_center, player_tracks[frame_num], ball_bbox)

            # if we find a player with possession
            if best_player_id != -1:
                # increment the number of frames a player has had poss
                number_of_consec_frames = consec_poss_count.get(best_player_id, 0) + 1
                consec_poss_count = {best_player_id : number_of_consec_frames}

                # if a player has poss for >= the min number of frames for a poss, assign the new best_player
                if consec_poss_count[best_player_id] >= self.min_frames:
                    poss_list[frame_num] = best_player_id
            # otherwise reset the poss_count and find a new player with poss
            else:
                consec_poss_count = {}

        return poss_list

