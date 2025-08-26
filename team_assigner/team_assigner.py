from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import cv2

import sys
sys.path.append('../')
from utils import read_stub, save_stub

class TeamAssigner:
    def __init__(self, team_1_class_name='white shirt', team_2_class_name='dark blue shirt'):
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

        self.player_team_dict = {}

    def load_model(self):
        self.model = CLIPModel.from_pretrained('patrickjohncyh/fashion-clip')
        self.processor = CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')

    def get_player_color(self, frame, bbox):
        # get cropped image of player in bbox
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]

        # convert image from BGR to RGB to be processed by model
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        # process inputs/how to classify inputs (returns pytorch object)
        classes = [self.team_1_class_name, self.team_2_class_name]
        inputs = self.processor(text=classes, images=pil_image, return_tensors='pt', padding=True)

        # runs inputs on model (**inputs unpacks the dict into key:value pairs for model)
        outputs = self.model(**inputs)

        # creates a similarity score for the image-text pair and scales it to a percentage
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

        # assign most likely jersey color as label
        class_name = classes[probs.argmax(dim=1)[0]]
        return class_name
    
    def get_player_team(self, frame, bbox, player_id):
        # if player has already been processed, use previous classification 
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # get player's color
        player_color = self.get_player_color(frame, bbox)

        # if team color is closer to other team color, reclassify
        team_id = 2
        if player_color == self.team_1_class_name:
            team_id = 1

        # cache classification for future use
        self.player_team_dict[player_id] = team_id
        return team_id
    
    def get_player_teams_across_frames(self, video_frames, player_tracks, read_from_stub=False, stub_path=None):
        # if there is previous stub of frame and each previous frame has a previous assignment, use stub and don't load model
        player_assignment = read_stub(stub_path, read_from_stub)
        if player_assignment is not None and len(player_assignment) == len(video_frames):
            return player_assignment
        
        # otherwise load model and define assignment list
        self.load_model()
        player_assignment = []

        # for each player track
        for frame_num, player_track in enumerate(player_tracks):
            # define player_assignment log
            player_assignment.append({})

            # after 50 frames, reset player's team classification to fix incorrect classifications 
            if frame_num % 50 == 0:
                self.player_team_dict = {}

            # for the values in said track
            for player_id, track in player_track.items():
                # get the player's team classification based on the cut image in the bbox
                team = self.get_player_team(video_frames[frame_num], track['bbox'], player_id)
                # assign team classification to log by frame
                player_assignment[frame_num][player_id] = team

        # save current stub for future frames
        save_stub(stub_path, player_assignment)

        return player_assignment


