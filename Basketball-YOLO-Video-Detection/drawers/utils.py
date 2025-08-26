import cv2
import numpy as np

import sys
sys.path.append('../')
from utils import get_bbox_width, get_center_of_bbox

def draw_ellipse(frame, bbox, color, track_id=None):
    # get bbox info
    y2 = int(bbox[3])
    x_center, y_center = get_center_of_bbox(bbox)
    width = get_bbox_width(bbox)
    # draw ellipse
    cv2.ellipse(frame, center=(x_center, y2), 
                axes=(int(width), int(0.35 * width)), 
                angle=0, startAngle=-45, endAngle=235, 
                color=color, thickness=2, lineType=cv2.LINE_4
    )

    # get player info location
    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = y2 - rectangle_height // 2
    y2_rect = y2 + rectangle_height // 2

    if track_id is not None:
        # draw rectangle
        cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)

        # define text buffers
        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        # draw text
        cv2.putText(frame, str(track_id), (int(x1_text), int(y1_rect + 15)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
    return frame

def draw_triangle(frame, bbox, color):
    # gets traingle location
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    # gets traingle dimensions
    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20]
    ])

    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

    return frame