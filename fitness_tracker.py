import math
import os

import numpy as np
import cv2
import mediapipe as mp

class ExceptionError(Exception):
    pass

class FitnessTracker:
    def __init__(self, source_path=0):
        self.source_path = self.validate_video(source_path)
        self.color_lines = (255, 255, 255)

    def validate_video(self, path):
        video = cv2.VideoCapture(path)
        if not video.isOpen():
            raise ExceptionError("Could not found the source path, try it again")
        return video
    
    def get_landmark_point(landmarks, landmark_point, width, height):
        x = int(landmarks.landmark[landmark_point].x * width)
        y = int(landmarks.landmark[landmark_point].y * height)

        return np.array([x, y])
    
    
    def pull_ups(self):
        video = self.source_path
        win_name = "Pull-Ups"
        cv2.namedWindow(win_name)

        height = int(video.get(cv2.CAP_PROPFRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_WIDTH))

        mp_pose = mp.solutions.pose

    

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
            while True:
                has_frame, frame = video.read()

                if not has_frame:
                    break

            frame = cv2.flip(frame, 1)

            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                landmarks = results.pose_landmarks
                enum_pose = mp_pose.PoseLandmark

                # Left wrist
                l_wrist_x = int(landmarks.landmark[enum_pose.LEFT_WRIST].x * width)
                l_wrist_y = int(landmarks.landmark[enum_pose.LEFT_WRIST].y * height)

                # Left elbow
                l_elbow_x = int(landmarks.landmark[enum_pose.LEFT_ELBOW].x * width)
                l_elbow_y = int(landmarks.landmark[enum_pose.LEFT_ELBOW].y * height)

                # Left shoulder
                l_shoulder_x = int(landmarks.landmark[enum_pose.LEFT_SHOULDER].x * width)
                l_shoulder_y = int(landmarks.landmark[enum_pose.LEFT_SHOULDER].y * height)

                # Right wrist
                r_wrist_x = int(landmarks.landmark[enum_pose.RIGHT_WRIST].x * width)
                r_wrist_y = int(landmarks.landmark[enum_pose.RIGHT_WRIST].y *height)

                # Right elbow
                r_elbow_x = int(landmarks.landmark[enum_pose.RIGHT_ELBOW].x * width)
                r_elbow_y = int(landmarks.landmark[enum_pose.RIGHT_ELBOW].y * height)

                # Right shoulder
                r_shoulder_x = int(landmarks.landmark[enum_pose.RIGHT_SHOULDER].x * width)
                r_shoulder_y = int(landmarks.landmark[enum_pose.RIGHT_SHOULDER].y * height)

                # Join landmarks
                cv2.line(frame, (l_wrist_x, l_wrist_y), (l_elbow_x, l_elbow_y), self.color_lines, 2, cv2.LINE_AA)
                cv2.line(frame, (l_elbow_x, l_elbow_y), (l_shoulder_x, l_shoulder_y), self.color_lines, 2, cv2.LINE_AA)
                cv2.line(frame,  (l_shoulder_x, l_shoulder_y),  (r_shoulder_x, r_shoulder_y), self.color_lines, 2, cv2.LINE_AA)
                cv2.line(frame, (r_shoulder_x, r_shoulder_y), (r_elbow_x, r_elbow_y), self.color_lines, 2, cv2.LINE_AA)
                cv2.line(frame, (r_elbow_x, r_elbow_y), (r_wrist_x, r_wrist_y), self.color_lines, 2, cv2.LINE_AA)

                # Draw 