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
        self.colors = {
            "color_lines": (0, 0, 0),
            "color_circles": (32, 252, 3),
            "initial_color_line": (128, 128, 128),
            "wrist": (242, 71, 15),
            "elbow": (0, 208, 255),
            "shoulder": (235, 7, 106),
        }
        

    def validate_video(self, path):
        video = cv2.VideoCapture(path)
        if not video.isOpened():
            raise ExceptionError("Could not found the source path, try it again")
        return video
    
    def get_landmark_point(self, landmarks, landmark_point, width, height):
        x = int(landmarks.landmark[landmark_point].x * width)
        y = int(landmarks.landmark[landmark_point].y * height)

        return np.array([x, y])
    
    def compute_angle(self, v1, v2):
        # Unit vector.
        v1u = v1 / np.linalg.norm(v1)
        # unit vector.
        v2u = v2 / np.linalg.norm(v2)
        # Compute the angle between the two unit vectors.
        angle_deg = np.arccos(np.dot(v1u, v2u)) * 180 / math.pi

        return angle_deg

    def pull_ups(self):
        
        UP_ANGLE_MIN = 7
        UP_ANGLE_MAX = 25

        SHOULDER_ANGLE_MIN = 128
        SHOULDER_ANGLE_MAX = 145
        video = self.source_path

        win_name = "Pull-Ups"
        cv2.namedWindow(win_name)

        first_frame = True
        over_head = False

        counter = 0

        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

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
                    l_wrist_p = self.get_landmark_point(landmarks, enum_pose.LEFT_WRIST, width, height)

                    # Left elbow
                    l_elbow_p = self.get_landmark_point(landmarks, enum_pose.LEFT_ELBOW, width, height)

                    # Left shoulder
                    l_shoulder_p = self.get_landmark_point(landmarks, enum_pose.LEFT_SHOULDER, width, height)

                    # Right wrist
                    r_wrist_p = self.get_landmark_point(landmarks, enum_pose.RIGHT_WRIST, width, height)

                    # Right elbow
                    r_elbow_p = self.get_landmark_point(landmarks, enum_pose.RIGHT_ELBOW, width, height)

                    # Right shoulder
                    r_shoulder_p = self.get_landmark_point(landmarks, enum_pose.RIGHT_SHOULDER, width, height)

                    # center head
                    center_head_p = self.get_landmark_point(landmarks, enum_pose.NOSE, width, height)

                    # Compute angles
                    v_wrist_elbow_left = np.subtract(l_wrist_p, l_elbow_p)
                    v_elbow_shoulder_left = np.subtract(l_shoulder_p, l_elbow_p)

                    v_wrist_elbow_right = np.subtract(r_elbow_p, r_wrist_p)
                    v_elbow_shoulder_right = np.subtract(r_elbow_p, r_shoulder_p)

                    v_shoulder_to_shoulder = np.subtract(l_shoulder_p, r_shoulder_p)

                    angle_left = self.compute_angle(v_wrist_elbow_left, v_elbow_shoulder_left)
                    angle_right = self.compute_angle(v_wrist_elbow_right, v_elbow_shoulder_right)

                    angle_shoulder_left = self.compute_angle(v_elbow_shoulder_left, v_shoulder_to_shoulder)
                    angle_shoulder_right = self.compute_angle(v_elbow_shoulder_right, v_shoulder_to_shoulder)


                    # Place text
                    text_loc_left = (l_wrist_p[0] - 25, l_wrist_p[1] - 10)
                    cv2.putText(frame, str(int(angle_left)), text_loc_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["wrist"], 2, cv2.LINE_AA)

                    text_loc_right = (r_wrist_p[0] - 25, r_wrist_p[1] - 10)
                    cv2.putText(frame, str(int(angle_right)), text_loc_right, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["wrist"], 2, cv2.LINE_AA)
                    
                    text_loc_left_sts = (l_shoulder_p[0] - 25, l_shoulder_p[1] - 15)
                    cv2.putText(frame, str(int(angle_shoulder_left)), text_loc_left_sts, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["shoulder"], 2, cv2.LINE_AA)

                    text_loc_right_sts = (r_shoulder_p[0] - 25, r_shoulder_p[1] - 15)
                    cv2.putText(frame, str(int(angle_shoulder_right)), text_loc_right_sts, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["elbow"], 2, cv2.LINE_AA)

                    # Join landmarks
                    cv2.line(frame, l_wrist_p, l_elbow_p, self.colors["color_lines"], 2, cv2.LINE_AA)
                    cv2.line(frame, l_elbow_p, l_shoulder_p, self.colors["color_lines"], 2, cv2.LINE_AA)
                    cv2.line(frame,  l_shoulder_p,  r_shoulder_p, self.colors["color_lines"], 2, cv2.LINE_AA)
                    cv2.line(frame, r_shoulder_p, r_elbow_p, self.colors["color_lines"], 2, cv2.LINE_AA)
                    cv2.line(frame, r_elbow_p, r_wrist_p, self.colors["color_lines"], 2, cv2.LINE_AA)

                    # Draw landmarks
                    cv2.circle(frame, l_wrist_p, 3, self.colors["color_circles"], -1)
                    cv2.circle(frame, l_elbow_p, 3, self.colors["color_circles"], -1)
                    cv2.circle(frame, l_shoulder_p, 3, self.colors["color_circles"], -1)
                    cv2.circle(frame, r_shoulder_p, 3, self.colors["color_circles"], -1)
                    cv2.circle(frame, r_elbow_p, 3, self.colors["color_circles"], -1)
                    cv2.circle(frame, r_wrist_p, 3, self.colors["color_circles"], -1)

                    if first_frame:
                        p_center_head_p = self.get_landmark_point(landmarks, enum_pose.NOSE, width, height)
                        first_frame = False

                    # Horizontal line
                    cv2.line(frame, (0, p_center_head_p[1]), (frame.shape[1] - 1, p_center_head_p[1]), self.colors["initial_color_line"], 1, cv2.LINE_AA)
                    cv2.line(frame, (0, center_head_p[1]), (frame.shape[1] - 1, center_head_p[1]), (0, 0, 255), 1, cv2.LINE_AA)

                    # Point center head
                    cv2.circle(frame, center_head_p, 3, self.colors["color_circles"], -1)

                    DOWN_ANGLE_THRESHOLD = 160


                    is_up_position = (
                        # Flexed elbows
                        (angle_left >= UP_ANGLE_MIN and angle_left <= UP_ANGLE_MAX) and
                        (angle_right >= UP_ANGLE_MIN and angle_right <= UP_ANGLE_MAX) and
                        
                        # Flexed shoulders
                        (angle_shoulder_left >= SHOULDER_ANGLE_MIN and angle_shoulder_left <= SHOULDER_ANGLE_MAX) and
                        (angle_shoulder_right >= SHOULDER_ANGLE_MIN and angle_shoulder_right <= SHOULDER_ANGLE_MAX) and
                        
                        # Head hight
                        (center_head_p[1] <= p_center_head_p[1] + 60)
                    )

                    is_down_position = (
                        (angle_left >= DOWN_ANGLE_THRESHOLD) and
                        (angle_right >= DOWN_ANGLE_THRESHOLD)
                    )

                    if is_up_position:
                        over_head = True

                    elif is_down_position and over_head:
                        counter += 1
                        over_head = False

                    # print how many push-ups
                    frame[25:70, 0:40] = 0
                    cv2.putText(frame, str(counter), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                            

                cv2.imshow(win_name, frame)
                key = cv2.waitKey(10)

                if key == 27:
                    break
        video.release()
        cv2.destroyAllWindows()

    def normal_squats(self):
        
        # Down
        DOWN_ANGLE_HIP_MAX = 55

        KNEE_ANGLE_MAX = 47

        # Up
        UP_ANGLE_HIP_MIN = 147

        UP_KNEE_ANGLE_MIN = 158

        video = self.source_path

        first_frame = True
        over_head = False

        counter = 0

        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

        win_name = "Normal Squats"
        cv2.namedWindow(win_name)

        mp_pose = mp.solutions.pose

        with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
            while True:
                has_frame, frame = video.read()

                if not has_frame:
                    break

                # frame = cv2.flip(frame, 1)

                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks
                    enum_pose = mp_pose.PoseLandmark

                    # Left ear
                    l_ear_p = self.get_landmark_point(landmarks, enum_pose.LEFT_EAR, width, height)

                    # Left hip
                    l_hip_p = self.get_landmark_point(landmarks, enum_pose.LEFT_HIP, width, height)

                    # Left knee
                    l_knee_p = self.get_landmark_point(landmarks, enum_pose.LEFT_KNEE, width, height)
                    
                    # left ankle
                    l_ankle_p = self.get_landmark_point(landmarks, enum_pose.LEFT_ANKLE, width, height)

                    ##############
                    # Compute Angles
                    v_ear_hip_left = np.subtract(l_hip_p, l_ear_p)
                    v_knee_hip_left = np.subtract(l_hip_p, l_knee_p)

                    v_hip_knee_left = np.subtract(l_knee_p, l_hip_p)
                    v_ankle_knee_left = np.subtract(l_knee_p, l_ankle_p)

                    angle_hip = self.compute_angle(v_ear_hip_left, v_knee_hip_left)
                    angle_knee = self.compute_angle(v_hip_knee_left, v_ankle_knee_left)

                    ##############
                    # Place text
                    text_loc_left = (l_ear_p[0] + 125, l_hip_p[1] - 10)
                    cv2.putText(frame, str(int(angle_hip)), text_loc_left, cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 2, cv2.LINE_AA)

                    text_loc_right = (l_hip_p[0] - 150, l_knee_p[1] - 10)
                    cv2.putText(frame, str(int(angle_knee)), text_loc_right, cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2, cv2.LINE_AA)

                    ##############
                    # Join landmarks
                    cv2.line(frame, l_ear_p, l_hip_p, self.colors["color_lines"], 2, cv2.LINE_AA)
                    cv2.line(frame, l_hip_p, l_knee_p, self.colors["color_lines"], 2, cv2.LINE_AA)
                    cv2.line(frame, l_knee_p, l_ankle_p, self.colors["color_lines"], 2, cv2.LINE_AA)

                    ##############
                    # Draw landmarks
                    cv2.circle(frame, l_ear_p, 3, self.colors["color_circles"], -1)
                    cv2.circle(frame, l_hip_p, 3, self.colors["color_circles"], -1)
                    cv2.circle(frame, l_knee_p, 3, self.colors["color_circles"], -1)
                    cv2.circle(frame, l_ankle_p, 3, self.colors["color_circles"], -1)

                    if first_frame:
                        first_ear_p = self.get_landmark_point(landmarks, enum_pose.LEFT_EAR, width, height)
                        first_frame = False

                    # Horizontal Line
                    cv2.line(frame, (0, first_ear_p[1]), (frame.shape[1] - 1, first_ear_p[1]), (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.line(frame, (0, l_ear_p[1]), (frame.shape[1] - 1, l_ear_p[1]), (46, 46, 46), 1, cv2.LINE_AA)

                    is_down_position = (
                        angle_knee <= KNEE_ANGLE_MAX and angle_hip <= DOWN_ANGLE_HIP_MAX and
                        l_ear_p[1] > (first_ear_p[1] + 50)
                    )

                    is_up_position = (
                        angle_knee >= UP_KNEE_ANGLE_MIN and angle_hip >= UP_ANGLE_HIP_MIN and
                        l_ear_p[1] <= (first_ear_p[1] + 20)
                    )

                    if is_down_position:
                        over_head = True

                    elif is_up_position and over_head:
                        counter += 1
                        over_head = False

                    # Print how many squats
                    frame[25:70, 0:35] = 0
                    cv2.putText(frame, str(counter), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow(win_name, frame)

                key = cv2.waitKey(1)

                if key == 27:
                    break

        video.release()
        cv2.destroyAllWindows()
