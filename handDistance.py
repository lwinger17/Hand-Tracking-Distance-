import cv2
import math
import mediapipe as mp

mp_hands = mp.solutions.hands

class handDistance:
    def __init__(self, pixels_per_inch=100):
        self.pixels_per_inch = pixels_per_inch
    
    # distance math
    def calculate_distance(self, hand_lms, image_shape):
        wrist = hand_lms.landmark[mp_hands.HandLandmark.WRIST]
        pinky_tip = hand_lms.landmark[mp_hands.HandLandmark.PINKY_TIP]
        h, w, _ = image_shape
        wrist_x = int(wrist.x * w)
        wrist_y = int(wrist.y * h)
        pinky_tip_x = int(pinky_tip.x * w)
        pinky_tip_y = int(pinky_tip.y * h)
        hand_height = abs(wrist_y - pinky_tip_y)
        distance = round(1 / (hand_height / h), 2)  # Inverse of normalized hand height
        return distance

    # distance image
    def draw_distance(self, img, hand_lms, distance, hand_side, font_color, x_offset=0):
        h, w, c = img.shape
        if hand_side == 'left':
            x = int(hand_lms.landmark[mp_hands.HandLandmark.WRIST].x * w) - 50 + x_offset
        else:
            x = int(hand_lms.landmark[mp_hands.HandLandmark.WRIST].x * w) + 10 + x_offset
        y = int(hand_lms.landmark[mp_hands.HandLandmark.WRIST].y * h) - 20
        cv2.putText(img, f"Distance: {distance:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 2)

hand_distance = handDistance()