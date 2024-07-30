import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

class handSide:
    def __init__(self):
        pass

    def identify_hand_side(self, hand_lms):
        wrist = hand_lms.landmark[mp_hands.HandLandmark.WRIST]
        # change to > for camera side 
        if wrist.x < 0.5:
            return 'right'
        else:
            return 'left'

    def draw_hand_side(self, img, hand_lms, font_color):
        h, w, c = img.shape
        wrist = hand_lms.landmark[mp_hands.HandLandmark.WRIST]
        if wrist.x < 0.5:
            color = (0, 0, 255)  # blue = left
            text = 'Left'
        else:
            color = (255, 0, 0)  # red = right
            text = 'Right'

       
        return img