import cv2
import mediapipe as mp
from handSide import handSide
from handDistance import handDistance

def main():
    hs = handSide()
    hd = handDistance()
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # hand markings
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                hand_side = hs.identify_hand_side(hand_landmarks)
                distance = hd.calculate_distance(hand_landmarks, image.shape)  # Pass image.shape as the second argument

                if hand_side == 'left':
                    font_color = (0, 0, 255)  # blue
                    x = 10
                else:
                    font_color = (255, 0, 0)  # red
                    x = image.shape[1] - 150

                # distance label
                cv2.putText(image, f"{hand_side.capitalize()} Hand", (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 2)
                cv2.putText(image, "------", (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 2)
                cv2.putText(image, f"{distance:.2f} units", (x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 2)

                # hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=font_color, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=font_color, thickness=2, circle_radius=2),
                )

        cv2.imshow('Hand Tracking', image)

        # press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()