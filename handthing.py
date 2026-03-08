# hand_lines.py
import cv2
import mediapipe as mp
import numpy as np

# --- Settings you can change ---
LINE_COLOR = (0, 0, 255)   # BGR color for skeleton lines (cyan)
JOINT_COLOR = (255, 0, 0) # color for joint dots (red)
LINE_THICKNESS = 4
JOINT_RADIUS = 6
GLOW_BLUR = (21, 21)         # must be odd numbers (bigger = more blur)
GLOW_WEIGHT = 0.7            # how strong the glow appears when added to frame
# -------------------------------

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Create a transparent mask to draw neon/skeleton on (same size as frame)
        mask = np.zeros_like(frame)  # black background

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Collect landmark pixel coordinates
                pts = []
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    pts.append((x, y))

                # Draw skeleton lines using MediaPipe's built-in connections
                for conn in mp_hands.HAND_CONNECTIONS:
                    start_idx, end_idx = conn
                    p1 = pts[start_idx]
                    p2 = pts[end_idx]
                    cv2.line(mask, p1, p2, LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA)

                # Draw joint circles
                for p in pts:
                    cv2.circle(mask, p, JOINT_RADIUS, JOINT_COLOR, -1, cv2.LINE_AA)

        # Blur mask to create glow / neon effect
        blurred = cv2.GaussianBlur(mask, GLOW_BLUR, 0)

        # Add blurred glow on top of original frame
        output = cv2.addWeighted(frame, 1.0, blurred, GLOW_WEIGHT, 0)

        # Optional: draw a faint non-blurred skeleton on top for crisp edges
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                pts = []
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    pts.append((x, y))
                for conn in mp_hands.HAND_CONNECTIONS:
                    s, e = conn
                    cv2.line(output, pts[s], pts[e], LINE_COLOR, 1, cv2.LINE_AA)
                for p in pts:
                    cv2.circle(output, p, max(2, JOINT_RADIUS // 2), JOINT_COLOR, -1, cv2.LINE_AA)

        cv2.imshow("Hand Lines", output)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
