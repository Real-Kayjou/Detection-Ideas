import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ---------- Settings ----------
CAM_IDX = 0
SIDEBAR_W = 260
PURPLE = (255, 0, 255)   # BGR (magenta/purple)
BG_PURPLE = (120, 0, 120)
TEXT_COLOR = (255, 255, 255)
MOVEMENT_THRESH = 20     # pixels (tweak to your camera)
SMILE_RATIO_THRESH = 2.0 # mouth width / mouth height -> larger = smile
OPEN_MOUTH_THRESH = 0.45 # mouth height relative to face height
# ------------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(CAM_IDX)
prev_center = None
prev_time = time.time()
fps = 0

def norm_to_pixel(norm_landmark, frame_w, frame_h):
    return int(norm_landmark.x * frame_w), int(norm_landmark.y * frame_h)

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w = frame.shape[:2]
    sidebar_x = w - SIDEBAR_W
    frame = cv2.flip(frame, 1)  # mirror
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)


    # Sidebar title
    cv2.putText(frame, "FACE STATUS", (sidebar_x + 12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

    movement_text = "No face"
    expression_text = "N/A"

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Draw face mesh as purple lines (tesselation)
        mp_draw.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            mp_draw.DrawingSpec(color=PURPLE, thickness=1, circle_radius=0),
            mp_draw.DrawingSpec(color=PURPLE, thickness=1)
        )

        # Collect pixel coords
        pts = []
        for lm in face_landmarks.landmark:
            pts.append((int(lm.x * w), int(lm.y * h)))

        # Compute face center (average)
        cx = int(sum([p[0] for p in pts]) / len(pts))
        cy = int(sum([p[1] for p in pts]) / len(pts))
        face_center = (cx, cy)

        # draw small purple dot at center
        cv2.circle(frame, face_center, 3, PURPLE, -1)

        # Movement detection (compare to prev_center)
        if prev_center is None:
            movement_text = "Stable"
            movement_val = 0.0
        else:
            movement_val = dist(face_center, prev_center)
            movement_text = f"Moving: {'YES' if movement_val > MOVEMENT_THRESH else 'NO'}"
        prev_center = face_center

        # Expression detection (use mouth landmarks)
        # Common FaceMesh indices: left corner=61, right corner=291, upper=13, lower=14
        # (widely used mapping; works in many examples)
        try:
            left_corner = pts[61]
            right_corner = pts[291]
            upper_lip = pts[13]
            lower_lip = pts[14]
        except IndexError:
            left_corner = right_corner = upper_lip = lower_lip = None

        if left_corner and right_corner and upper_lip and lower_lip:
            mouth_w = dist(left_corner, right_corner)
            mouth_h = dist(upper_lip, lower_lip)
            # Face height approx: distance between top of forehead (10) and chin (152) as rough scale
            top_face = pts[10]
            chin = pts[152]
            face_h = dist(top_face, chin) if top_face and chin else h

            # Avoid division by zero
            if mouth_h > 1:
                ratio = mouth_w / mouth_h
            else:
                ratio = 0

            mouth_rel = mouth_h / face_h if face_h > 1 else 0

            if mouth_rel > OPEN_MOUTH_THRESH:
                expression_text = "Mouth: Open"
            elif ratio > SMILE_RATIO_THRESH:
                expression_text = "Expression: Smile"
            else:
                expression_text = "Expression: Neutral"

            

            # Show computed ratios in sidebar
            cv2.putText(frame, f"Move Δ: {movement_val:.1f}px", (sidebar_x + 12, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Mouth W/H: {ratio:.2f}", (sidebar_x + 12, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
            cv2.putText(frame, f"MouthRel: {mouth_rel:.2f}", (sidebar_x + 12, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
        else:
            expression_text = "Expression: Unknown"

        # Draw arrowed line from face center to middle of sidebar
        target_x = sidebar_x + 6
        target_y = int(h * 0.25)
        cv2.arrowedLine(frame, face_center, (target_x, target_y), PURPLE, 2, tipLength=0.03)

        # Put movement & expression inside the sidebar (big)
        cv2.putText(frame, movement_text, (sidebar_x + 12, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, expression_text, (sidebar_x + 12, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

    else:
        # No face detected: indicate in sidebar
        cv2.putText(frame, "No face detected", (sidebar_x + 12, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT_COLOR, 2, cv2.LINE_AA)

    # FPS calc & show
    curr_time = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PURPLE, 2)

    cv2.imshow("Face Lines + Sidebar (Purple)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
"""
#EYEES THING DONT TOUCH-----------------------------------------------------------------------------------------------------------------------------------------------------------
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Eye landmark indices (from Mediapipe FaceMesh)
            left_eye_indices = [33, 133]
            right_eye_indices = [362, 263]
            left_pupil_indices = [468]
            right_pupil_indices = [473]

            # Draw eye outlines
            for idx in left_eye_indices + right_eye_indices:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Draw pupils
            for idx in left_pupil_indices + right_pupil_indices:
                x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    cv2.imshow("Eye Tracking Test", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

cap.release()
cv2.destroyAllWindows()"""