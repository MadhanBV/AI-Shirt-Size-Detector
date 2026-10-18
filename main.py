import cv2
import mediapipe as mp
import math
from collections import deque
from size_predictor import predict_shirt_size

# ============================================================
# 1. MODEL SETUP
# ============================================================

MODEL_PATH = "pose_landmarker_heavy.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = PoseLandmarker.create_from_options(options)


# ============================================================
# 2. USER CALIBRATION
# ============================================================

# Your known height
USER_HEIGHT_CM = 172

# Smoothed body-span calibration obtained earlier
AVERAGE_BODY_SPAN_PX = 326

# Approximate conversion factor
CM_PER_PIXEL = USER_HEIGHT_CM / AVERAGE_BODY_SPAN_PX


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def distance(point1, point2):
    return math.sqrt(
        (point2[0] - point1[0]) ** 2 +
        (point2[1] - point1[1]) ** 2
    )


def interpolate_point(point1, point2, ratio):
    """
    Returns a point between point1 and point2.

    ratio = 0   -> point1
    ratio = 1   -> point2
    """

    x = point1[0] + (point2[0] - point1[0]) * ratio
    y = point1[1] + (point2[1] - point1[1]) * ratio

    return (x, y)


def pixels_to_cm(value):
    return value * CM_PER_PIXEL


# ============================================================
# 4. SHIRT SIZE PREDICTION
# ============================================================


# ============================================================
# 5. SMOOTHING
# ============================================================

measurement_history = deque(maxlen=30)


# ============================================================
# 6. CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    landmarker.close()
    exit()


print("\n==========================================")
print("     AI SHIRT SIZE DETECTOR")
print("==========================================")
print("Camera started.")
print("Stand far enough so your full body is visible.")
print("Press Q to quit.")
print("==========================================\n")


# ============================================================
# 7. MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read from camera.")
        break

    # --------------------------------------------------------
    # Convert OpenCV BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # --------------------------------------------------------
    # Pose detection
    # --------------------------------------------------------

    result = landmarker.detect(mp_image)

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        width = frame.shape[1]
        height = frame.shape[0]

        # ----------------------------------------------------
        # Convert normalized landmarks → pixel coordinates
        # ----------------------------------------------------

        points = []

        for landmark in landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

            # Draw landmark
            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # ----------------------------------------------------
        # Important body points
        # ----------------------------------------------------

        left_shoulder = points[11]
        right_shoulder = points[12]

        left_elbow = points[13]
        right_elbow = points[14]

        left_wrist = points[15]
        right_wrist = points[16]

        left_hip = points[23]
        right_hip = points[24]

        # ----------------------------------------------------
        # SHOULDER WIDTH
        # ----------------------------------------------------

        shoulder_px = distance(
            left_shoulder,
            right_shoulder
        )

        # ----------------------------------------------------
        # HIP WIDTH
        # ----------------------------------------------------

        hip_px = distance(
            left_hip,
            right_hip
        )

        # ----------------------------------------------------
        # CHEST WIDTH PROXY
        # ----------------------------------------------------
        #
        # 65% from shoulder toward hip
        # ----------------------------------------------------

        chest_left = interpolate_point(
            left_shoulder,
            left_hip,
            0.35
        )

        chest_right = interpolate_point(
            right_shoulder,
            right_hip,
            0.35
        )

        chest_px = distance(
            chest_left,
            chest_right
        )

        # ----------------------------------------------------
        # WAIST WIDTH PROXY
        # ----------------------------------------------------
        #
        # 65% from shoulder toward hip
        # ----------------------------------------------------

        waist_left = interpolate_point(
            left_shoulder,
            left_hip,
            0.65
        )

        waist_right = interpolate_point(
            right_shoulder,
            right_hip,
            0.65
        )

        waist_px = distance(
            waist_left,
            waist_right
        )

        # ----------------------------------------------------
        # LEFT ARM
        # ----------------------------------------------------

        left_upper_arm_px = distance(
            left_shoulder,
            left_elbow
        )

        left_forearm_px = distance(
            left_elbow,
            left_wrist
        )

        left_arm_px = (
            left_upper_arm_px +
            left_forearm_px
        )

        # ----------------------------------------------------
        # RIGHT ARM
        # ----------------------------------------------------

        right_upper_arm_px = distance(
            right_shoulder,
            right_elbow
        )

        right_forearm_px = distance(
            right_elbow,
            right_wrist
        )

        right_arm_px = (
            right_upper_arm_px +
            right_forearm_px
        )

        # Average both arms
        arm_px = (
            left_arm_px +
            right_arm_px
        ) / 2

        # ----------------------------------------------------
        # TORSO LENGTH
        # ----------------------------------------------------

        left_torso_px = distance(
            left_shoulder,
            left_hip
        )

        right_torso_px = distance(
            right_shoulder,
            right_hip
        )

        torso_px = (
            left_torso_px +
            right_torso_px
        ) / 2

        # ----------------------------------------------------
        # CONVERT TO CM
        # ----------------------------------------------------

        shoulder_cm = pixels_to_cm(shoulder_px)
        chest_cm = pixels_to_cm(chest_px)
        waist_cm = pixels_to_cm(waist_px)
        hip_cm = pixels_to_cm(hip_px)
        arm_cm = pixels_to_cm(arm_px)
        torso_cm = pixels_to_cm(torso_px)

        # ----------------------------------------------------
        # SMOOTH MEASUREMENTS
        # ----------------------------------------------------

        current_measurements = (
            shoulder_cm,
            chest_cm,
            waist_cm,
            hip_cm,
            arm_cm,
            torso_cm
        )

        measurement_history.append(
            current_measurements
        )

        # Average last 30 frames
        if len(measurement_history) > 0:

            averaged = []

            for i in range(6):

                value = sum(
                    measurement[i]
                    for measurement in measurement_history
                ) / len(measurement_history)

                averaged.append(value)

            (
                shoulder_cm,
                chest_cm,
                waist_cm,
                hip_cm,
                arm_cm,
                torso_cm
            ) = averaged

        # ----------------------------------------------------
        # PREDICT SHIRT SIZE
        # ----------------------------------------------------

        shirt_size = predict_shirt_size(
            shoulder_cm,
            chest_cm
        )

        # ----------------------------------------------------
        # DRAW BODY CONNECTIONS
        # ----------------------------------------------------

        connections = [

            # Shoulders
            (11, 12),

            # Left arm
            (11, 13),
            (13, 15),

            # Right arm
            (12, 14),
            (14, 16),

            # Torso
            (11, 23),
            (12, 24),

            # Hips
            (23, 24),

            # Legs
            (23, 25),
            (25, 27),

            (24, 26),
            (26, 28)
        ]

        for start, end in connections:

            x1 = int(
                landmarks[start].x * width
            )

            y1 = int(
                landmarks[start].y * height
            )

            x2 = int(
                landmarks[end].x * width
            )

            y2 = int(
                landmarks[end].y * height
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # DRAW MEASUREMENT LINES
        # ----------------------------------------------------

        # Shoulder line
        cv2.line(
            frame,
            left_shoulder,
            right_shoulder,
            (255, 0, 0),
            3
        )

        # Chest line
        chest_left_pixel = (
            int(chest_left[0]),
            int(chest_left[1])
        )

        chest_right_pixel = (
            int(chest_right[0]),
            int(chest_right[1])
        )

        cv2.line(
            frame,
            chest_left_pixel,
            chest_right_pixel,
            (255, 0, 255),
            3
        )

        # Waist line
        waist_left_pixel = (
            int(waist_left[0]),
            int(waist_left[1])
        )

        waist_right_pixel = (
            int(waist_right[0]),
            int(waist_right[1])
        )

        cv2.line(
            frame,
            waist_left_pixel,
            waist_right_pixel,
            (0, 255, 255),
            3
        )

        # Hip line
        cv2.line(
            frame,
            left_hip,
            right_hip,
            (0, 165, 255),
            3
        )

        # ----------------------------------------------------
        # DISPLAY MEASUREMENTS
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Shoulder: {shoulder_cm:.1f} cm",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Chest Width: {chest_cm:.1f} cm",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Waist Width: {waist_cm:.1f} cm",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Hip Width: {hip_cm:.1f} cm",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Arm Length: {arm_cm:.1f} cm",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Torso Length: {torso_cm:.1f} cm",
            (20, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # SIZE RECOMMENDATION
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"RECOMMENDED SIZE: {shirt_size}",
            (20, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 255, 255),
            3
        )

    else:

        # ----------------------------------------------------
        # NO PERSON DETECTED
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "NO PERSON DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------------
    # SHOW CAMERA
    # --------------------------------------------------------

    cv2.imshow(
        "AI Shirt Size Detector",
        frame
    )

    # --------------------------------------------------------
    # PRESS Q TO EXIT
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 8. CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

landmarker.close()

print("\nAI Shirt Size Detector stopped.")