import cv2
import mediapipe as mp
import math
from collections import deque


# ============================================================
# MODEL SETUP
# ============================================================

MODEL_PATH = "pose_landmarker_heavy.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def distance(point1, point2):

    return math.sqrt(
        (point2[0] - point1[0]) ** 2 +
        (point2[1] - point1[1]) ** 2
    )


def interpolate_point(point1, point2, ratio):

    x = point1[0] + (
        point2[0] - point1[0]
    ) * ratio

    y = point1[1] + (
        point2[1] - point1[1]
    ) * ratio

    return (x, y)


# ============================================================
# MEASUREMENT ENGINE
# ============================================================

class MeasurementEngine:

    def __init__(self):

        self.landmarker = (
            PoseLandmarker.create_from_options(options)
        )

        # ====================================================
        # CALIBRATION (PROTOTYPE)
        # ====================================================
        # Fixed reference calibration based on a known height of 172 cm
        # and an empirical 326 px vertical body span in standard 480p space.
        # NOTE: Webcam perspective, lens field-of-view, distance from
        # camera, and user posture significantly affect measurement accuracy.
        # This is an AI-assisted clothing size prototype, not a medical or
        # professional body tailoring tool.
        # Structured for future extension with real-time reference scale estimation.
        self.user_height_cm = 172
        self.average_body_span_px = 326

        self.cm_per_pixel = (
            self.user_height_cm /
            self.average_body_span_px
        )

        # Smooth measurements
        self.history = deque(maxlen=30)


    # ========================================================
    # PROCESS IMAGE
    # ========================================================

    def process_frame(self, frame):

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks:

            return None

        landmarks = result.pose_landmarks[0]

        width = frame.shape[1]
        height = frame.shape[0]


        # ----------------------------------------------------
        # Convert landmarks to pixel coordinates
        # ----------------------------------------------------

        points = []

        for landmark in landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            points.append((x, y))


        # ----------------------------------------------------
        # Important landmarks
        # ----------------------------------------------------

        left_shoulder = points[11]
        right_shoulder = points[12]

        left_elbow = points[13]
        right_elbow = points[14]

        left_wrist = points[15]
        right_wrist = points[16]

        left_hip = points[23]
        right_hip = points[24]


        # ====================================================
        # MEASUREMENTS
        # ====================================================

        # Shoulder
        shoulder_px = distance(
            left_shoulder,
            right_shoulder
        )


        # Hip
        hip_px = distance(
            left_hip,
            right_hip
        )


        # ----------------------------------------------------
        # Chest
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
        # Waist
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
        # Left arm
        # ----------------------------------------------------

        left_arm_px = (
            distance(
                left_shoulder,
                left_elbow
            )
            +
            distance(
                left_elbow,
                left_wrist
            )
        )


        # ----------------------------------------------------
        # Right arm
        # ----------------------------------------------------

        right_arm_px = (
            distance(
                right_shoulder,
                right_elbow
            )
            +
            distance(
                right_elbow,
                right_wrist
            )
        )


        # Average arm length
        arm_px = (
            left_arm_px +
            right_arm_px
        ) / 2


        # ----------------------------------------------------
        # Torso
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


        # ====================================================
        # PIXELS → CM
        # ====================================================

        shoulder_cm = (
            shoulder_px *
            self.cm_per_pixel
        )

        chest_cm = (
            chest_px *
            self.cm_per_pixel
        )

        waist_cm = (
            waist_px *
            self.cm_per_pixel
        )

        hip_cm = (
            hip_px *
            self.cm_per_pixel
        )

        arm_cm = (
            arm_px *
            self.cm_per_pixel
        )

        torso_cm = (
            torso_px *
            self.cm_per_pixel
        )


        # ====================================================
        # SMOOTHING
        # ====================================================

        measurements = (
            shoulder_cm,
            chest_cm,
            waist_cm,
            hip_cm,
            arm_cm,
            torso_cm
        )

        self.history.append(measurements)


        averaged = []

        for i in range(6):

            average = sum(
                item[i]
                for item in self.history
            ) / len(self.history)

            averaged.append(average)


        (
            shoulder_cm,
            chest_cm,
            waist_cm,
            hip_cm,
            arm_cm,
            torso_cm
        ) = averaged


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {

            "shoulder_cm": round(
                shoulder_cm, 2
            ),

            "chest_width_cm": round(
                chest_cm, 2
            ),

            "waist_cm": round(
                waist_cm, 2
            ),

            "hip_cm": round(
                hip_cm, 2
            ),

            "arm_cm": round(
                arm_cm, 2
            ),

            "torso_cm": round(
                torso_cm, 2
            )
        }


    # ========================================================
    # CLOSE MODEL
    # ========================================================

    def close(self):

        self.landmarker.close()