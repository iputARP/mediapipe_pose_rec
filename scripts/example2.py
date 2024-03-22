#pose_simple.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from mediapipe import solutions

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
VisionRunningMode = mp.tasks.vision.RunningMode
pose_model = "C:\Python_Works\AI\mediapipe\model\pose_landmarker_heavy.task"


def draw_landmarks_on_image(rgb_image, detection_result):
    print("aaaaaa")
    pose_landmarks_list = detection_result.pose_landmarks
   
    annotated_image = np.copy(rgb_image)
   
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image



# Webカメラから入力
cap = cv2.VideoCapture(1)

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path=pose_model)
options = vision.PoseLandmarkerOptions(
    # running_mode=VisionRunningMode.LIVE_STREAM,
    base_options=base_options,
    num_poses = 10,
    min_pose_detection_confidence=0.4,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
    image.flags.writeable = False
    #カメラ画像の取得
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    # mp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = mp.Image.create_from_file(image)
    #姿勢の認識
    print(image)
    
    results = detector.detect(mp_image)
    # print("detection_result.pose_landmarks")
   
    # 右肩、右手首のランドマークを取得
    # r_shoulder_landmark = results.pose_landmarks.landmark[12] 
    # r_wrist_landmark = results.pose_landmarks.landmark[16]
    rec_num = len(results.pose_landmarks)
    print(f"人数は{rec_num}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("================================================")
    # 検出されたポーズの骨格をカメラ画像に重ねて描画
    # image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #画像サイズの取得
    # height, width, channels = image.shape[:3]
    # print("width: " + str(width))
    # print("height: " + str(height))

    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), results)
    print("================================================")
    # mp_drawing.draw_landmarks(
    #     image,
    #     results.pose_landmarks,
    #     mp_pose.POSE_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    print(annotated_image)
    # cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.imshow('MediaPipe Pose',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    # landmark_list = results.pose_landmarks
    # print(landmark_list.landmark[12])
    print("!!!")

    #手振りの検知を行う
    # check_wave_hand(r_shoulder_landmark,r_wrist_landmark)
    # print(counter)

    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()





# # STEP 3: Load the input image.
# image = mp.Image.create_from_file(pose_image)

# # STEP 4: Detect pose landmarks from the input image.
# detection_result = detector.detect(image)

# # Check if landmarks were detected
# if detection_result.pose_landmarks:
#     landmarks = detection_result.pose_landmarks[0]  # Get landmarks of the first detected person

#     # Landmark index for nose and wrists
#     NOSE_INDEX = 0
#     LEFT_WRIST_INDEX = 15
#     RIGHT_WRIST_INDEX = 16

#     nose = landmarks[NOSE_INDEX]
#     left_wrist = landmarks[LEFT_WRIST_INDEX]
#     right_wrist = landmarks[RIGHT_WRIST_INDEX]

#     if left_wrist.y < nose.y and right_wrist.y < nose.y:
#         print("Both wrists are above the nose position.")
#     else :print("non")
