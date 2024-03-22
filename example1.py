# import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

model_path = "C:\Python_Works\AI\mediapipe\model\pose_landmarker_lite.task"

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


plus_flag = False  #肩よりx方向にプラスの位置しているかどうか
minus_flag = False #肩よりx方向にマイナスの位置にあるかどうか
counter = 0

# 手振り確認関数
def check_wave_hand(base,point):
    #変数定義
    # counter = 0 #手振りの回数をカウント
    global counter
    global plus_flag 
    global minus_flag 
    print(base)
    print(point)
    point_x = point.x
    base_x = base.x

    if point_x > base_x:
        plus_flag = True
        print("qwerty")
        return 
    if point_x < base_x:
        minus_flag = True
        print("00000")
    if plus_flag == True & minus_flag == True:
        counter += 1
        plus_flag = False
        minus_flag = False
    
    
    #肩よりプラスの位置にあるか

# Webカメラから入力
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        print(results.pose_landmarks.landmark[0])
        # 右肩のランドマークを取得
        r_shoulder_landmark = results.pose_landmarks.landmark[12] 
        # 右手首のランドマークを取得
        r_wrist_landmark = results.pose_landmarks.landmark[16]
        print(r_shoulder_landmark.x)
        print(r_wrist_landmark.x)
        print("================================================")
        # 検出されたポーズの骨格をカメラ画像に重ねて描画
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #画像サイズの取得
        height, width, channels = image.shape[:3]
        print("width: " + str(width))
        print("height: " + str(height))

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
        landmark_list = results.pose_landmarks
        # print(landmark_list.landmark[12])
        print("!!!")

        #手振りの検知を行う
        check_wave_hand(r_shoulder_landmark,r_wrist_landmark)
        print(counter)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()