import cv2
from ultralytics import YOLO
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


plus_flag = False  #肩よりx方向にプラスの位置しているかどうか
minus_flag = False #肩よりx方向にマイナスの位置にあるかどうか
counter = 0


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
    


model = YOLO('yolov8n.pt')

video_path = 0 # 本体に付属のカメラを指定
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            # 推論を実行。人を検知
            results = model.predict(frame,classes=0)

            # キャプチャした画像サイズを取得
            imageWidth = results[0].orig_shape[0]
            imageHeight = results[0].orig_shape[1]

            # 後のオブジェクト名出力などのため
            names = results[0].names
            classes = results[0].boxes.cls
            boxes = results[0].boxes
            annotatedFrame = results[0].plot()
            count = 0
            v_list = []
            # 検出したオブジェクトのバウンディングボックス座標とオブジェクト名を取得し、ターミナルに出力
            for box, cls in zip(boxes, classes):
                name = names[int(cls)]
                
                #人物画像の切り抜き
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                print(f"{x1},{x2},{y1},{y2}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                timg = frame[y1:y2,x1:x2]
                v_list.append(timg)
                #推定用に画像サイズを統一。カメラの画角を元に余白を追加
                f_height, f_width, channels = frame.shape[:3]
                # print("width: " + str(f_width))
                # print("height: " + str(f_height))
                # image = cv2.copyMakeBorder(image,y1,f_height-y2,x1,f_width-x2,cv2.BORDER_CONSTANT,value= [255,0,0])
                # image = cv2.copyMakeBorder(image,50,50,50,50,cv2.BORDER_CONSTANT,value= [255,0,0])
                #切り取った人物画像から、骨格推定 --------------------------------------------
                results = pose.process(timg)

                if not results.pose_landmarks:
                    continue

                print(results.pose_landmarks.landmark[0])
                # # 右肩のランドマークを取得
                r_shoulder_landmark = results.pose_landmarks.landmark[12] 
                # # 右手首のランドマークを取得
                r_wrist_landmark = results.pose_landmarks.landmark[16]
                # print(r_shoulder_landmark.x)
                # print(r_wrist_landmark.x)
                # print("================================================")
                # # 検出されたポーズの骨格をカメラ画像に重ねて描画
                # image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                height, width, channels = timg.shape[:3]
                print("width: " + str(width))
                print("height: " + str(height))

                mp_drawing.draw_landmarks(
                    timg,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # #推定した骨格座標から手振りの判定 --------------------------------------------
                check_wave_hand(r_shoulder_landmark,r_wrist_landmark)
                print(counter)
                
                # cv2.imwrite(f"{name}_timg{count}.jpg",timg)
                # print(f"Object: {name} Coordinates: StartX={x1}, StartY={y1}, EndX={x2}, EndY={y2}")
                # バウンディングBOXの座標情報を書き込む
                cv2.putText(annotatedFrame, f"{x1} {y1} {x2} {y2}", (x1, y1 - 40), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2, cv2.LINE_AA)
                count += 1
            # merge_img = cv2.hconcat(v_list)

            # プレビューウィンドウに画像出力
            cv2.imshow("YOLOv8 Inference", annotatedFrame)
            cv2.namedWindow("MediaPipe Pose", cv2.WINDOW_NORMAL)
            cv2.imshow("MediaPipe Pose",timg)
            # if count >= 1:
            #     cv2.imshow('MediaPipe Pose', merge_img)
            # else:
            #     cv2.imshow('example',timg)
            # アプリケーション終了
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break	
        else: 
            # キャプチャに失敗したら終了
            break
        
# 終了処理
cap.release()
cv2.destroyAllWindows()