import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipeの顔検出・ランドマーク検出モデルを初期化
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 描画設定
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Webカメラからの入力を開始
# ファイルから読み込む場合は cv2.VideoCapture('video.mp4') のようにファイルパスを指定
cap = cv2.VideoCapture(0)

# スムージング用の係数
SMOOTHING_FACTOR = 0.15
smoothed_pose = None

# MediaPipeの顔ランドマーク検出モデルを使用
with mp_face_mesh.FaceMesh(
    max_num_faces=1, # 検出する顔の最大数
    refine_landmarks=True, # 瞳などの細かいランドマークも取得
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # パフォーマンス向上のため、画像を書き込み不可にして参照渡しにする
        image.flags.setflags(write=False)
        # BGR画像をRGBに変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 顔ランドマークを検出
        results = face_mesh.process(image)

        # 画像を書き込み可能に戻し、BGRに再変換
        image.flags.setflags(write=True)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        img_h, img_w, _ = image.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # --- 1. ランドマークの描画 ---
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                # --- 2. 頭部姿勢の推定 ---
                # 必要なランドマークの座標を取得 (正規化されているので画像サイズを掛ける)
                landmarks = face_landmarks.landmark
                
                # JS版のロジックに似せた簡易的な推定
                # 左右の目、鼻、顎のランドマークを使用
                # MediaPipeのランドマークインデックスはドキュメント参照
                right_eye_x = landmarks[33].x * img_w
                left_eye_x = landmarks[263].x * img_w
                
                nose_tip_x = landmarks[1].x * img_w
                nose_tip_y = landmarks[1].y * img_h
                
                chin_x = landmarks[199].x * img_w
                chin_y = landmarks[199].y * img_h

                eye_mid_y = (landmarks[33].y + landmarks[263].y) / 2 * img_h

                # ヨー (左右の傾き)
                face_center_x = (left_eye_x + right_eye_x) / 2
                yaw = (nose_tip_x - face_center_x) / (img_w / 4) # 正規化と感度調整
                yaw = np.clip(yaw, -1, 1) * (math.pi / 2) * 0.8
                
                # ピッチ (上下の傾き)
                pitch = (eye_mid_y - nose_tip_y) / (nose_tip_y - chin_y + 1e-6)
                pitch = (pitch - 0.3) * -1 # 正規化と感度調整
                pitch = np.clip(pitch, -1, 1) * (math.pi / 2) * 0.8

                # スムージング
                current_pose = {'yaw': yaw, 'pitch': pitch}
                if smoothed_pose is None:
                    smoothed_pose = current_pose
                else:
                    smoothed_pose['yaw'] = SMOOTHING_FACTOR * current_pose['yaw'] + (1 - SMOOTHING_FACTOR) * smoothed_pose['yaw']
                    smoothed_pose['pitch'] = SMOOTHING_FACTOR * current_pose['pitch'] + (1 - SMOOTHING_FACTOR) * smoothed_pose['pitch']

                # --- 3. 視線ベクトルの描画 ---
                # 目の中心を開始点とする
                start_point_x = int((landmarks[168].x * img_w + landmarks[398].x * img_w) / 2)
                start_point_y = int((landmarks[168].y * img_h + landmarks[398].y * img_h) / 2)

                line_length = 150
                # ヨーとピッチから終了点を計算 (カメラ映像は反転しているのでヨーを反転)
                end_point_x = int(start_point_x + line_length * math.sin(smoothed_pose['yaw']))
                end_point_y = int(start_point_y - line_length * math.sin(smoothed_pose['pitch']))
                
                # 視線を描画
                cv2.line(image, (start_point_x, start_point_y), (end_point_x, end_point_y), (233, 165, 14), 3)


        # 結果をウィンドウに表示
        # Webカメラは鏡像になっているので左右反転して表示する
        cv2.imshow('Python Gaze Estimation Demo', cv2.flip(image, 1))
        
        # 'q'キーが押されたらループを抜ける
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# リソースを解放
cap.release()
cv2.destroyAllWindows()