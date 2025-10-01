import cv2
import dlib
import numpy as np

# --- 初期設定 ---

# Dlibの顔検出器とランドマーク予測器を初期化
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat" # 事前にダウンロードしたモデルファイルのパス
predictor = dlib.shape_predictor(predictor_path)

# カメラを初期化
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("エラー: カメラを開けませんでした。")
    exit()

# 結果表示用のウィンドウを作成
cv2.namedWindow("Gaze Point", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze Point", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
screen_w, screen_h = 1920, 1080 # ご自身のスクリーン解像度に合わせてください

# 複数人検出時の色分け用
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]


def get_gaze_point(frame, landmarks):
    """
    顔ランドマークから視線のスクリーン座標を推定する関数
    """
    # 1. 3Dモデル上の顔の基準点
    #    これは一般的な顔モデルの座標。調整は不要。
    model_points = np.array([
        (0.0, 0.0, 0.0),             # 鼻先 (30)
        (0.0, -330.0, -65.0),        # 顎 (8)
        (-225.0, 170.0, -135.0),     # 左目尻 (36)
        (225.0, 170.0, -135.0),      # 右目尻 (45)
        (-150.0, -150.0, -125.0),    # 左口角 (48)
        (150.0, -150.0, -125.0)      # 右口角 (54)
    ])

    # 2. カメラの内部パラメータ（重要：キャリブレーションが必要）
    #    ここでは一般的なWebカメラの値を仮定している
    height, width, _ = frame.shape
    focal_length = width
    camera_center = (width / 2, height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, camera_center[0]],
         [0, focal_length, camera_center[1]],
         [0, 0, 1]], dtype="double"
    )
    # レンズ歪みはないと仮定
    dist_coeffs = np.zeros((4, 1))

    # 3. 2D画像上の対応するランドマーク点
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),   # 鼻先
        (landmarks.part(8).x, landmarks.part(8).y),     # 顎
        (landmarks.part(36).x, landmarks.part(36).y),   # 左目尻
        (landmarks.part(45).x, landmarks.part(45).y),   # 右目尻
        (landmarks.part(48).x, landmarks.part(48).y),   # 左口角
        (landmarks.part(54).x, landmarks.part(54).y)    # 右口角
    ], dtype="double")

    # 4. 頭部姿勢を推定 (SolvePnP)
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # 5. 視線ベクトルの計算（簡易版）
    #    ここでは「頭の向き＝視線の向き」と単純化している
    #    3D空間での視線の終点を計算
    (gaze_end_point_3d, _) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]), # カメラ座標系でZ軸方向に1000mm先の点
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeffs
    )

    # 6. 目の中心位置を取得
    left_eye_center = (
        (landmarks.part(36).x + landmarks.part(39).x) // 2,
        (landmarks.part(36).y + landmarks.part(39).y) // 2
    )
    right_eye_center = (
        (landmarks.part(42).x + landmarks.part(45).x) // 2,
        (landmarks.part(42).y + landmarks.part(45).y) // 2
    )
    eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2, 
                   (left_eye_center[1] + right_eye_center[1]) // 2)

    # 7. スクリーン座標へのマッピング（簡易版）
    #    カメラ画像上の視点と視線ベクトルからスクリーン上の位置を推定
    p1 = eyes_center
    p2 = (int(gaze_end_point_3d[0][0][0]), int(gaze_end_point_3d[0][0][1]))

    # カメラの画角とスクリーンの大きさから比率を計算してマッピング
    # これは非常に単純化したモデルであり、精度は低い
    gaze_x = p1[0] + (p2[0] - p1[0]) * 3.0 # 倍率を調整して感度を変更
    gaze_y = p1[1] + (p2[1] - p1[1]) * 3.0

    # スクリーン座標に変換
    screen_x = int(gaze_x * screen_w / width)
    screen_y = int(gaze_y * screen_h / height)
    
    # 画面外に出ないようにクリッピング
    screen_x = max(0, min(screen_w - 1, screen_x))
    screen_y = max(0, min(screen_h - 1, screen_y))

    return (screen_x, screen_y)


# --- メインループ ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 処理負荷を減らすために画像をグレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔を検出
    faces = detector(gray)

    # 結果表示用の黒い画面を作成
    gaze_display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    # 検出された各顔に対して処理
    for i, face in enumerate(faces):
        # ランドマークを検出
        landmarks = predictor(gray, face)

        # 視点を推定
        gaze_point = get_gaze_point(frame, landmarks)
        
        # 視点に円を描画（人物ごとに色を変える）
        color = colors[i % len(colors)]
        cv2.circle(gaze_display, gaze_point, 30, color, -1)
        cv2.putText(gaze_display, f"Person {i+1}", (gaze_point[0] + 35, gaze_point[1] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    # 元のカメラ映像も表示（デバッグ用）
    # for face in faces:
    #     x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("Camera Feed", frame)

    # 視点表示ウィンドウを更新
    cv2.imshow("Gaze Point", gaze_display)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 終了処理 ---
cap.release()
cv2.destroyAllWindows()