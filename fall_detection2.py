
import cv2
import mediapipe as mp
import time

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# 낙상 기준
FALL_WARNING_TIME = 10  # 초

# 상태 변수
fall_timer = 0
fall_start_time = None
fall_confirmed = False

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    fall_detected_now = False

    if result.pose_landmarks:
        h, w, _ = frame.shape
        lm = result.pose_landmarks.landmark

        # 코와 골반 좌표 추출
        nose_y = lm[mp_pose.PoseLandmark.NOSE].y * h
        left_hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP].y * h
        right_hip_y = lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h
        hip_y = (left_hip_y + right_hip_y) / 2

        # 낙상 기준: 머리(코)가 골반보다 아래
        if nose_y >= hip_y:
            fall_detected_now = True

        # 키포인트 시각화
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # 낙상 감지 시간 계산
    if fall_detected_now:
        if fall_start_time is None:
            fall_start_time = time.time()
        fall_timer = time.time() - fall_start_time

        # 경고 메시지
        cv2.putText(frame, "⚠️ FALL WARNING!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)

        # 낙상 확정
        if fall_timer >= FALL_WARNING_TIME:
            fall_confirmed = True
    else:
        fall_start_time = None
        fall_timer = 0
        fall_confirmed = False

    if fall_confirmed:
        cv2.putText(frame, "✅ FALL CONFIRMED!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 0, 0), 3)

    # 화면 출력
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
