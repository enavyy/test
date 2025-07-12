import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 기준 설정
FALL_WARNING_TIME = 3  # 낙상 감지 시간 (더 짧게)
HEAD_PROXIMITY_PX = 100  # 고통 기준 거리 (더 가깝게)
HAND_NEAR_HEAD_DURATION = 1.0  # 고통 지속 시간
WOBBLE_WINDOW_SEC = 2.0  # 비틀거림 관측 시간
WOBBLE_THRESHOLD = 300  # 비틀거림 임계값 (더 민감하게)
WOBBLE_MIN_MOVEMENTS = 3  # 최소 움직임 횟수

# 상태 변수
fall_start_time = None
fall_confirmed = False

hand_near_head_start = None
pain_detected = False
pain_cooldown = 0  # 쿨다운 타이머

# 비틀거림을 위한 더 정교한 추적
wobble_positions = deque(maxlen=20)  # 최근 20개 위치 저장
wobble_detected = False
wobble_cooldown = 0

# 웹캠 열기
cap = cv2.VideoCapture(0)

def calculate_wobble_score(positions):
    """비틀거림 점수 계산"""
    if len(positions) < 3:
        return 0
    
    # 위치 변화의 표준편차와 평균 변화량 계산
    x_positions = [pos[0] for pos in positions]
    y_positions = [pos[1] for pos in positions]
    
    # X축 변화량 (좌우 흔들림)
    x_changes = [abs(x_positions[i] - x_positions[i-1]) for i in range(1, len(x_positions))]
    
    # 급격한 변화의 횟수
    significant_changes = sum(1 for change in x_changes if change > 15)
    
    # 전체 변화량
    total_change = sum(x_changes)
    
    return total_change + (significant_changes * 20)

def is_hand_near_head(hand_pos, head_landmarks, w, h):
    """손이 머리 근처에 있는지 확인 (더 정확한 계산)"""
    # 머리 영역을 더 넓게 정의
    nose = head_landmarks[mp_pose.PoseLandmark.NOSE]
    l_ear = head_landmarks[mp_pose.PoseLandmark.LEFT_EAR]
    r_ear = head_landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
    
    # 머리 중심점과 크기 계산
    head_center_x = (nose.x + l_ear.x + r_ear.x) / 3 * w
    head_center_y = (nose.y + l_ear.y + r_ear.y) / 3 * h
    
    # 머리 크기 기반 동적 거리 계산
    head_width = abs(l_ear.x - r_ear.x) * w
    dynamic_threshold = max(HEAD_PROXIMITY_PX, head_width * 1.5)
    
    hand_x = hand_pos.x * w
    hand_y = hand_pos.y * h
    
    distance = np.sqrt((hand_x - head_center_x)**2 + (hand_y - head_center_y)**2)
    
    return distance < dynamic_threshold, distance

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(frame_rgb)
    current_time = time.time()

    # 쿨다운 타이머 감소
    if pain_cooldown > 0:
        pain_cooldown -= 1
    if wobble_cooldown > 0:
        wobble_cooldown -= 1

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        # 랜드마크 추출
        nose = lm[mp_pose.PoseLandmark.NOSE]
        l_hand = lm[mp_pose.PoseLandmark.LEFT_WRIST]
        r_hand = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # -------------------- 낙상 판단 (개선) --------------------
        nose_y = nose.y * h
        hip_y = (l_hip.y + r_hip.y) / 2 * h
        shoulder_y = (l_shoulder.y + r_shoulder.y) / 2 * h

        # 더 정확한 낙상 판단: 코가 엉덩이보다 아래에 있고, 어깨도 많이 아래에 있을 때
        if nose_y > hip_y and nose_y >= shoulder_y:
            if fall_start_time is None:
                fall_start_time = current_time
            elif current_time - fall_start_time >= FALL_WARNING_TIME:
                fall_confirmed = True
        else:
            fall_start_time = None
            fall_confirmed = False

        # -------------------- 고통신호 판단 (개선) --------------------
        if pain_cooldown == 0:
            l_near_head, l_dist = is_hand_near_head(l_hand, lm, w, h)
            r_near_head, r_dist = is_hand_near_head(r_hand, lm, w, h)

            if l_near_head or r_near_head:
                if hand_near_head_start is None:
                    hand_near_head_start = current_time
                elif current_time - hand_near_head_start >= HAND_NEAR_HEAD_DURATION:
                    pain_detected = True
                    pain_cooldown = 30  # 3초 쿨다운 (30프레임)
            else:
                hand_near_head_start = None
                if pain_cooldown == 0:
                    pain_detected = False
        
        # -------------------- 비틀거림 판단 (개선) --------------------
        if wobble_cooldown == 0:
            # 몸통 중심 계산 (엉덩이 + 어깨 중심)
            body_center_x = (l_hip.x + r_hip.x + l_shoulder.x + r_shoulder.x) / 4 * w
            body_center_y = (l_hip.y + r_hip.y + l_shoulder.y + r_shoulder.y) / 4 * h
            
            wobble_positions.append((body_center_x, body_center_y, current_time))
            
            # 오래된 데이터 제거
            while wobble_positions and current_time - wobble_positions[0][2] > WOBBLE_WINDOW_SEC:
                wobble_positions.popleft()
            
            # 비틀거림 점수 계산
            wobble_score = calculate_wobble_score(wobble_positions)
            
            if wobble_score > WOBBLE_THRESHOLD and len(wobble_positions) >= WOBBLE_MIN_MOVEMENTS:
                wobble_detected = True
                wobble_cooldown = 60  # 6초 쿨다운
            elif wobble_cooldown == 0:
                wobble_detected = False

        # 랜드마크 그리기
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 디버깅용 원 그리기 (손과 머리 거리 시각화)
        if 'l_dist' in locals() and 'r_dist' in locals():
            l_hand_px = (int(l_hand.x * w), int(l_hand.y * h))
            r_hand_px = (int(r_hand.x * w), int(r_hand.y * h))
            
            # 손 위치에 원 그리기
            cv2.circle(frame, l_hand_px, 10, (0, 255, 0), -1)
            cv2.circle(frame, r_hand_px, 10, (0, 255, 0), -1)
            
            # 머리 중심에 원 그리기
            head_center_px = (int(nose.x * w), int(nose.y * h))
            cv2.circle(frame, head_center_px, HEAD_PROXIMITY_PX, (255, 0, 0), 2)

    # -------------------- 시각화 --------------------
    if fall_confirmed:
        cv2.putText(frame, "✅ FALL CONFIRMED!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    elif fall_start_time:
        remaining_time = FALL_WARNING_TIME - (current_time - fall_start_time)
        cv2.putText(frame, f"⚠️ FALL WARNING! {remaining_time:.1f}s", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

    if pain_detected:
        cv2.putText(frame, "😖 PAIN SIGNAL DETECTED!", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)

    if wobble_detected:
        cv2.putText(frame, "🌀 WOBBLING DETECTED!", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)

    # 디버깅 정보 출력
    if 'l_dist' in locals() and 'r_dist' in locals():
        cv2.putText(frame, f"L_Hand: {int(l_dist)} R_Hand: {int(r_dist)}", (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if 'wobble_score' in locals():
        cv2.putText(frame, f"Wobble Score: {int(wobble_score)}", (30, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 100), 1)
    
    cv2.putText(frame, f"Wobble Positions: {len(wobble_positions)}", (30, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 100), 1)

    cv2.imshow("Abnormal Behavior Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
