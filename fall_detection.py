import cv2
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
import numpy as np

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
model = keypointrcnn_resnet50_fpn(weights=weights)
model.to(device)
model.eval()

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 낙상 판단 함수
def is_fallen(keypoints):
    # keypoints: [17, 3] -> [x, y, confidence]
    if keypoints is None or len(keypoints) != 17:
        return False

    # 중요 포인트
    nose = keypoints[0]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    # 평균 y 위치
    torso_y = np.mean([left_shoulder[1], right_shoulder[1], left_hip[1], right_hip[1]])
    nose_y = nose[1]

    # 머리가 골반보다 아래 있으면 낙상으로 추정
    if nose_y > torso_y + 30:
        return True
    return False

# 영상 처리 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # RGB 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = T.ToTensor()
    image_tensor = transform(image).to(device).unsqueeze(0)

    # 추론
    with torch.no_grad():
        output = model(image_tensor)

    keypoints = output[0]['keypoints']
    scores = output[0]['scores']

    for i in range(len(scores)):
        if scores[i] > 0.9:
            person_kps = keypoints[i].cpu().numpy()
            fall_detected = is_fallen(person_kps)

            for x, y, c in person_kps:
                if c > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

            if fall_detected:
                cv2.putText(frame, "FALL DETECTED!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)
            else:
                cv2.putText(frame, "Normal", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
