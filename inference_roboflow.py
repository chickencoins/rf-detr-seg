import os
import supervision as sv
from inference import get_model
from PIL import Image

# 이미지 로드
image_path = "test2.png"
image = Image.open(image_path)

# RF-DETR 모델 로드 (최신 Medium 모델 사용)
model = get_model("rfdetr-medium")

# 추론 실행
predictions = model.infer(image, confidence=0.5)[0]

# 결과를 supervision 형식으로 변환
detections = sv.Detections.from_inference(predictions)

# 라벨 생성
labels = [prediction.class_name for prediction in predictions.predictions]

# 결과 시각화
annotated_image = image.copy()
annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)

# 결과 저장 (RGB로 변환 후 저장)
annotated_image_rgb = annotated_image.convert('RGB')
annotated_image_rgb.save("result_inference.jpg")
print("탐지 완료! 결과가 result_inference.jpg에 저장되었습니다.")
