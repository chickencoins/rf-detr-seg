#!/usr/bin/env python3
"""
RF-DETR 커스텀 모델 추론 스크립트 (수정 완료)
"""

from rfdetr import RFDETRMedium
from PIL import Image
import supervision as sv

def main():
    try:
        print("=== RF-DETR 커스텀 모델 추론 ===")
        
        # 체크포인트 경로
        checkpoint_path = "customtrain/checkpoint_best_ema.pth"
        print(f"체크포인트 로드: {checkpoint_path}")
        
        # 모델 로드 (optimize_for_inference 제거!)
        model = RFDETRMedium(pretrain_weights=checkpoint_path)
        print("모델 로드 완료!")
        
        # 추론 실행
        image_path = "test3.jpg"
        print(f"이미지 로드: {image_path}")
        
        image = Image.open(image_path)
        print(f"원본 이미지 크기: {image.size}, 모드: {image.mode}")
        
        # RGBA -> RGB 변환
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
            print(f"변환된 이미지 모드: {image.mode}")
        
        detections = model.predict(image, threshold=0.5)
        print(f"탐지된 객체: {len(detections)}개")
        
        if len(detections) > 0:
            # 결과 시각화
            labels = [f"tomato {conf:.2f}" for conf in detections.confidence]
            
            annotated_image = image.copy()
            annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
            annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
            annotated_image.save("result_custom.jpg")
            
            # 결과 출력
            print("탐지 결과:")
            for i, (label, bbox) in enumerate(zip(labels, detections.xyxy)):
                print(f"  {i+1}: {label}")
                print(f"      BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            
            print("결과가 result_custom.jpg에 저장되었습니다.")
        else:
            print("탐지된 객체가 없습니다.")
            print("임계값을 낮춰서 다시 시도...")
            detections = model.predict(image, threshold=0.1)
            print(f"낮은 임계값 결과: {len(detections)}개")
            
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
