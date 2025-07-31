from rfdetr import RFDETRMedium  # 또는 RFDETRSmall, RFDETRNano
import os
from pathlib import Path

def verify_dataset_structure(dataset_dir):
    """
    데이터셋 구조가 올바른지 확인
    """
    dataset_path = Path(dataset_dir)
    required_dirs = ["train", "valid"]  # RF-DETR은 'valid' 폴더 사용
    
    print(f"데이터셋 구조 확인: {dataset_dir}")
    
    for split in required_dirs:
        split_dir = dataset_path / split
        json_file = split_dir / "_annotations.coco.json"
        
        if not split_dir.exists():
            print(f"❌ 필수 폴더 없음: {split_dir}")
            return False
        
        if not json_file.exists():
            print(f"❌ 어노테이션 파일 없음: {json_file}")
            return False
        
        # 이미지 파일 개수 확인
        image_count = (
            len(list(split_dir.glob("*.jpg"))) + 
            len(list(split_dir.glob("*.jpeg"))) + 
            len(list(split_dir.glob("*.png")))
        )
        
        print(f"✓ {split}: {image_count}개 이미지, 어노테이션 파일 존재")
    
    return True

def train_rfdetr_custom(dataset_dir, output_dir, epochs=50, batch_size=4):
    """
    RF-DETR 커스텀 학습 함수
    
    Args:
        dataset_dir: COCO 형식 데이터셋 디렉터리 경로
        output_dir: 학습 결과가 저장될 디렉터리
        epochs: 학습 에포크 수
        batch_size: 배치 크기 (GPU 메모리에 따라 조정)
    """
    
    # 데이터셋 구조 확인
    if not verify_dataset_structure(dataset_dir):
        print("❌ 데이터셋 구조가 올바르지 않습니다.")
        print("fix_coco_structure.py를 먼저 실행하세요.")
        return False
    
    # 출력 디렉터리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== RF-DETR 커스텀 학습 시작 ===")
    print(f"데이터셋: {dataset_dir}")
    print(f"출력 디렉터리: {output_dir}")
    print(f"에포크: {epochs}")
    print(f"배치 크기: {batch_size}")
    
    try:
        # RF-DETR 모델 로드
        print("모델 로드 중...")
        model = RFDETRMedium()  # 또는 RFDETRSmall(), RFDETRNano()
        
        # 학습 실행
        print("학습 시작...")
        model.train(
            dataset_dir=dataset_dir,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=4,  # 총 배치 크기 = batch_size * grad_accum_steps
            lr=1e-4,             # 학습률
            output_dir=output_dir,
            
            # 선택적 매개변수들
            resolution=640,       # 입력 이미지 해상도 (56의 배수여야 함)
            weight_decay=1e-4,    # 가중치 감쇠
            use_ema=True,         # Exponential Moving Average 사용
            
            # 로깅 설정
            tensorboard=True,     # TensorBoard 로깅
            wandb=False,          # Weights & Biases 로깅 (선택사항)
            
            # 조기 종료 설정
            early_stopping=True,
            early_stopping_patience=10,  # 10 에포크 동안 개선 없으면 중단
            early_stopping_min_delta=0.001,
            
            # 체크포인트 설정
            checkpoint_interval=5,  # 5 에포크마다 체크포인트 저장
        )
        
        print(f"\n🎉 학습 완료! 모델이 {output_dir}에 저장되었습니다.")
        print(f"최종 모델: {output_dir}/checkpoint.pth")
        print(f"EMA 모델: {output_dir}/checkpoint_ema.pth")
        
        return True
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    # 설정
    dataset_dir = "cocoeddata"  # COCO 형식 데이터셋 폴더
    output_dir = "customtrain"  # 학습된 모델 저장 폴더
    
    # GPU 메모리에 따른 배치 크기 조정 가이드:
    # - NVIDIA A100 (40GB): batch_size=16, grad_accum_steps=1
    # - NVIDIA RTX 3090 (24GB): batch_size=8, grad_accum_steps=2  
    # - NVIDIA RTX 3080 (10GB): batch_size=4, grad_accum_steps=4
    # - NVIDIA T4 (16GB): batch_size=4, grad_accum_steps=4
    
    success = train_rfdetr_custom(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=50,
        batch_size=4  # GPU 메모리에 맞게 조정
    )
    
    if success:
        print("\n📊 학습 과정을 모니터링하려면 다음 명령어를 별도 터미널에서 실행하세요:")
        print(f"tensorboard --logdir {output_dir}")
        print("그 후 브라우저에서 http://localhost:6006 으로 접속")
    else:
        print("\n❌ 학습에 실패했습니다. 위의 오류 메시지를 확인하세요.")
