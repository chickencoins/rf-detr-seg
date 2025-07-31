#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RF-DETR Command Line Interface
Ultralytics YOLO와 유사한 CLI 인터페이스를 제공하는 RF-DETR 도구

사용법:
    python rf-detr.py MODE [OPTIONS]
    
    MODE:
        train       - 모델 학습
        predict     - 추론 실행
        val         - 모델 검증
        convert     - Labelme to COCO 변환
        
    예시:
        python rf-detr.py train --data cocoeddata --epochs 50 --batch-size 4
        python rf-detr.py predict --model customtrain/checkpoint_best_ema.pth --source test.jpg
        python rf-detr.py convert --input testdataset --output cocoeddata
"""

import argparse
import sys
import os
from pathlib import Path
import json
import shutil
from typing import Optional, Union, List

# RF-DETR 관련 import
try:
    from rfdetr import RFDETRMedium, RFDETRSmall, RFDETRNano, RFDETRBase
except ImportError:
    print("❌ RF-DETR 패키지가 설치되지 않았습니다.")
    print("다음 명령어로 설치하세요: pip install rfdetr")
    sys.exit(1)

from PIL import Image
import supervision as sv

# 내장 모듈들 import
from customtrain import train_rfdetr_custom, verify_dataset_structure
from labelmetococo_up import convert_labelme_to_coco_final, create_test_split_final, final_verification


class RFDETRConfig:
    """RF-DETR 설정 클래스"""
    
    # 지원되는 모델 크기
    MODELS = {
        'nano': RFDETRNano,
        'small': RFDETRSmall, 
        'medium': RFDETRMedium,
        'base': RFDETRBase
    }
    
    # 기본 설정
    DEFAULT_EPOCHS = 50
    DEFAULT_BATCH_SIZE = 4
    DEFAULT_LR = 1e-4
    DEFAULT_RESOLUTION = 640
    DEFAULT_CONFIDENCE = 0.5
    DEFAULT_TRAIN_SPLIT = 0.8
    DEFAULT_TEST_SPLIT = 0.5


class RFDETRTrainer:
    """RF-DETR 학습 클래스"""
    
    def __init__(self, config: RFDETRConfig):
        self.config = config
    
    def train(self, 
              data: str,
              model: str = 'medium',
              epochs: int = RFDETRConfig.DEFAULT_EPOCHS,
              batch_size: int = RFDETRConfig.DEFAULT_BATCH_SIZE,
              lr: float = RFDETRConfig.DEFAULT_LR,
              output_dir: str = 'runs/train',
              resume: Optional[str] = None,
              tensorboard: bool = True,
              wandb: bool = False,
              early_stopping: bool = True,
              **kwargs):
        """
        RF-DETR 모델 학습
        
        Args:
            data: 데이터셋 경로 (COCO 형식)
            model: 모델 크기 ('nano', 'small', 'medium', 'base')
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            lr: 학습률
            output_dir: 출력 디렉터리
            resume: 재개할 체크포인트 경로
            tensorboard: TensorBoard 로깅 활성화
            wandb: Weights & Biases 로깅 활성화
            early_stopping: 조기 종료 활성화
        """
        
        print(f"🚀 RF-DETR {model.upper()} 모델 학습 시작")
        print(f"📊 데이터셋: {data}")
        print(f"📁 출력 디렉터리: {output_dir}")
        
        # 데이터셋 구조 확인
        if not verify_dataset_structure(data):
            print("❌ 데이터셋 구조가 올바르지 않습니다.")
            print("convert 명령어로 데이터를 변환하세요.")
            return False
        
        # 출력 디렉터리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 모델 초기화
            if model.lower() not in self.config.MODELS:
                raise ValueError(f"지원되지 않는 모델: {model}. 지원 모델: {list(self.config.MODELS.keys())}")
            
            model_class = self.config.MODELS[model.lower()]
            rf_model = model_class()
            
            if resume:
                print(f"📂 체크포인트에서 재개: {resume}")
            
            # 학습 실행
            rf_model.train(
                dataset_dir=data,
                epochs=epochs,
                batch_size=batch_size,
                grad_accum_steps=max(1, 16 // batch_size),  # 총 배치 크기 16 유지
                lr=lr,
                output_dir=output_dir,
                resolution=self.config.DEFAULT_RESOLUTION,
                weight_decay=1e-4,
                use_ema=True,
                tensorboard=tensorboard,
                wandb=wandb,
                early_stopping=early_stopping,
                early_stopping_patience=10,
                early_stopping_min_delta=0.001,
                checkpoint_interval=5,
                resume=resume,
                **kwargs
            )
            
            print(f"\n🎉 학습 완료!")
            print(f"📁 결과 저장 위치: {output_dir}")
            print(f"🔗 TensorBoard: tensorboard --logdir {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ 학습 중 오류 발생: {e}")
            return False


class RFDETRPredictor:
    """RF-DETR 추론 클래스"""
    
    def __init__(self, config: RFDETRConfig):
        self.config = config
    
    def predict(self,
                source: Union[str, Path],
                model: str = 'medium',
                weights: Optional[str] = None,
                conf: float = RFDETRConfig.DEFAULT_CONFIDENCE,
                save: bool = True,
                save_dir: str = 'runs/predict',
                **kwargs):
        """
        RF-DETR 추론 실행
        
        Args:
            source: 입력 이미지/비디오 경로
            model: 모델 크기 또는 가중치 파일 경로
            weights: 커스텀 가중치 파일 경로
            conf: 신뢰도 임계값
            save: 결과 저장 여부
            save_dir: 저장 디렉터리
        """
        
        print(f"🔍 RF-DETR 추론 시작")
        print(f"📷 입력: {source}")
        print(f"🎯 신뢰도 임계값: {conf}")
        
        try:
            # 모델 로드
            if weights:
                # 커스텀 가중치 사용
                if model.lower() not in self.config.MODELS:
                    model = 'medium'  # 기본값
                model_class = self.config.MODELS[model.lower()]
                rf_model = model_class(pretrain_weights=weights)
                print(f"✅ 커스텀 모델 로드: {weights}")
            else:
                # 사전 훈련된 모델 사용
                if model.lower() not in self.config.MODELS:
                    raise ValueError(f"지원되지 않는 모델: {model}")
                model_class = self.config.MODELS[model.lower()]
                rf_model = model_class()
                print(f"✅ 사전 훈련된 {model.upper()} 모델 로드")
            
            # 이미지 로드 및 전처리
            source_path = Path(source)
            if not source_path.exists():
                raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {source}")
            
            image = Image.open(source_path)
            print(f"📏 이미지 크기: {image.size}")
            
            # RGBA -> RGB 변환
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                else:
                    image = image.convert('RGB')
                print(f"🔄 이미지 모드 변환: RGB")
            
            # 추론 실행
            detections = rf_model.predict(image, threshold=conf)
            print(f"🎯 탐지된 객체: {len(detections)}개")
            
            if len(detections) > 0:
                # 결과 시각화
                labels = [f"object {conf:.2f}" for conf in detections.confidence]
                
                annotated_image = image.copy()
                annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
                annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
                
                if save:
                    # 저장 디렉터리 생성
                    save_path = Path(save_dir)
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    # 결과 저장
                    output_file = save_path / f"result_{source_path.stem}.jpg"
                    annotated_image.save(output_file)
                    print(f"💾 결과 저장: {output_file}")
                
                # 탐지 결과 출력
                print("\n📋 탐지 결과:")
                for i, (label, bbox) in enumerate(zip(labels, detections.xyxy)):
                    print(f"  {i+1}: {label}")
                    print(f"      BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            else:
                print("❌ 탐지된 객체가 없습니다.")
                print(f"💡 신뢰도 임계값을 낮춰보세요 (현재: {conf})")
            
            return True
            
        except Exception as e:
            print(f"❌ 추론 중 오류 발생: {e}")
            return False


class RFDETRValidator:
    """RF-DETR 검증 클래스"""
    
    def __init__(self, config: RFDETRConfig):
        self.config = config
    
    def validate(self,
                 data: str,
                 model: str = 'medium',
                 weights: Optional[str] = None,
                 batch_size: int = 8,
                 **kwargs):
        """
        RF-DETR 모델 검증
        
        Args:
            data: 검증 데이터셋 경로
            model: 모델 크기
            weights: 가중치 파일 경로
            batch_size: 배치 크기
        """
        
        print(f"📊 RF-DETR 모델 검증 시작")
        print(f"📁 데이터셋: {data}")
        
        # 현재 RF-DETR 패키지에서 별도의 검증 함수가 없으므로
        # 기본적인 데이터셋 구조 확인만 수행
        if verify_dataset_structure(data):
            print("✅ 데이터셋 구조가 올바릅니다.")
            return True
        else:
            print("❌ 데이터셋 구조를 확인하세요.")
            return False


class RFDETRConverter:
    """Labelme to COCO 변환 클래스"""
    
    def __init__(self, config: RFDETRConfig):
        self.config = config
    
    def convert(self,
                input_dir: str,
                output_dir: str,
                train_split: float = RFDETRConfig.DEFAULT_TRAIN_SPLIT,
                test_split: float = RFDETRConfig.DEFAULT_TEST_SPLIT,
                **kwargs):
        """
        Labelme 형식을 COCO 형식으로 변환
        
        Args:
            input_dir: Labelme 데이터셋 디렉터리
            output_dir: 출력 디렉터리
            train_split: 학습 데이터 비율
            test_split: 테스트 데이터 비율 (valid에서 분할)
        """
        
        print(f"🔄 Labelme → COCO 변환 시작")
        print(f"📁 입력: {input_dir}")
        print(f"📁 출력: {output_dir}")
        print(f"📊 학습/검증 분할: {train_split:.1%}/{1-train_split:.1%}")
        
        try:
            # 1단계: Labelme to COCO 변환
            convert_labelme_to_coco_final(input_dir, output_dir, train_split)
            
            # 2단계: 테스트 데이터 분할
            if test_split > 0:
                print(f"📊 테스트 데이터 분할: {test_split:.1%}")
                create_test_split_final(output_dir, test_split)
            
            # 3단계: 최종 검증
            success = final_verification(output_dir)
            
            if success:
                print(f"\n🎉 변환 완료!")
                print(f"📁 결과: {output_dir}")
                print(f"🚀 이제 학습을 시작할 수 있습니다:")
                print(f"   python rf-detr.py train --data {output_dir}")
            
            return success
            
        except Exception as e:
            print(f"❌ 변환 중 오류 발생: {e}")
            return False


def print_usage():
    """사용법 출력"""
    print("""
RF-DETR Command Line Interface

사용법:
    python rf-detr.py MODE [OPTIONS]

MODE:
    train       모델 학습
    predict     추론 실행  
    val         모델 검증
    convert     Labelme to COCO 변환
    help        도움말 출력

TRAIN 옵션:
    --data          데이터셋 경로 (필수)
    --model         모델 크기 [nano|small|medium|base] (기본: medium)
    --epochs        학습 에포크 수 (기본: 50)
    --batch-size    배치 크기 (기본: 4)
    --lr            학습률 (기본: 1e-4)
    --output-dir    출력 디렉터리 (기본: runs/train)
    --resume        재개할 체크포인트 경로
    --tensorboard   TensorBoard 로깅 활성화 (기본: True)
    --wandb         W&B 로깅 활성화 (기본: False)

PREDICT 옵션:
    --source        입력 이미지/비디오 경로 (필수)
    --model         모델 크기 [nano|small|medium|base] (기본: medium)
    --weights       커스텀 가중치 파일 경로
    --conf          신뢰도 임계값 (기본: 0.5)
    --save-dir      저장 디렉터리 (기본: runs/predict)

CONVERT 옵션:
    --input         Labelme 데이터셋 디렉터리 (필수)
    --output        출력 디렉터리 (필수)
    --train-split   학습 데이터 비율 (기본: 0.8)
    --test-split    테스트 데이터 비율 (기본: 0.5)

예시:
    # 모델 학습
    python rf-detr.py train --data cocoeddata --epochs 50 --batch-size 4
    
    # 추론 실행
    python rf-detr.py predict --source test.jpg --weights customtrain/checkpoint_best_ema.pth
    
    # 데이터 변환
    python rf-detr.py convert --input testdataset --output cocoeddata
""")


def main():
    """메인 함수"""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    # 첫 번째 인자는 MODE
    mode = sys.argv[1].lower()
    
    if mode == 'help':
        print_usage()
        sys.exit(0)
    
    # ArgumentParser 설정
    parser = argparse.ArgumentParser(description='RF-DETR CLI', add_help=False)
    parser.add_argument('mode', choices=['train', 'predict', 'val', 'convert', 'help'])
    
    # 공통 옵션
    parser.add_argument('--help', '-h', action='store_true', help='도움말 출력')
    
    # TRAIN 모드 옵션
    if mode == 'train':
        parser.add_argument('--data', required=True, help='데이터셋 경로')
        parser.add_argument('--model', default='medium', choices=['nano', 'small', 'medium', 'base'], help='모델 크기')
        parser.add_argument('--epochs', type=int, default=50, help='학습 에포크 수')
        parser.add_argument('--batch-size', type=int, default=4, help='배치 크기')
        parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
        parser.add_argument('--output-dir', default='runs/train', help='출력 디렉터리')
        parser.add_argument('--resume', help='재개할 체크포인트 경로')
        parser.add_argument('--tensorboard', action='store_true', default=True, help='TensorBoard 로깅')
        parser.add_argument('--wandb', action='store_true', help='W&B 로깅')
        parser.add_argument('--early-stopping', action='store_true', default=True, help='조기 종료')
    
    # PREDICT 모드 옵션
    elif mode == 'predict':
        parser.add_argument('--source', required=True, help='입력 이미지/비디오 경로')
        parser.add_argument('--model', default='medium', choices=['nano', 'small', 'medium', 'base'], help='모델 크기')
        parser.add_argument('--weights', help='커스텀 가중치 파일 경로')
        parser.add_argument('--conf', type=float, default=0.5, help='신뢰도 임계값')
        parser.add_argument('--save-dir', default='runs/predict', help='저장 디렉터리')
    
    # VAL 모드 옵션
    elif mode == 'val':
        parser.add_argument('--data', required=True, help='검증 데이터셋 경로')
        parser.add_argument('--model', default='medium', choices=['nano', 'small', 'medium', 'base'], help='모델 크기')
        parser.add_argument('--weights', help='가중치 파일 경로')
        parser.add_argument('--batch-size', type=int, default=8, help='배치 크기')
    
    # CONVERT 모드 옵션
    elif mode == 'convert':
        parser.add_argument('--input', required=True, help='Labelme 데이터셋 디렉터리')
        parser.add_argument('--output', required=True, help='출력 디렉터리')
        parser.add_argument('--train-split', type=float, default=0.8, help='학습 데이터 비율')
        parser.add_argument('--test-split', type=float, default=0.5, help='테스트 데이터 비율')
    
    try:
        args = parser.parse_args()
        
        if args.help:
            print_usage()
            sys.exit(0)
        
        # 설정 및 실행 클래스 초기화
        config = RFDETRConfig()
        
        # 모드별 실행
        if mode == 'train':
            trainer = RFDETRTrainer(config)
            success = trainer.train(
                data=args.data,
                model=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                output_dir=args.output_dir,
                resume=args.resume,
                tensorboard=args.tensorboard,
                wandb=args.wandb,
                early_stopping=args.early_stopping
            )
            
        elif mode == 'predict':
            predictor = RFDETRPredictor(config)
            success = predictor.predict(
                source=args.source,
                model=args.model,
                weights=args.weights,
                conf=args.conf,
                save_dir=args.save_dir
            )
            
        elif mode == 'val':
            validator = RFDETRValidator(config)
            success = validator.validate(
                data=args.data,
                model=args.model,
                weights=args.weights,
                batch_size=args.batch_size
            )
            
        elif mode == 'convert':
            converter = RFDETRConverter(config)
            success = converter.convert(
                input_dir=args.input,
                output_dir=args.output,
                train_split=args.train_split,
                test_split=args.test_split
            )
        
        sys.exit(0 if success else 1)
        
    except SystemExit:
        raise
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
