#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
최종 수정된 Labelme to COCO 변환 스크립트
파일명 정규화 문제 완전 해결
"""

import json
import os
import re
import shutil
import random
import hashlib
from pathlib import Path
from PIL import Image
import numpy as np

def safe_normalize_filename(filename):
    """
    파일명을 완전히 안전한 ASCII 문자로 정규화
    """
    # 파일명과 확장자 분리
    name_part = Path(filename).stem
    ext_part = Path(filename).suffix.lower()
    
    # 1단계: 모든 비ASCII 문자를 제거하고 안전한 문자만 유지
    # 한글, 중국어, 일본어, 특수문자 등을 모두 제거
    safe_chars = re.sub(r'[^a-zA-Z0-9_\-]', '_', name_part)
    
    # 2단계: 연속된 언더스코어 제거
    safe_chars = re.sub(r'_+', '_', safe_chars)
    
    # 3단계: 앞뒤 언더스코어 제거
    safe_chars = safe_chars.strip('_')
    
    # 4단계: 빈 문자열이거나 너무 짧은 경우 해시 기반 이름 생성
    if len(safe_chars) < 3:
        # 원본 파일명의 해시값 사용
        hash_obj = hashlib.md5(filename.encode('utf-8'))
        safe_chars = f"img_{hash_obj.hexdigest()[:12]}"
    
    # 5단계: 너무 긴 이름 줄이기 (최대 50자)
    if len(safe_chars) > 50:
        safe_chars = safe_chars[:50]
    
    # 6단계: 확장자 정규화
    if not ext_part:
        ext_part = '.jpg'
    elif ext_part not in ['.jpg', '.jpeg', '.png', '.bmp']:
        ext_part = '.jpg'
    
    return f"{safe_chars}{ext_part}"

def polygon_to_bbox(points):
    """폴리곤을 바운딩 박스로 변환"""
    if not points or len(points) < 3:
        return None
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    
    if width <= 0 or height <= 0:
        return None
    
    return [x_min, y_min, width, height]

def convert_labelme_to_coco_final(
    labelme_folder, 
    export_dir, 
    train_split_rate=0.8
):
    """
    완전히 수정된 Labelme to COCO 변환
    """
    print(f"🔄 최종 Labelme → COCO 변환 시작")
    
    labelme_path = Path(labelme_folder)
    export_path = Path(export_dir)
    
    # 기존 출력 폴더 완전 삭제
    if export_path.exists():
        shutil.rmtree(export_path)
    export_path.mkdir(exist_ok=True)
    
    # Labelme JSON 파일 찾기
    json_files = list(labelme_path.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"Labelme JSON 파일을 찾을 수 없습니다: {labelme_path}")
    
    print(f"📁 발견된 Labelme 파일: {len(json_files)}개")
    
    # 파일명 중복 방지를 위한 카운터
    used_names = set()
    name_counter = {}
    
    # 데이터 수집 및 파일 복사
    all_data = []
    categories = set()
    
    print("📋 데이터 수집 및 파일명 정규화 중...")
    
    for i, json_file in enumerate(json_files):
        if i % 500 == 0:
            print(f"  진행: {i}/{len(json_files)}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)
            
            image_path = labelme_data.get('imagePath', '')
            if not image_path:
                continue
            
            # 원본 이미지 파일 경로
            original_image_path = labelme_path / image_path
            if not original_image_path.exists():
                continue
            
            # 파일명 정규화 (중복 방지)
            base_normalized = safe_normalize_filename(image_path)
            normalized_name = base_normalized
            
            # 중복된 이름인 경우 번호 추가
            if normalized_name in used_names:
                if base_normalized not in name_counter:
                    name_counter[base_normalized] = 1
                else:
                    name_counter[base_normalized] += 1
                
                name_part = Path(base_normalized).stem
                ext_part = Path(base_normalized).suffix
                normalized_name = f"{name_part}_{name_counter[base_normalized]:03d}{ext_part}"
            
            used_names.add(normalized_name)
            
            # 이미지 정보
            image_width = labelme_data.get('imageWidth', 0)
            image_height = labelme_data.get('imageHeight', 0)
            
            # 실제 이미지에서 크기 확인
            if image_width == 0 or image_height == 0:
                try:
                    with Image.open(original_image_path) as img:
                        image_width, image_height = img.size
                except:
                    continue
            
            # 어노테이션 처리
            annotations = []
            for shape in labelme_data.get('shapes', []):
                if shape['shape_type'] not in ['polygon', 'rectangle']:
                    continue
                
                label = shape.get('label', '')
                if not label:
                    continue
                
                categories.add(label)
                points = shape.get('points', [])
                
                # 바운딩 박스 변환
                if shape['shape_type'] == 'rectangle' and len(points) == 2:
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    bbox = [min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1)]
                else:
                    bbox = polygon_to_bbox(points)
                
                if bbox and bbox[2] > 0 and bbox[3] > 0:
                    annotations.append({
                        'label': label,
                        'bbox': bbox,
                        'area': bbox[2] * bbox[3]
                    })
            
            if annotations:
                all_data.append({
                    'original_path': str(original_image_path),
                    'normalized_name': normalized_name,
                    'width': image_width,
                    'height': image_height,
                    'annotations': annotations
                })
        
        except Exception as e:
            print(f"  ⚠️ 파일 처리 실패: {json_file} - {e}")
            continue
    
    if not all_data:
        raise ValueError("변환할 유효한 데이터가 없습니다.")
    
    print(f"✅ 처리된 이미지: {len(all_data)}개")
    print(f"📋 발견된 카테고리: {sorted(categories)}")
    
    # 카테고리 정보 생성 (ID 0부터 시작)
    category_list = []
    for idx, cat_name in enumerate(sorted(categories), 0):
        category_list.append({
            'id': idx,
            'name': cat_name,
            'supercategory': cat_name
        })
    
    # 학습/검증 분할
    random.shuffle(all_data)
    split_idx = int(len(all_data) * train_split_rate)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    print(f"📊 학습 데이터: {len(train_data)}개")
    print(f"📊 검증 데이터: {len(val_data)}개")
    
    # 폴더 생성 및 이미지 복사
    train_dir = export_path / "train"
    valid_dir = export_path / "valid"
    train_dir.mkdir(exist_ok=True)
    valid_dir.mkdir(exist_ok=True)
    
    # COCO 형식 변환 및 이미지 복사
    for split_name, split_data, split_dir in [
        ('train', train_data, train_dir), 
        ('valid', val_data, valid_dir)
    ]:
        print(f"\n📁 {split_name} 데이터 처리 중...")
        
        # COCO 형식 변환 및 이미지 복사
        coco_data = {
            'info': {
                'year': 2025,
                'version': '1.0',
                'description': 'Custom dataset converted from Labelme',
                'contributor': 'RF-DETR Training',
                'url': '',
                'date_created': '2025-01-01'
            },
            'licenses': [
                {
                    'id': 1,
                    'name': 'Custom License',
                    'url': ''
                }
            ],
            'images': [],
            'annotations': [],
            'categories': category_list
        }
        
        annotation_id = 1
        copied_count = 0
        
        for img_idx, data_item in enumerate(split_data, 1):
            # 이미지 복사
            src_path = Path(data_item['original_path'])
            dst_path = split_dir / data_item['normalized_name']
            
            try:
                shutil.copy(str(src_path), str(dst_path))
                copied_count += 1
            except Exception as e:
                print(f"    ❌ 이미지 복사 실패: {src_path} -> {dst_path} ({e})")
                continue
            
            # 이미지 정보
            image_info = {
                'id': img_idx,
                'file_name': data_item['normalized_name'],
                'width': data_item['width'],
                'height': data_item['height']
            }
            coco_data['images'].append(image_info)
            
            # 어노테이션 정보
            for ann in data_item['annotations']:
                category_id = next(cat['id'] for cat in category_list if cat['name'] == ann['label'])
                
                annotation_info = {
                    'id': annotation_id,
                    'image_id': img_idx,
                    'category_id': category_id,
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                    'iscrowd': 0
                }
                
                coco_data['annotations'].append(annotation_info)
                annotation_id += 1
        
        # JSON 파일 저장
        json_output = split_dir / "_annotations.coco.json"
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ {copied_count}개 이미지 복사 완료")
        print(f"  ✅ {len(coco_data['annotations'])}개 어노테이션 생성")
        print(f"  ✅ {json_output} 저장 완료")
    
    return export_dir

def create_test_split_final(coco_dir, test_ratio=0.5):
    """테스트 데이터 분할 및 JSON 동기화"""
    print(f"\n📊 테스트 데이터 분할 (비율: {test_ratio})...")
    coco_path = Path(coco_dir)
    valid_dir = coco_path / "valid"
    test_dir = coco_path / "test"
    test_dir.mkdir(exist_ok=True)

    # valid JSON 로드
    valid_json = valid_dir / "_annotations.coco.json"
    if not valid_json.exists():
        print("  ⚠️ valid JSON 파일이 없습니다.")
        return

    with open(valid_json, 'r', encoding='utf-8') as f:
        valid_data = json.load(f)

    # valid 폴더의 실제 이미지 파일 목록
    valid_image_files = list(valid_dir.glob("*.jpg")) + list(valid_dir.glob("*.png"))
    valid_filenames = {img.name for img in valid_image_files}
    
    print(f"  📁 valid 폴더 실제 이미지: {len(valid_image_files)}개")
    print(f"  📄 valid JSON 이미지 정보: {len(valid_data['images'])}개")

    # JSON에서 실제 존재하는 이미지만 필터링
    existing_images = [img for img in valid_data['images'] if img['file_name'] in valid_filenames]
    existing_image_ids = {img['id'] for img in existing_images}
    existing_annotations = [ann for ann in valid_data['annotations'] if ann['image_id'] in existing_image_ids]
    
    print(f"  🔍 실제 존재하는 이미지: {len(existing_images)}개")

    if not existing_images:
        print("  ⚠️ valid 폴더에 유효한 이미지가 없습니다.")
        return

    # 테스트용 이미지 선택
    num_test = int(len(existing_images) * test_ratio)
    if num_test == 0 and len(existing_images) > 0:
        num_test = 1
    
    # 랜덤하게 테스트 이미지 선택
    test_images_info = random.sample(existing_images, num_test)
    test_filenames = {img['file_name'] for img in test_images_info}
    test_image_ids = {img['id'] for img in test_images_info}
    
    # 남은 valid 이미지 정보
    remaining_valid_images = [img for img in existing_images if img['file_name'] not in test_filenames]
    remaining_valid_ids = {img['id'] for img in remaining_valid_images}
    
    print(f"  📊 테스트로 이동: {len(test_images_info)}개")
    print(f"  📊 valid에 남음: {len(remaining_valid_images)}개")

    # 실제 이미지 파일 이동
    moved_count = 0
    for filename in test_filenames:
        src_path = valid_dir / filename
        dst_path = test_dir / filename
        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            moved_count += 1
    
    print(f"  ✅ {moved_count}개 이미지 파일을 test 폴더로 이동")

    # 어노테이션 분할
    test_annotations = [ann for ann in existing_annotations if ann['image_id'] in test_image_ids]
    remaining_valid_annotations = [ann for ann in existing_annotations if ann['image_id'] in remaining_valid_ids]

    # test JSON 생성
    test_data = {
        "info": {
            "year": 2025,
            "version": "1.0",
            "description": "Test split of custom dataset",
            "contributor": "RF-DETR Training",
            "url": "",
            "date_created": "2025-01-01"
        },
        "licenses": [
            {
                "id": 1,
                "name": "Custom License",
                "url": ""
            }
        ],
        "images": test_images_info,
        "annotations": test_annotations,
        "categories": valid_data['categories']
    }

    test_json = test_dir / "_annotations.coco.json"
    with open(test_json, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ test JSON 생성: {len(test_images_info)}개 이미지, {len(test_annotations)}개 어노테이션")

    # valid JSON 업데이트 (남은 데이터만)
    updated_valid_data = {
        "info": valid_data.get('info', {
            "year": 2025,
            "version": "1.0", 
            "description": "Valid split of custom dataset",
            "contributor": "RF-DETR Training",
            "url": "",
            "date_created": "2025-01-01"
        }),
        "licenses": valid_data.get('licenses', [
            {
                "id": 1,
                "name": "Custom License", 
                "url": ""
            }
        ]),
        "images": remaining_valid_images,
        "annotations": remaining_valid_annotations,
        "categories": valid_data['categories']
    }

    with open(valid_json, 'w', encoding='utf-8') as f:
        json.dump(updated_valid_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ valid JSON 업데이트: {len(remaining_valid_images)}개 이미지, {len(remaining_valid_annotations)}개 어노테이션")

def final_verification(coco_dir):
    """최종 검증"""
    print(f"\n🔍 최종 데이터셋 검증...")
    coco_path = Path(coco_dir)
    
    all_good = True
    
    for split in ["train", "valid", "test"]:
        split_dir = coco_path / split
        json_file = split_dir / "_annotations.coco.json"
        
        if not split_dir.exists():
            if split == "test":
                print(f"  - {split}: 선택사항 (없음)")
                continue
            else:
                print(f"  ❌ {split}: 폴더 없음")
                all_good = False
                continue
        
        if not json_file.exists():
            print(f"  ❌ {split}: JSON 파일 없음")
            all_good = False
            continue
        
        # 실제 이미지 파일 개수
        image_files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
        
        # JSON 정보 확인
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        json_images = len(data.get('images', []))
        json_annotations = len(data.get('annotations', []))
        
        print(f"  ✅ {split}:")
        print(f"     📁 이미지 파일: {len(image_files)}개")
        print(f"     📄 JSON 이미지: {json_images}개")
        print(f"     🏷️ 어노테이션: {json_annotations}개")
        
        if len(image_files) != json_images:
            print(f"     ❌ 이미지 파일 수와 JSON 정보 불일치!")
            all_good = False
        
        # 파일명 검증 (ASCII만 포함되어야 함)
        non_ascii_files = []
        for img_file in image_files:
            try:
                img_file.name.encode('ascii')
            except UnicodeEncodeError:
                non_ascii_files.append(img_file.name)
        
        if non_ascii_files:
            print(f"     ❌ 비ASCII 파일명 발견: {len(non_ascii_files)}개")
            all_good = False
        else:
            print(f"     ✅ 모든 파일명이 ASCII로 정규화됨")
    
    if all_good:
        print(f"\n🎉 데이터셋이 완벽하게 준비되었습니다!")
        print(f"이제 안전하게 학습을 진행할 수 있습니다.")
    else:
        print(f"\n❌ 일부 문제가 있습니다.")
    
    return all_good

def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("🔧 파일명 정규화 문제 완전 해결 버전")
    print("   - 모든 한글/특수문자 → ASCII 변환")
    print("   - 파일 복사와 JSON 정보 완전 동기화")
    print("   - RF-DETR 100% 호환")
    print("=" * 70)
    
    # === 설정 ===
    labelme_folder = "testdataset"
    export_dir = "cocoeddata"
    train_split_rate = 0.8
    test_split_rate = 0.5
    # === 설정 끝 ===
    
    if not Path(labelme_folder).exists():
        print(f"❌ 입력 폴더가 없습니다: {labelme_folder}")
        return False

    try:
        # 1단계: 완전한 변환
        print(f"\n{'='*20} 1단계: 완전한 Labelme → COCO 변환 {'='*20}")
        convert_labelme_to_coco_final(labelme_folder, export_dir, train_split_rate)
        
        # 2단계: 테스트 분할
        print(f"\n{'='*20} 2단계: 테스트 데이터 분할 {'='*20}")
        create_test_split_final(export_dir, test_split_rate)
        
        # 3단계: 최종 검증
        print(f"\n{'='*20} 3단계: 최종 검증 {'='*20}")
        success = final_verification(export_dir)
        
        if success:
            print(f"\n🎉 모든 문제가 해결되었습니다!")
            print(f"이제 학습을 시작하세요:")
            print(f"python3 customtrain.py")
        
        return success
        
    except Exception as e:
        print(f"\n❌ 변환 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
