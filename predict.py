"""Inference for the learned folder-label classifier, without manual calibration."""
import hashlib
import json
from pathlib import Path

from extract import ROOT, Extractor
from features import features, VERSION
from binary_policy import decide, DEFAULT_POLICY


def predict_image(path, output, driver_side='right', model_path=None, yolo_model=None, device='auto'):
    import joblib
    from PIL import Image, ImageOps, ImageDraw
    model_path = Path(model_path or ROOT/'artifacts/classifier.joblib')
    if not model_path.is_file():
        raise FileNotFoundError('학습 모델이 없습니다. README.md의 모델 준비 및 재학습 절차를 확인하세요.')
    bundle = joblib.load(model_path)  # Only load your own locally trained model.
    if bundle['feature_version'] != VERSION:
        raise ValueError('특징 버전이 변경되었습니다. python -m training.train_cascade로 재학습하세요.')
    engine = Extractor(driver_side, yolo_model=yolo_model, device=device)
    try:
        observation = engine.extract(path)
    finally:
        engine.close()
    f, info = features(observation, driver_side)
    result = dict(image=str(Path(path).resolve()), driver_side=driver_side,
                  yolo_device=observation['yolo_device'], yolo_sha256=observation['yolo_sha256'], **info)
    probability = None
    if f is not None:
        values = [[f[k] for k in bundle['feature_names']]]
        model = bundle['model']
        probability = float(model.predict_proba(values)[0, list(model.classes_).index(1)])
    first_score = probability
    cascade = bundle.get('cascade')
    stage = 1
    if cascade:
        threshold = cascade['stage1_threshold']
        if probability is None:
            from cascade import ImageFallback
            probability = ImageFallback(cascade, device).score(path, observation, driver_side)
            threshold = cascade['stage2_threshold']
            stage = 2
        policy = dict(threshold=threshold, unscored_class='negative')
    else:
        policy = bundle.get('binary_policy', DEFAULT_POLICY)
    status = decide(probability, policy)
    result.update(status=status, label='들고있음' if status == 'positive' else '안들고있음',
                  positive_score=probability, score_is_calibrated_probability=False,
                  binary_policy=policy, fallback_used=first_score is None,
                  stage_used=stage, stage1_score=first_score,
                  stage2_score=probability if stage == 2 else None,
                  reason=info.get('reason', 'missing_features') if first_score is None else 'landmark_classifier')
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result['present_in_training_dataset'] = digest in bundle.get('training_hashes', [])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output/Path(path).stem
    target.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
    image.thumbnail((1280,1280))
    draw = ImageDraw.Draw(image)
    w,h = image.size
    for i, pose in enumerate(observation['poses']):
        color = 'lime' if i == result.get('driver_index') else 'orange'
        for a,b in ((11,12),(11,13),(13,15),(12,14),(14,16)):
            if min(pose[a][3], pose[b][3])>=0.4:
                draw.line((pose[a][0]*w,pose[a][1]*h,pose[b][0]*w,pose[b][1]*h),fill=color,width=4)
    for phone in observation['boxes']:
        if phone['cls']==67:
            draw.rectangle(tuple(v*d for v,d in zip(phone['box'],[w,h,w,h])),outline='red',width=3)
    if result.get('driver_box'):
        draw.rectangle(tuple(v*d for v,d in zip(result['driver_box'],[w,h,w,h])),outline='magenta',width=3)
    draw.rectangle((0,0,w,35), fill='black')
    caption=result['status']+' '+('score='+str(round(result['positive_score'],3))
                                if result.get('positive_score') is not None else 'fallback: '+result['reason'])
    draw.text((10,10),caption,fill='white')
    image.save(target.with_suffix('.jpg'))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
