"""Generate a readable evaluation report and contact sheet of CV errors."""
import csv
import json
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from extract import ROOT


def main():
    report=json.loads((ROOT/'artifacts/evaluation.json').read_text())
    rows=list(csv.DictReader((ROOT/'artifacts/predictions.csv').open(encoding='utf-8-sig')))
    errors=[r for r in rows if r['correct']=='False']
    selected=errors[::max(1,len(errors)//24)][:24]
    sheet=Image.new('RGB',(1440,1080),'white')
    draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(selected):
        with Image.open(row['path']) as source:
            im=ImageOps.fit(ImageOps.exif_transpose(source).convert('RGB'),(235,240))
        x,y=i%6*240,i//6*270
        sheet.paste(im,(x,y))
        draw.text((x,y+242),Path(row['path']).name+' score='+str(round(float(row['probability']),2)),fill='black')
        draw.text((x,y+254),'TRUE='+('1' if row['label']=='보고있음' else '0'),fill='black')
    sheet.save(ROOT/'artifacts/cv_errors.jpg')
    total=report['overall']; selective=report['selective']
    lines=['# 실제 사진 평가', '',
           f"- 전체 파일: {report['total_files']}장 / SHA-256 중복 제거: {report['unique_images']}장",
           f"- 랜드마크 추출 성공: {report['valid_landmarks']}장 / 실패: {report['missing_landmarks']}장",
           '- 같은 IMG 번호의 100단위 묶음 전체를 평가용으로 빼는 교차검증(9개 묶음).',
           '- 오분류를 확인하며 특징을 개선한 **개발 교차검증**입니다. 손대지 않은 최종 테스트 성능이 아닙니다.',
           '- 같은 인물·실내가 여러 묶음에 등장합니다. 새 운전자·실제 차량으로 일반화되는 성능은 별도 검증이 필요합니다.', '',
           '| 평가 | 정확도 | 양성 정밀도 | 양성 재현율 |', '|---|---:|---:|---:|',
           f"| 랜드마크 성공 전체, 점수 0.5 기준 | {total['accuracy']:.1%} | {total['precision']:.1%} | {total['recall']:.1%} |",
           f"| 판단 유보 제외 | {selective['accuracy']:.1%} | {selective['precision']:.1%} | {selective['recall']:.1%} |", '',
           f"판단 유보 제외 결과의 처리율은 전체 고유 사진 대비 **{selective['coverage']:.1%}**입니다. 이 비율과 정확도를 함께 읽어야 합니다.", '',
           '## 묶음별 결과', '', '| IMG 번호 대역 | 사진 수 | 정확도 |', '|---|---:|---:|']
    lines += [f"| {f['group']*100}–{f['group']*100+99} | {f['n']} | {f['accuracy']:.1%} |" for f in report['folds']]
    lines += ['', '## 결과 파일', '',
              '- `artifacts/all_images.csv`: 원본 858장 각각의 교차검증 결과. 중복 사진에는 동일한 예측을 사용합니다.',
              '- `artifacts/predictions.csv`: 중복 제거 후 0.5 기준 예측과 오분류 목록.',
              '- `artifacts/evaluation.json`: 혼동행렬, 전체/묶음별/판단 유보 제외 지표.',
              '- `artifacts/cv_errors.jpg`: 오분류 샘플. TRUE=1은 보고있음, TRUE=0은 안보고있음.',
              '- `artifacts/classifier.joblib`: 평가가 끝난 뒤 전체 유효 데이터로 다시 학습한 실행용 모델.', '',
              '실행용 모델로 학습 사진을 다시 예측한 결과를 위 교차검증 성능과 혼동하지 마세요.', '']
    (ROOT/'artifacts/development_report.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
