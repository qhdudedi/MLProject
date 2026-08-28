"""EV 최적 순항 속도 추천 로직.

노트북(energy_consumption.ipynb)과 웹 서비스(app.py)가 같은 코드를 쓰도록 분리한 모듈.
학습 코드는 없고, 저장된 모델을 불러 예측/추천만 수행한다.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).parent / 'model'
MODEL_PATH = MODEL_DIR / 'ev_energy_xgb.pkl'
META_PATH = MODEL_DIR / 'model_meta.json'

# 보고서 기준 모드 프리셋. "최소 전력 대비 N% 까지 더 써도 좋다"는 뜻이다.
MODE_TOLERANCES = {'eco': 0.10, 'balanced': 0.15, 'fast': 0.25}
MODE_LABELS = {'eco': 'ECO', 'balanced': 'BALANCED', 'fast': 'FAST'}


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[Any, dict]:
    """모델과 메타데이터를 로드한다. 프로세스당 한 번만 읽도록 캐시한다.

    모델은 노트북에서 joblib.dump 로 저장한 pickle 이고,
    피처 순서/속도 범위 같은 부가 정보는 옆의 model_meta.json 에 들어 있다.
    """
    for path in (MODEL_PATH, META_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f'모델 파일이 없습니다: {path}\n'
                '노트북(energy_consumption.ipynb)의 "모델 추출" 셀을 먼저 실행하세요.'
            )

    meta = json.loads(META_PATH.read_text(encoding='utf-8'))

    # joblib 은 pickle 기반이라 학습 때와 라이브러리 버전이 다르면 로드가 깨질 수 있다.
    # 그때 원인을 바로 알 수 있도록 메타에 적힌 학습 당시 버전을 함께 알려 준다.
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f'모델 로드 실패: {MODEL_PATH} ({e})\n'
            f"학습 당시 버전: {meta.get('versions', {})}\n"
            '라이브러리 버전을 맞추거나 노트북에서 모델을 다시 추출하세요.'
        ) from e

    return model, meta


def feature_importance() -> dict[str, float]:
    """저장된 모델의 피처 중요도를 비중(%)으로 환산해 돌려준다.

    UI 가 '어떤 항목을 입력받을지'를 이 값으로 결정하기 때문에,
    상수로 박아 두지 않고 모델에서 직접 읽는다.
    모델을 다시 추출하면 화면도 따라서 바뀐다.
    """
    model, meta = load_artifacts()
    raw = np.asarray(model.feature_importances_, dtype=float)
    total = raw.sum()
    if total <= 0:  # 이론상 나오기 어렵지만 0으로 나누는 것은 막는다
        return {name: 0.0 for name in meta['feature_order']}
    return {
        name: float(value / total * 100)
        for name, value in zip(meta['feature_order'], raw)
    }


def predict_speed_profile(
    distance: float,
    road_grade: float,
    payload: float,
    ambient_temp: float,
    hvac_power: float,
    battery_temp: float,
    driving_style: float,
    tire_pressure: float,
) -> pd.DataFrame:
    """후보 속도별 전비 / 총 소비 전력량 / 주행 시간을 계산한다.

    노트북의 predict_speed_profile 과 같은 결과를 내되,
    속도마다 predict 를 호출하는 대신 후보 전체를 한 번에 예측한다(서빙 지연 감소).
    """
    if distance <= 0:
        raise ValueError('distance 는 0보다 커야 합니다.')

    model, meta = load_artifacts()
    low, high = meta['speed_range']
    speeds = np.arange(low, high + meta['speed_step'], meta['speed_step'])

    X_input = pd.DataFrame({
        'speed_kmh': speeds,
        'payload_kg': payload,
        'ambient_temp_C': ambient_temp,
        'hvac_power_kw': hvac_power,
        'road_grade_pct': road_grade,
        'battery_temp_C': battery_temp,
        'driving_style_index': driving_style,
        'tire_pressure_bar': tire_pressure,
        'trip_distance_km': distance,
    })[meta['feature_order']]  # 학습 때와 동일한 컬럼 순서로 정렬

    consumption = model.predict(X_input)  # kWh / 100km

    return pd.DataFrame({
        'speed': speeds,
        'consumption_per_100km': consumption,
        'total_energy': consumption * distance / 100,
        'travel_time': distance / speeds,
    })


def select_mode_speed(
    result: pd.DataFrame,
    energy_tolerance: float,
    baseline: float,
) -> pd.Series:
    """기준 전력 대비 energy_tolerance 만큼 더 쓰는 것까지 허용했을 때의 최고 속도.

    energy_tolerance = 0.10 이면 "전력을 10% 까지 더 써도 좋다"는 뜻이다.
    baseline 은 그 기준이 되는 최소 소비 전력이다.
    """
    if energy_tolerance < 0:
        raise ValueError('energy_tolerance 는 0 이상이어야 합니다.')

    allowed = result[result['total_energy'] <= baseline * (1 + energy_tolerance)]
    return allowed.loc[allowed['speed'].idxmax()]


def recommend_modes(
    distance: float,
    road_grade: float,
    payload: float,
    ambient_temp: float,
    hvac_power: float,
    battery_temp: float,
    driving_style: float,
    tire_pressure: float,
    battery_kwh: float | None = None,
) -> dict:
    """에코 / 일반 / 쾌속 세 모드를 한 번에 계산한다 (화면의 카드 3장).

    기준점은 전체 속도 범위(20~130)의 최소 소비 전력이다.
    노트북과 보고서가 이 기준으로 59 / 86 / 122 km/h 를 냈으므로 동일하게 맞춘다.

    battery_kwh 를 주면 각 모드가 실제로 도달 가능한지만 표시한다.
    배터리는 '얼마나 쓰고 싶은가(모드)'가 아니라 '얼마나 쓸 수 있는가(제약)'이므로
    모드 선택을 대체하지 않고 안전 가드로만 쓴다.
    """
    profile = predict_speed_profile(
        distance=distance,
        road_grade=road_grade,
        payload=payload,
        ambient_temp=ambient_temp,
        hvac_power=hvac_power,
        battery_temp=battery_temp,
        driving_style=driving_style,
        tire_pressure=tire_pressure,
    )

    baseline = float(profile['total_energy'].min())
    baseline_speed = float(profile.loc[profile['total_energy'].idxmin(), 'speed'])

    modes = []
    for key, tolerance in MODE_TOLERANCES.items():
        row = select_mode_speed(profile, tolerance, baseline)
        total = float(row['total_energy'])
        modes.append({
            'key': key,
            'label': MODE_LABELS[key],
            'energy_tolerance': tolerance,
            'recommended_speed_kmh': float(row['speed']),
            'consumption_per_100km': float(row['consumption_per_100km']),
            'total_energy_kwh': total,
            'travel_time_min': float(row['travel_time'] * 60),
            'actual_extra_energy_pct': (total / baseline - 1) * 100,
            # 배터리 안전 가드 : 이 모드로 갔을 때 목적지에 닿는가
            'reachable': None if battery_kwh is None else bool(total <= battery_kwh),
        })

    return {
        'modes': modes,
        'baseline_min_energy_kwh': baseline,
        'baseline_speed_kmh': baseline_speed,
        'battery_kwh': battery_kwh,
        # 속도-에너지 프로파일 그래프용
        'profile': {
            'speed': [float(v) for v in profile['speed']],
            'total_energy': [float(v) for v in profile['total_energy']],
            'consumption_per_100km': [float(v) for v in profile['consumption_per_100km']],
        },
    }
