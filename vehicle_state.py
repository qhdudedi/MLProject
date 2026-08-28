"""차량 상태 조회 계층.

모델은 9개 피처를 요구하지만, 그중 사용자가 직접 입력해야 하는 것은 목적지까지의
거리 하나뿐이다. 나머지는 차량 센서(OBD-II / CAN / 텔레매틱스 API)나 내비게이션
경로 정보에서 읽어오는 값이다.

이 모듈은 그 '읽어오는 부분'을 한 곳에 모아 둔 것이다. 지금은 실제 차량이 없으므로
학습 데이터 중앙값에 맞춘 모의(mock) 값을 돌려주지만, read_vehicle_state() 하나만
실제 API 호출로 교체하면 나머지 코드는 그대로 동작한다.
"""

from dataclasses import dataclass, field

# 학습 데이터에서 관측된 각 피처의 범위.
# 이 범위를 벗어난 입력은 모델이 외삽(extrapolation)하게 되므로 신뢰할 수 없다.
# 화면의 슬라이더 최소/최대도 이 값에서 나온다.
TRAINING_RANGES = {
    'speed_kmh': (20.0, 130.0),
    'payload_kg': (0.0, 500.0),
    'ambient_temp_C': (-10.0, 40.0),
    'hvac_power_kw': (0.0, 5.0),
    'road_grade_pct': (-5.0, 8.0),
    'battery_temp_C': (15.0, 45.0),
    'driving_style_index': (0.0, 1.0),
    'tire_pressure_bar': (2.0, 2.8),
    'trip_distance_km': (5.1, 200.0),
}

# 실제 차량 연동 전까지 사용할 모의 값. 학습 데이터 중앙값 기준.
_MOCK_STATE = {
    'battery_kwh': 62.0,
    'payload_kg': 246.85,
    'ambient_temp_C': 15.8,
    'hvac_power_kw': 2.48,
    'battery_temp_C': 29.9,
    'tire_pressure_bar': 2.4,
    'driving_style_index': 0.5,
}


@dataclass
class VehicleState:
    """차량에서 읽어온 현재 상태."""

    battery_kwh: float          # 배터리 잔량 (BMS)
    payload_kg: float           # 탑승자 + 화물 하중 (시트 하중 센서 / 추정)
    ambient_temp_C: float       # 외기온 센서
    hvac_power_kw: float        # 공조 시스템 소비 전력
    battery_temp_C: float       # 배터리 온도 (BMS)
    tire_pressure_bar: float    # 타이어 공기압 (TPMS)
    driving_style_index: float  # 최근 주행 이력에서 산출한 운전 성향

    # 각 값이 어디서 왔는지 (UI 에 '차량 센서' / '사용자 입력' 으로 표시하기 위함)
    sources: dict[str, str] = field(default_factory=dict)


def read_vehicle_state(overrides: dict | None = None) -> VehicleState:
    """차량의 현재 상태를 읽어온다.

    overrides 에 값이 있으면 그 항목만 사용자 입력으로 대체한다
    (예: 동승자가 더 타서 하중을 직접 조정하는 경우).

    실제 서비스에서는 이 함수 안에서 차량 텔레매틱스 API를 호출하면 된다.
    반환 타입만 유지하면 호출하는 쪽 코드는 바꿀 필요가 없다.
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}

    values = dict(_MOCK_STATE)
    sources = {k: 'vehicle(mock)' for k in values}

    for key, value in overrides.items():
        if key in values:
            values[key] = value
            sources[key] = 'user'

    return VehicleState(**values, sources=sources)


def estimate_route_grade(override: float | None = None) -> tuple[float, str]:
    """경로의 평균 도로 경사도를 추정한다.

    실제로는 내비게이션 경로의 고도 프로파일에서 계산해야 한다.
    지금은 평지(0%)를 가정한다.

    주의: 현재 모델은 경사도를 '주행 구간 전체의 단일 값'으로 받는다.
    실제 장거리 경로는 오르막과 내리막이 섞여 있으므로, 정확히 하려면
    경로를 구간별로 쪼개 각각 추천을 계산하고 합산해야 한다.
    """
    if override is not None:
        return override, 'user'
    return 0.0, 'route(assumed-flat)'


def check_training_range(feature_values: dict) -> list[str]:
    """학습 데이터 범위를 벗어난 입력을 찾아 경고 문구로 돌려준다.

    범위 밖이라고 예측이 실패하지는 않지만, 모델이 본 적 없는 구간이므로
    결과를 그대로 신뢰하면 안 된다는 것을 사용자에게 알려야 한다.
    """
    warnings = []
    for name, value in feature_values.items():
        if name not in TRAINING_RANGES or value is None:
            continue
        low, high = TRAINING_RANGES[name]
        if not low <= value <= high:
            warnings.append(
                f'{name}={value:g} 은 학습 데이터 범위({low:g} ~ {high:g}) 밖입니다. '
                f'예측 신뢰도가 낮을 수 있습니다.'
            )
    return warnings
