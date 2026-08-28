"""EV 최적 순항 속도 추천 웹 서비스.

화면 구성
  /      주행 조건 슬라이더 + 모드별 비교 표 + 속도-전력/시간 곡선

추천 기준은 노트북과 동일하다. 전체 속도 범위(20~130)에서 구한 최소 소비 전력을
기준점으로 삼고, 거기서 10 / 15 / 25 % 까지 더 쓰는 것을 허용했을 때의 최고 속도를 낸다.

실행:
    uv run uvicorn app:app --reload
    http://localhost:8000        입력 화면
    http://localhost:8000/docs   자동 생성 API 문서
"""

import math
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import ev_recommender as rec
import vehicle_state as vs

STATIC = Path(__file__).parent / 'static'


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 첫 요청이 아니라 서버 기동 시점에 모델을 올려둔다.
    # 여기서 실패하면 잘못된 모델로 서비스가 뜨는 대신 바로 죽는 편이 낫다.
    _, meta = rec.load_artifacts()
    app.state.meta = meta
    print(f"[startup] 모델 로드 완료 : {meta['model_file']} "
          f"(R2={meta['metrics']['r2']:.4f}, 학습 {meta['created_at']})")
    yield


app = FastAPI(
    title='EV 최적 순항 속도 추천 API',
    description='주행 조건을 입력하면 에코 / 일반 / 쾌속 모드별 최적 순항 속도를 추천합니다.',
    version='0.4.0',
    lifespan=lifespan,
)


class RecommendRequest(BaseModel):
    """반드시 필요한 값은 distance 하나뿐이다.

    나머지는 모두 선택 항목이며, 비워 두면 차량에서 읽은 값(학습 데이터 중앙값)을 쓴다.
    """

    distance: float = Field(..., gt=0, le=1000, description='목적지까지 남은 거리 (km)')

    road_grade_pct: float | None = Field(None, ge=-15, le=15, description='도로 경사도 (%)')
    payload_kg: float | None = Field(None, ge=0, le=500, description='탑승자 및 화물 하중 (kg)')
    hvac_power_kw: float | None = Field(None, ge=0, le=10, description='냉난방 전력 (kW)')
    driving_style_index: float | None = Field(None, ge=0, le=1, description='운전 성향 (0~1)')
    ambient_temp_C: float | None = Field(None, ge=-30, le=50, description='외부 기온 (C)')
    tire_pressure_bar: float | None = Field(None, ge=1.5, le=4.0, description='타이어 공기압 (bar)')
    battery_temp_C: float | None = Field(None, ge=-30, le=60, description='배터리 온도 (C)')
    battery_kwh: float | None = Field(None, gt=0, le=200, description='남은 배터리 (kWh)')


# 슬라이더로 노출할 주행 조건.
# speed_kmh 는 '추천 대상'이라 입력이 아니고, trip_distance_km 는 별도로 다룬다.
SLIDER_SPECS = {
    'road_grade_pct':      {'label': 'Road Grade',    'ko': '도로 경사도',   'unit': '%',   'step': 0.1,  'low': '내리막',  'high': '오르막'},
    'payload_kg':          {'label': 'Payload',       'ko': '적재 하중',     'unit': 'kg',  'step': 10,   'low': '혼자',    'high': '만차 + 짐'},
    'hvac_power_kw':       {'label': 'HVAC Power',    'ko': '냉난방 전력',   'unit': 'kW',  'step': 0.1,  'low': '끔',      'high': '최대'},
    'driving_style_index': {'label': 'Driving Style', 'ko': '운전 성향',     'unit': '',    'step': 0.05, 'low': '정속',    'high': '급가속'},
    'ambient_temp_C':      {'label': 'Ambient Temp',  'ko': '외부 기온',     'unit': 'C',   'step': 1,    'low': '한파',    'high': '폭염'},
    'tire_pressure_bar':   {'label': 'Tire Pressure', 'ko': '타이어 공기압', 'unit': 'bar', 'step': 0.05, 'low': '낮음',    'high': '높음'},
    'battery_temp_C':      {'label': 'Battery Temp',  'ko': '배터리 온도',   'unit': 'C',   'step': 1,    'low': '차가움',  'high': '뜨거움'},
}

# 이 중요도 이상이면 첫 화면에 바로 노출하고, 나머지는 Advanced Settings 로 접어 둔다.
PRIMARY_IMPORTANCE_PCT = 10.0


@app.get('/health')
def health():
    """모델이 정상적으로 올라와 있는지 확인 (배포 헬스체크용)."""
    meta = app.state.meta
    return {
        'status': 'ok',
        'model_file': meta['model_file'],
        'r2': meta['metrics']['r2'],
        'features': meta['feature_order'],
        'created_at': meta['created_at'],
    }


@app.get('/vehicle')
def vehicle():
    """차량에서 현재 읽어온 값."""
    state = vs.read_vehicle_state()
    return {'state': state.__dict__ | {'sources': state.sources}}


@app.get('/inputs')
def inputs():
    """화면에 그릴 슬라이더 구성을 돌려준다.

    어떤 항목을 위로 올릴지는 모델의 피처 중요도가 정하고,
    슬라이더가 움직일 수 있는 범위는 학습 데이터 구간이 정한다.
    범위를 학습 구간에 묶어 두면 사용자가 외삽 영역으로 갈 수가 없다.
    """
    importance = rec.feature_importance()
    current = vs.read_vehicle_state().__dict__

    primary, advanced = [], []
    for key, spec in SLIDER_SPECS.items():
        pct = importance.get(key, 0.0)
        low, high = vs.TRAINING_RANGES[key]
        item = {
            'key': key,
            'label': spec['label'],
            'ko': spec['ko'],
            'unit': spec['unit'],
            'min': low,
            'max': high,
            'step': spec['step'],
            # 차량에서 읽은 현재값(= 학습 데이터 중앙값)에서 출발한다.
            # 경사도만은 센서가 아니라 경로 정보라 평지를 가정한다.
            'value': current.get(key, vs.estimate_route_grade()[0]),
            'importance_pct': round(pct, 1),
            'hint_low': spec['low'],
            'hint_high': spec['high'],
        }
        (primary if pct >= PRIMARY_IMPORTANCE_PCT else advanced).append(item)

    primary.sort(key=lambda i: i['importance_pct'], reverse=True)
    advanced.sort(key=lambda i: i['importance_pct'], reverse=True)

    dist_low, dist_high = vs.TRAINING_RANGES['trip_distance_km']
    # 반올림하면 슬라이더 양 끝이 학습 구간 밖으로 나가 자기 자신이 경고를 띄운다.
    # 안쪽으로 잘라서 슬라이더가 학습 구간을 벗어날 수 없게 한다.
    dmin, dmax = math.ceil(dist_low), math.floor(dist_high)

    return {
        'distance': {
            'key': 'distance', 'label': 'Trip Distance', 'ko': '남은 거리', 'unit': 'km',
            'min': dmin, 'max': dmax, 'step': 1, 'value': 100,
            'importance_pct': round(importance.get('trip_distance_km', 0.0), 1),
            'hint_low': f'{dmin} km', 'hint_high': f'{dmax} km',
            # 보고서의 324km 시나리오처럼 구간 밖까지 슬라이더를 여는 화면이
            # '여기부터 외삽'임을 표시할 수 있어야 한다.
            'training_max': dmax,
        },
        'primary': primary,
        'advanced': advanced + [{
            'key': 'battery_kwh', 'label': 'Battery Remaining', 'ko': '배터리 잔량',
            'unit': 'kWh', 'min': 5, 'max': 100, 'step': 1,
            'value': current['battery_kwh'],
            'importance_pct': 0.0,   # 모델 피처가 아니라 도달 가능 판정에만 쓴다
            'hint_low': '5 kWh', 'hint_high': '100 kWh',
        }],
        'modes': [
            {'key': k, 'label': rec.MODE_LABELS[k], 'tolerance': t}
            for k, t in rec.MODE_TOLERANCES.items()
        ],
    }


@app.post('/recommend')
def recommend(req: RecommendRequest):
    """주행 조건을 받아 에코 / 일반 / 쾌속 모드별 추천 속도를 돌려준다."""
    # 1. 차량 상태를 읽고, 요청에 들어온 항목만 덮어쓴다
    state = vs.read_vehicle_state({
        'battery_kwh': req.battery_kwh,
        'payload_kg': req.payload_kg,
        'ambient_temp_C': req.ambient_temp_C,
        'hvac_power_kw': req.hvac_power_kw,
        'battery_temp_C': req.battery_temp_C,
        'tire_pressure_bar': req.tire_pressure_bar,
        'driving_style_index': req.driving_style_index,
    })

    # 2. 경사도는 경로 정보에서 (요청에 있으면 그 값)
    road_grade, grade_source = vs.estimate_route_grade(req.road_grade_pct)

    # 3. 학습 범위를 벗어난 입력이 있으면 경고를 모아 둔다 (예측은 그대로 수행)
    warnings = vs.check_training_range({
        'trip_distance_km': req.distance,
        'road_grade_pct': road_grade,
        'payload_kg': state.payload_kg,
        'ambient_temp_C': state.ambient_temp_C,
        'hvac_power_kw': state.hvac_power_kw,
        'battery_temp_C': state.battery_temp_C,
        'tire_pressure_bar': state.tire_pressure_bar,
        'driving_style_index': state.driving_style_index,
    })

    try:
        result = rec.recommend_modes(
            distance=req.distance,
            road_grade=road_grade,
            payload=state.payload_kg,
            ambient_temp=state.ambient_temp_C,
            hvac_power=state.hvac_power_kw,
            battery_temp=state.battery_temp_C,
            driving_style=state.driving_style_index,
            tire_pressure=state.tire_pressure_bar,
            battery_kwh=state.battery_kwh,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 4. 배터리로 갈 수 없는 모드가 있으면 세이프티 가드 문구를 얹는다.
    #    전부 불가능한 경우와 일부만 불가능한 경우는 해야 할 일이 다르다.
    blocked = [m['label'] for m in result['modes'] if m['reachable'] is False]
    if blocked and len(blocked) == len(result['modes']):
        need = min(m['total_energy_kwh'] for m in result['modes'])
        warnings.append(
            f'배터리 {state.battery_kwh:.1f} kWh 로는 어떤 모드로도 목적지에 도달할 수 없습니다. '
            f"가장 아끼는 {result['modes'][0]['label']} 모드에도 {need:.1f} kWh 가 필요합니다. "
            f'중간 충전을 반드시 경유하세요.'
        )
    elif blocked:
        ok = [m['label'] for m in result['modes'] if m['reachable']]
        warnings.append(
            f"배터리 {state.battery_kwh:.1f} kWh 로는 {', '.join(blocked)} 모드에 도달할 수 없습니다. "
            f"{', '.join(ok)} 모드를 사용하세요."
        )

    # 어떤 값이 어디서 왔는지 함께 돌려줘야 사용자가 결과를 납득할 수 있다
    result['inputs'] = {
        'trip_distance_km': req.distance,
        'battery_kwh': state.battery_kwh,
        'road_grade_pct': road_grade,
        'payload_kg': state.payload_kg,
        'ambient_temp_C': state.ambient_temp_C,
        'hvac_power_kw': state.hvac_power_kw,
        'battery_temp_C': state.battery_temp_C,
        'tire_pressure_bar': state.tire_pressure_bar,
        'driving_style_index': state.driving_style_index,
    }
    result['sources'] = state.sources | {
        'trip_distance_km': 'user',
        'road_grade_pct': grade_source,
    }
    result['warnings'] = warnings
    return result


def page(name: str) -> HTMLResponse:
    """static/ 의 화면 파일을 그대로 내려 준다.

    HTML 을 파이썬 문자열이 아니라 파일로 두는 이유는, 문자열로 두면
    JS 안의 \\n 을 파이썬이 먼저 해석해 스크립트가 통째로 깨지기 때문이다.
    """
    return HTMLResponse((STATIC / name).read_text(encoding='utf-8'))


@app.get('/', response_class=HTMLResponse)
def index():
    return page('index.html')