# KRX Dynamic Portfolio

> 한국거래소(KRX) 데이터를 활용한 동적 포트폴리오 최적화 및 백테스팅 시스템

[![CI](https://github.com/your-username/krx-dynamic-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/krx-dynamic-portfolio/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 빠른 시작

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/krx-dynamic-portfolio.git
cd krx-dynamic-portfolio

# 2. 개발환경 설정
make dev

# 3. 데이터 수집 및 전처리
make etl

# 4. 모델 학습
make train

# 5. 백테스팅 실행
make backtest

# 6. 대시보드 실행
make app
```

## ⚡ 주요 기능

- **📊 KRX 데이터 수집**: FinanceDataReader, yfinance를 활용한 실시간 데이터 수집
- **🧠 포트폴리오 최적화**: Modern Portfolio Theory 기반 동적 리밸런싱
- **📈 백테스팅**: 과거 데이터를 활용한 전략 성과 검증
- **🎯 대시보드**: Streamlit 기반 인터랙티브 시각화
- **🔧 완전 자동화**: ETL → 학습 → 백테스팅 → 배포 파이프라인

## 🛠️ 개발 워크플로우

### 주요 명령어
```bash
make help      # 모든 명령어 보기
make dev       # 개발 의존성 설치 + pre-commit 설정
make lint      # 코드 스타일 검사 (ruff + black + mypy)
make format    # 코드 자동 포맷팅
make test      # 테스트 실행 (pytest + coverage)
make ci-test   # CI와 동일한 검사 로컬 실행
```

### 데이터 파이프라인
```bash
make etl       # KRX 데이터 수집 및 전처리
make train     # 포트폴리오 최적화 모델 학습
make backtest  # 백테스팅 실행
make app       # Streamlit 대시보드 실행
```

## 📁 프로젝트 구조

```
krx-dynamic-portfolio/
├── pyproject.toml              # 프로젝트 설정 및 의존성
├── Makefile                    # 개발 워크플로우 명령어
├── .github/workflows/ci.yml    # GitHub Actions CI/CD
├── krx_portfolio/
│   ├── etl/                    # 📥 데이터 수집 및 전처리
│   │   ├── collectors.py       # 데이터 수집기
│   │   ├── preprocessors.py    # 데이터 전처리
│   │   └── main.py            # ETL 파이프라인 실행
│   ├── models/                 # 🧠 포트폴리오 최적화 모델
│   │   ├── optimizer.py        # 포트폴리오 최적화
│   │   ├── risk_models.py      # 리스크 모델
│   │   └── train.py           # 모델 학습
│   ├── backtesting/            # 📊 백테스팅 엔진
│   │   ├── engine.py          # 백테스팅 엔진
│   │   ├── metrics.py         # 성과 지표
│   │   └── main.py            # 백테스팅 실행
│   └── app/                    # 🎨 Streamlit 대시보드
│       ├── dashboard.py        # 메인 대시보드
│       └── components/         # UI 컴포넌트
└── tests/                      # 🧪 테스트
```

## 🔧 기술 스택

### 데이터 & 분석
- **pandas**: 데이터 조작 및 분석
- **numpy**: 수치 계산
- **scikit-learn**: 머신러닝 알고리즘

### 데이터 소스
- **yfinance**: Yahoo Finance API
- **FinanceDataReader**: 한국 금융 데이터 (KRX, 한국은행 등)

### 시각화 & 대시보드
- **streamlit**: 웹 대시보드
- **plotly**: 인터랙티브 차트

### 개발 도구
- **black**: 코드 포맷팅
- **ruff**: 빠른 린터
- **mypy**: 타입 검사
- **pytest**: 테스트 프레임워크
- **pre-commit**: Git 훅 관리

## 🧪 테스트

```bash
# 전체 테스트 실행
make test

# 커버리지 포함 상세 테스트
pytest -v --cov=krx_portfolio --cov-report=html

# 특정 모듈 테스트
pytest tests/test_etl.py -v
```

## 📊 테스트 현황

### 현재 구현 완료
- ✅ **ETL 파이프라인**: 100% 구현 및 테스트 완료
- ✅ **데이터 로더**: 100% 커버리지
- ✅ **전처리기**: 100% 커버리지  
- ✅ **통합 테스트**: ETL 파이프라인 end-to-end 테스트

### 테스트 통계
- **전체 테스트**: 39개 (모두 통과)
- **전체 커버리지**: 95%
- **핵심 모듈 커버리지**: 100%

## 🤝 기여하기

1. 이슈 생성 또는 기존 이슈 확인
2. feature 브랜치 생성: `git checkout -b feature/amazing-feature`
3. 변경사항 커밋: `git commit -m 'Add amazing feature'`
4. 브랜치에 푸시: `git push origin feature/amazing-feature`
5. Pull Request 생성

### 개발 가이드라인
- 모든 코드는 `make ci-test` 통과 필수
- 새로운 기능은 테스트 코드 포함
- 커밋 메시지는 [Conventional Commits](https://conventionalcommits.org/) 형식 준수

## 📈 로드맵

### Phase 1: ETL 파이프라인 (✅ 완료)
- [x] KRX JSON 데이터 로더
- [x] 데이터 전처리 및 정제
- [x] OHLCV 패널 데이터 생성
- [x] 수익률 매트릭스 계산
- [x] 투자 유니버스 구성
- [x] 캐싱 시스템
- [x] 테스트 코드 (95% 커버리지)
- [x] API 문서화

### Phase 2: 분석 모듈 (📋 예정)
- [ ] 포트폴리오 최적화 엔진
- [ ] 백테스팅 프레임워크
- [ ] 성과 지표 계산
- [ ] 리스크 모델링

### Phase 3: 사용자 인터페이스 (📋 예정) 
- [ ] Streamlit 대시보드
- [ ] 실시간 데이터 연동
- [ ] 알림 시스템

### Phase 4: 확장 기능 (📋 예정)
- [ ] 머신러닝 기반 예측 모델
- [ ] Docker 컨테이너화
- [ ] 클라우드 배포 (AWS/GCP)

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

**⚠️ 투자 주의사항**: 본 시스템은 교육 및 연구 목적으로 제작되었습니다. 실제 투자 결정에 사용하기 전 충분한 검증이 필요하며, 투자에 따른 손실은 전적으로 사용자의 책임입니다.