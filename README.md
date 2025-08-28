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
│   ├── etl/                    # 📥 데이터 수집 및 전처리 (완료)
│   │   ├── data_loader.py      # KRX JSON 데이터 로더
│   │   ├── preprocessor.py     # 데이터 전처리 및 특성 엔지니어링  
│   │   └── main.py            # ETL 파이프라인 CLI
│   ├── models/                 # 🧠 포트폴리오 최적화 모델 (완료)
│   │   ├── mpt.py             # Modern Portfolio Theory 최적화
│   │   ├── risk.py            # 리스크 모델링 (Ledoit-Wolf, EWMA, OAS)
│   │   ├── rebalance.py       # 동적 리밸런싱 알고리즘
│   │   └── pipeline.py        # 통합 최적화 파이프라인
│   ├── backtesting/            # 📊 백테스팅 엔진 (완료)
│   │   ├── engine.py          # 포트폴리오 시뮬레이션 엔진
│   │   ├── metrics.py         # 40+ 성과 지표 계산
│   │   ├── risk_analytics.py  # VaR/CVaR/극값분석/스트레스테스팅
│   │   └── main.py            # End-to-end 백테스팅 파이프라인
│   └── app/                    # 🎨 Streamlit 대시보드 (예정)
│       ├── dashboard.py        # 메인 대시보드
│       └── components/         # UI 컴포넌트
├── configs/                    # ⚙️ 설정 파일
│   └── portfolio.yaml         # 포트폴리오 최적화 설정
├── docs/                       # 📖 문서화
│   ├── etl_api.md             # ETL API 문서
│   ├── portfolio_api.md       # 포트폴리오 API 문서
│   ├── portfolio_optimization_guide.md  # 최적화 가이드
│   └── backtesting_guide.md   # 백테스팅 가이드
└── tests/                      # 🧪 테스트 (230+ 테스트)
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
- ✅ **데이터 로더**: 100% 커버리지 (`data_loader.py`)
- ✅ **전처리기**: 100% 커버리지 (`preprocessor.py`)
- ✅ **통합 테스트**: ETL 파이프라인 end-to-end 테스트
- ✅ **코드 품질**: Ruff/Black 포맷팅 완료

### 테스트 통계
- **전체 테스트**: 230+ 개 (ETL 39개 + 포트폴리오 83개 + 백테스팅 110개)
- **전체 커버리지**: 95%+ 
- **핵심 모듈**: ETL, MPT, Risk Model, Rebalancer, Backtesting 완전 구현
- **테스트 범위**: 단위 테스트, 통합 테스트, End-to-End 테스트, 성과 검증
- **구현 수준**: 학술연구급 → 실무급 포트폴리오 시스템
- **총 코드량**: 10,000+ 줄 (핵심 모듈 + 테스트 + 문서)
- **개발 환경**: Python 3.9+, scipy, pandas, scikit-learn, pytest

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

### Phase 2: 포트폴리오 최적화 모듈 (✅ 완료)
- [x] **Modern Portfolio Theory (MPT)** - Max Sharpe, Min Variance, Mean-Variance 최적화
- [x] **리스크 모델링** - Ledoit-Wolf, EWMA, OAS, PSD 보정, 리스크 버짓팅
- [x] **팩터 모델링** - 팩터 노출도 분석 및 특이 리스크 추정
- [x] **동적 리밸런싱** - 스케줄/임계치 기반 거래비용 최적화
- [x] **통합 파이프라인** - End-to-end 최적화 워크플로우
- [x] **제약 조건** - 포지션 제한, 섹터 캡, 회전율 예산
- [x] **설정 관리** - YAML 기반 포트폴리오 구성
- [x] **문서화** - 포트폴리오 최적화 완전 가이드

### Phase 3: 백테스팅 모듈 (✅ 완료)
- [x] **백테스팅 프레임워크** - 포트폴리오 시뮬레이션 엔진 및 성과 평가
- [x] **성과 지표 계산** - 40+ 지표 (Sharpe, Sortino, Calmar, MDD, 드로다운 등)
- [x] **고급 리스크 분석** - VaR/CVaR (다중 방법), 극값이론, 스트레스 테스팅
- [x] **거래비용 모델링** - Linear/Square-root 마켓임팩트, 리밸런싱 최적화
- [x] **통합 파이프라인** - End-to-end 백테스팅 워크플로우 및 CLI
- [x] **포괄적 테스트** - 110+ 테스트 케이스로 완전 검증
- [x] **완전한 문서화** - 400줄 실무급 백테스팅 가이드

### Phase 4: 사용자 인터페이스 (📋 예정) 
- [ ] **Streamlit 대시보드** - 인터랙티브 포트폴리오 시각화
- [ ] **실시간 데이터 연동** - yfinance API 통합
- [ ] **알림 시스템** - 리밸런싱 신호 및 리스크 알림

### Phase 5: 확장 기능 (📋 예정)
- [ ] **머신러닝 기반 예측 모델** - LSTM, Transformer 기반 가격 예측
- [ ] **Docker 컨테이너화** - 배포 환경 표준화
- [ ] **클라우드 배포** - AWS/GCP 자동화 파이프라인

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

**⚠️ 투자 주의사항**: 본 시스템은 교육 및 연구 목적으로 제작되었습니다. 실제 투자 결정에 사용하기 전 충분한 검증이 필요하며, 투자에 따른 손실은 전적으로 사용자의 책임입니다.