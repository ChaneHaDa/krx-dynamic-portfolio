# 알려진 이슈 및 제한사항

> **KRX Dynamic Portfolio** - 현재 버전의 알려진 문제점과 해결 방안

## 📋 목차

1. [테스트 실패 이슈](#테스트-실패-이슈)
2. [대시보드 제한사항](#대시보드-제한사항)
3. [데이터 연동 이슈](#데이터-연동-이슈)
4. [성능 관련 이슈](#성능-관련-이슈)
5. [해결 계획](#해결-계획)

---

## 🧪 테스트 실패 이슈

### 현재 상태: 13개 테스트 실패 (총 210개 중) - 4개 수정 완료! ✅

#### 1. ✅ 백테스팅 엔진 이슈 (4개 모두 수정 완료)

**`test_apply_market_returns`** ✅ **수정 완료**
- **문제**: 마켓 리턴 적용 후 포트폴리오 가치 변화 감지 실패
- **원인**: 부동소수점 정밀도 이슈 및 연간 캐시 이율을 일간으로 잘못 적용
- **해결**: 연간 요율을 일간으로 변환 (`cash_rate / 252.0`) 및 tolerance 기반 비교 적용

```python
# 수정된 코드
daily_cash_rate = self.cash_rate / 252.0
new_cash = portfolio_state["cash"] * (1 + daily_cash_rate)
assert not np.isclose(updated_state['total_value'], original_total_value, rtol=1e-10)
```

**`test_should_rebalance`** ✅ **수정 완료**
- **문제**: 리밸런싱 조건 판단 로직 오류
- **원인**: 인덱스 범위 체크가 스케줄 기반 리밸런싱을 차단
- **해결**: 스케줄 리밸런싱을 우선 체크하도록 로직 개선

**`test_edge_case_single_asset`** ✅ **수정 완료**
- **문제**: 단일 자산 포트폴리오 처리 시 이력 길이 불일치  
- **원인**: 테스트 데이터의 가중치와 백테스트 기간 불일치
- **해결**: 테스트 데이터를 일별 가중치로 변경하여 현실적으로 수정

**`test_export_results`** ✅ **수정 완료**
- **문제**: parquet 파일 저장 시 직렬화 오류
- **원인**: pandas Series 객체의 pyarrow 변환 실패
- **해결**: CSV 형식으로 변경 및 객체 타입 문자열 변환

```python
# 수정된 코드
self.results["portfolio_history"].to_csv(output_path / "portfolio_history.csv")
rebalance_df[col] = rebalance_df[col].astype(str)  # 객체 타입 변환
```

#### 2. 백테스팅 파이프라인 이슈 (6개 실패)

**공통 문제점**:
- 설정 검증 로직 미완성
- 날짜 범위 처리 엣지 케이스
- 최소 설정 시나리오 오류

**영향도**: 높음 - 전체 백테스팅 워크플로우 영향

#### 3. 성과 지표 계산 이슈 (4개 실패)

**극단적 케이스 처리 실패**:
- **제로 변동성**: `RuntimeWarning: invalid value encountered in scalar power`
- **빈 시리즈**: 길이 0 데이터 처리 오류
- **단일 관측값**: 통계 계산 불가
- **양수만 있는 수익률**: 하방 편차 계산 오류

```python
# 오류 발생하는 계산
annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1

# 안전한 계산으로 수정 필요
if len(returns) > 0 and not np.isnan(total_return):
    annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
else:
    annualized_return = np.nan
```

#### 4. 리스크 분석 이슈 (2개 실패)

**`test_edge_case_extreme_outliers`**
- **문제**: 극값 처리 시 수치적 불안정
- **원인**: 아웃라이어가 있는 데이터의 VaR/CVaR 계산

**`test_invalid_method_error`**
- **문제**: 잘못된 메서드명 오류 처리 미흡
- **원인**: 예외 처리 로직 불완전

#### 5. 리스크 모델 이슈 (1개 실패)

**`test_factor_exposure_placeholder`**
- **문제**: 팩터 노출도 분석 기능 미구현
- **원인**: placeholder 함수로만 존재
- **영향**: 낮음 - 고급 기능이므로 기본 사용에는 영향 없음

---

## 📊 대시보드 제한사항

### 1. 테스트 커버리지 0%

**현재 상태**: 대시보드 모듈 전체가 테스트되지 않음

```
krx_portfolio/app/dashboard.py                           159    159     0%
krx_portfolio/app/data_integration.py                    107    107     0%
krx_portfolio/app/components/charts.py                    81     81     0%
krx_portfolio/app/components/data_components.py          162    162     0%
krx_portfolio/app/components/portfolio_components.py      93     93     0%
```

**영향**: 
- UI 기능의 정확성 미보장
- 실제 데이터 연동 시 예상치 못한 오류 가능
- 사용자 입력 검증 부족

### 2. 실시간 데이터 연동 한계

**yfinance API 제한사항**:
- 분당 요청 제한 (정확한 한도 불명)
- 데이터 지연 시간 (15분~1시간)
- 네트워크 의존성으로 인한 불안정성

**KRX 종목코드 변환 이슈**:
```python
# 현재 간단한 구현
def krx_symbol_to_yfinance(krx_code: str) -> str:
    return f"{krx_code}.KS"  # 모든 종목을 KOSPI로 처리

# 실제로는 KOSPI(.KS)와 KOSDAQ(.KQ) 구분 필요
```

### 3. 메모리 관리 이슈

**대용량 데이터 처리 시**:
- Streamlit 캐시 메모리 누적
- 장시간 실행 시 메모리 사용량 증가
- 브라우저 메모리 부족 현상 가능

### 4. UI/UX 제한사항

**반응성 이슈**:
- 대용량 차트 렌더링 지연
- 실시간 업데이트 미지원 (수동 새로고침 필요)
- 모바일 최적화 부족

---

## 🔄 데이터 연동 이슈

### 1. ETL 파이프라인 연동 부족

**현재 상태**: 대시보드에서 실제 ETL 실행 기능 미구현

```python
# dashboard.py의 placeholder 코드
if st.button("🔄 ETL 파이프라인 실행"):
    with st.spinner("ETL 파이프라인을 실행하는 중..."):
        # 실제 ETL 실행 코드 필요
        st.success("ETL 파이프라인이 성공적으로 실행되었습니다!")
```

**필요 작업**:
- 실제 ETL 모듈과의 연동 구현
- 진행 상황 실시간 표시
- 오류 처리 및 로그 표시

### 2. 최적화/백테스팅 연동 부족

**현재 상태**: 샘플 데이터만 사용, 실제 계산 엔진 미연동

**필요한 연동**:
```python
# 실제 최적화 실행
optimization_pipeline = PortfolioOptimizationPipeline(config=config)
results = optimization_pipeline.build_weights(mu, returns, sector_map)

# 실제 백테스팅 실행  
backtest_pipeline = BacktestPipeline(config=config)
results = backtest_pipeline.run_full_backtest(data_root, start_date, end_date)
```

### 3. 데이터 형식 불일치

**문제점**:
- 대시보드 컴포넌트가 기대하는 데이터 형식
- 실제 ETL/최적화 모듈의 출력 형식
- 두 형식 간의 변환 로직 부족

---

## ⚡ 성능 관련 이슈

### 1. Streamlit 캐싱 이슈

**문제점**:
- `@st.cache_data` 데코레이터의 과도한 사용
- 캐시 무효화 로직 부족
- 메모리 누수 가능성

**임시 해결**:
```python
# 캐시 수동 정리
st.cache_data.clear()

# TTL 기반 캐시 사용
@st.cache_data(ttl=3600)  # 1시간
def expensive_computation():
    pass
```

### 2. 대용량 데이터 처리

**현재 한계**:
- 1,000개 이상 종목 처리 시 성능 저하
- 5년 이상 일간 데이터 로딩 지연
- 복잡한 차트 렌더링 속도 이슈

**권장사항**:
- 데이터 샘플링 적용
- 청크 단위 처리
- 지연 로딩 구현

### 3. 브라우저 호환성

**테스트 환경**: Chrome 기준으로만 테스트됨

**미지원 기능**:
- Safari에서 일부 차트 렌더링 이슈
- Internet Explorer 완전 미지원
- 모바일 브라우저 터치 이벤트 부족

---

## 🔧 해결 계획

### 단기 계획 (1-2주)

#### 1. 핵심 테스트 오류 수정
- [ ] 백테스팅 엔진 부동소수점 정밀도 수정
- [ ] 성과 지표 극단적 케이스 처리 추가  
- [ ] parquet 저장 오류 → CSV 대안 구현

#### 2. 대시보드 기본 연동 구현
- [ ] ETL 파이프라인 실제 실행 연동
- [ ] 샘플 데이터 → 실제 데이터 전환
- [ ] 기본적인 오류 처리 추가

### 중기 계획 (1개월)

#### 1. 테스트 커버리지 확대
- [ ] 대시보드 모듈 단위 테스트 추가 (목표: 70%+)
- [ ] UI 컴포넌트 렌더링 테스트
- [ ] 데이터 연동 통합 테스트

#### 2. 성능 최적화
- [ ] 메모리 사용량 모니터링 및 최적화
- [ ] 대용량 데이터 처리 로직 개선
- [ ] 캐싱 전략 재설계

#### 3. 사용자 경험 개선
- [ ] 실시간 업데이트 기능 (WebSocket)
- [ ] 진행 상황 표시 개선
- [ ] 모바일 반응형 레이아웃

### 장기 계획 (2-3개월)

#### 1. 고급 기능 구현
- [ ] 팩터 모델링 완전 구현
- [ ] 머신러닝 기반 예측 모델
- [ ] 고급 리스크 분석 (스트레스 테스팅)

#### 2. 아키텍처 개선
- [ ] 마이크로서비스 아키텍처 도입
- [ ] Docker 컨테이너화
- [ ] 클라우드 배포 자동화

#### 3. 확장성 강화
- [ ] 다중 사용자 지원
- [ ] 포트폴리오 저장/로드 기능
- [ ] API 서버 분리

---

## 🚨 우선순위 분류

### 🔴 높음 (즉시 수정 필요)
1. 백테스팅 리밸런싱 로직 오류
2. ETL-대시보드 연동 구현
3. 성과 지표 극단적 케이스 처리

### 🟡 중간 (1개월 내 수정)
1. parquet 저장 오류 → CSV 대안
2. 대시보드 테스트 커버리지 확대
3. 메모리 누수 방지

### 🟢 낮음 (2개월 내 개선)
1. 팩터 노출도 분석 구현
2. 모바일 최적화
3. 고급 차트 기능 추가

---

## 📞 이슈 리포팅

새로운 이슈 발견 시:
1. **GitHub Issues**에 상세한 재현 방법과 함께 등록
2. **우선순위 라벨** 추가 (Critical/High/Medium/Low)
3. **관련 모듈 태그** 추가 (ETL/Optimization/Backtesting/Dashboard)

**이슈 템플릿**:
```markdown
## 🐛 버그 리포트

### 환경
- OS: [Linux/Windows/macOS]
- Python: [3.9/3.10/3.11/3.12]
- 브라우저: [Chrome/Safari/Firefox]

### 재현 방법
1. 대시보드 실행
2. [구체적인 단계]
3. 오류 발생

### 예상 동작
[정상적으로 동작해야 하는 내용]

### 실제 동작  
[실제로 일어나는 오류 현상]

### 추가 정보
- 오류 로그
- 스크린샷
- 관련 데이터 파일
```

---

**📝 문서 버전**: v1.0.0  
**📅 마지막 업데이트**: 2025-08-29  
**✍️ 작성자**: KRX Dynamic Portfolio Team