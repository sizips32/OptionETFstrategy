# YieldMax MSTY ETF 전략 대시보드

이 프로젝트는 [YieldMax MSTY ETF](https://www.yieldmaxetfs.com/our-etfs/msty/) 및 다양한 ETF/주식의 가격 데이터를 활용하여, 여러 기술적 지표와 전략 신호를 Streamlit 대시보드로 시각화하고 해석을 제공합니다.

## 주요 기능
- FinanceDataReader를 통한 미국/한국 주식 및 ETF 데이터 자동 수집 (또는 CSV 업로드 지원)
- 20/50/100/200일 이동평균선(SMA) 및 20일 거래량 이동평균선
- MFI(20), RSI(14), Stochastic Oscillator(14,3,3), MACD(12,26,9), Squeeze Momentum(LazyBear 방식), 볼린저밴드, 이동평균선 교차(골든/데드크로스), CCI(20), OBV, 캔들패턴(자동 인식) 등 다양한 기술적 지표 계산 및 시각화
- **카테고리별(추세, 모멘텀, 상대강도, 거래량, 패턴)로 분석 결과를 그룹화하여 메인화면에 표시**
- **사이드바에 각 카테고리별 대표 신호(매수/매도/중립 확률)를 요약 표시**
- 각 차트 아래 expander에서 최신값 기준 매수/매도/중립 확률 해석 제공
- 사이드바에서 티커, 기간, 가격 타입 선택 및 CSV 업로드 지원

## 사용 방법
1. Python 3.8 이상 설치
2. 필요한 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```
3. Streamlit 앱 실행:
   ```bash
   streamlit run main.py
   ```
4. 웹 브라우저에서 대시보드 확인 (기본: http://localhost:8501)

## 파일 구조
- `main.py` : Streamlit 대시보드 메인 코드
- `README.md` : 프로젝트 설명 파일
- `requirements.txt` : 의존성 패키지 목록

## 지원 지표 및 해석 (카테고리별)
- **추세(Trend)**
  - 이동평균선(SMA 20/50/100/200)
  - MACD
  - 볼린저밴드
  - 이동평균선 교차(골든/데드크로스)
- **모멘텀(Momentum)**
  - MACD Histogram
  - Squeeze Momentum
  - Stochastic Oscillator
- **상대강도(Relative Strength)**
  - RSI
  - CCI(20)
- **거래량(Volume)**
  - 거래량(Bar), 20일 거래량 이동평균
  - MFI (Money Flow Index)
  - OBV (On-Balance Volume)
- **패턴(Pattern)**
  - 캔들차트
  - 볼린저밴드
  - 캔들패턴 자동 인식 (Hammer, Engulfing 등)

- 각 차트 아래 expander에서 최신값 기준 매수/매도/중립 확률 해석 제공
- **사이드바에서 카테고리별 대표 신호(매수/매도/중립 확률) 요약 제공**

## 사이드바 사용법
1. 분석할 티커(예: SPY, QQQ, AAPL 등)를 입력하거나, CSV 파일을 업로드하세요.
2. 조회 기간, 가격 타입을 선택하세요.
3. 차트와 각 지표별 해석(expander), 그리고 사이드바의 카테고리별 신호 요약을 확인하세요.

## CSV 업로드 형식
- 반드시 'Date', 'Open', 'High', 'Low', 'Close', 'Volume' 컬럼이 포함되어야 합니다.
- 'Date' 컬럼은 날짜 형식이어야 하며, 인덱스로 사용됩니다.

## 참고 자료
- [YieldMax MSTY ETF 공식 페이지](https://www.yieldmaxetfs.com/our-etfs/msty/)
- [FinanceDataReader 공식 문서](https://financedata.github.io/FinanceDataReader/)
- [Streamlit 공식 문서](https://docs.streamlit.io/)

## 라이선스
본 프로젝트는 교육 및 연구 목적이며, 투자 조언이 아닙니다. 
