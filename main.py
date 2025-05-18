import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import FinanceDataReader as fdr
import time
import numpy as np

# 0. 메인화면 상단: Expander로 사용법 및 설명
with st.expander("ℹ️ 사용법 및 설명 보기", expanded=True):
  st.markdown("""
  **YieldMax MSTY ETF 전략 대시보드**
  
  - 이 대시보드는 ETF(또는 주식) 가격 데이터를 기반으로 다양한 기술적 지표와 전략 신호를 시각화합니다.
  - 사이드바에서 티커를 직접 입력하거나, CSV 파일을 업로드하여 데이터를 분석할 수 있습니다.
  - MSTY(또는 입력한 티커)의 데이터가 FinanceDataReader에 없을 경우, 예시로 SPY ETF 데이터가 자동으로 사용됩니다.
  
  **카테고리별 지원 지표 및 기능**
  - **추세(Trend)**: 이동평균선(20/50/100/200), MACD, 볼린저밴드, 이동평균선 교차(골든/데드크로스)
  - **모멘텀(Momentum)**: MACD Histogram, Squeeze Momentum, Stochastic Oscillator
  - **상대강도(Relative Strength)**: RSI, CCI(20)
  - **거래량(Volume)**: 거래량, 거래량 이동평균, MFI, OBV
  - **패턴(Pattern)**: 캔들차트, 볼린저밴드, 캔들패턴 자동 인식(Hammer, Engulfing 등)
  
  - **분석 결과는 카테고리별로 그룹화되어 메인화면에 표시됩니다.**
  - **사이드바에는 각 카테고리별 대표 신호(매수/매도/중립 확률)가 요약되어 표시됩니다.**
  - 각 차트 아래 expander에서 최신값 기준 매수/매도/중립 확률 해석을 제공합니다.
  
  **사이드바 사용법**
  1. 분석할 티커(예: SPY, QQQ, AAPL 등)를 입력하거나, CSV 파일을 업로드하세요.
  2. 조회 기간, 가격 타입을 선택하세요.
  3. 차트와 각 지표별 해석(expander), 그리고 사이드바의 카테고리별 신호 요약을 확인하세요.
  
  **CSV 업로드 형식**
  - 반드시 'Date', 'Open', 'High', 'Low', 'Close', 'Volume' 컬럼이 포함되어야 합니다.
  - 'Date' 컬럼은 날짜 형식이어야 하며, 인덱스로 사용됩니다.
  """)

# 1. 사이드바: 파라미터 입력
st.sidebar.title("ETF 전략 파라미터")
ticker = st.sidebar.text_input("티커 입력 (예: SPY, QQQ, AAPL 등)", value="SPY")
csv_file = st.sidebar.file_uploader("또는 CSV 파일 업로드", type=["csv"])
period = st.sidebar.selectbox(
  "조회 기간",
  options=["1개월", "3개월", "6개월", "1년", "최대"],
  index=2
)
price_type = st.sidebar.selectbox(
  "가격 타입",
  options=["종가", "시가", "고가", "저가"]
)

# 2. 기간 변환 및 날짜 계산
from datetime import datetime, timedelta
now = datetime.now()
period_map = {
  "1개월": 30,
  "3개월": 90,
  "6개월": 180,
  "1년": 365,
  "최대": 3650
}
days = period_map[period]
start_date = now - timedelta(days=days)

# 3. 데이터 불러오기 함수 (FinanceDataReader)
def load_fdr_data(ticker, start, end):
  try:
    df = fdr.DataReader(ticker, start, end)
    if not df.empty:
      df.index = pd.to_datetime(df.index)
      return df
  except Exception as e:
    st.warning(f"{ticker} 데이터 요청 중 오류 발생: {e}")
  return pd.DataFrame()

# 4. 데이터 소스 결정
if csv_file is not None:
  try:
    df = pd.read_csv(csv_file)
    if 'Date' in df.columns:
      df['Date'] = pd.to_datetime(df['Date'])
      df.set_index('Date', inplace=True)
    else:
      st.error("CSV 파일에 'Date' 컬럼이 존재해야 합니다.")
      st.stop()
    # 컬럼명 표준화
    rename_map = {c: c.capitalize() for c in ['open','high','low','close','volume']}
    df.rename(columns=rename_map, inplace=True)
    # 데이터가 충분한지 확인
    if not all(col in df.columns for col in ['Open','High','Low','Close','Volume']):
      st.error("CSV 파일에 'Open', 'High', 'Low', 'Close', 'Volume' 컬럼이 모두 존재해야 합니다.")
      st.stop()
  except Exception as e:
    st.error(f"CSV 파일을 읽는 중 오류 발생: {e}")
    st.stop()
else:
  df = load_fdr_data(ticker, start_date, now)
  if df.empty:
    st.warning(f"{ticker} 데이터가 FinanceDataReader에 존재하지 않거나, 정책상 차단되었습니다. 예시로 SPY ETF 데이터를 사용합니다.")
    ticker = "SPY"
    df = load_fdr_data(ticker, start_date, now)
    if df.empty:
      st.error("SPY 데이터도 불러올 수 없습니다. 네트워크 또는 FinanceDataReader 상태를 확인하세요.")
      st.stop()

# 5. 가격 타입 매핑 (FinanceDataReader 표준 컬럼명)
price_type_map = {
  "종가": "Close",
  "시가": "Open",
  "고가": "High",
  "저가": "Low"
}
pt = price_type_map[price_type]

# 6. 단순 이동평균선(SMA) 추가 (20, 50, 100, 200일)
df["SMA20"] = df[pt].rolling(window=20).mean()
df["SMA50"] = df[pt].rolling(window=50).mean()
df["SMA100"] = df[pt].rolling(window=100).mean()
df["SMA200"] = df[pt].rolling(window=200).mean()

# 6-1. 20일 거래량 이동평균선 추가
df["VOL_MA20"] = df["Volume"].rolling(window=20).mean()

# 7. MFI 계산
tp = (df["High"] + df["Low"] + df["Close"]) / 3
mf = tp * df["Volume"]
df["TP"] = tp
df["MF"] = mf

df["PMF"] = 0
df["NMF"] = 0
for i in range(1, len(df)):
  if df["TP"].iloc[i] > df["TP"].iloc[i-1]:
    df["PMF"].iloc[i] = df["MF"].iloc[i]
    df["NMF"].iloc[i] = 0
  else:
    df["PMF"].iloc[i] = 0
    df["NMF"].iloc[i] = df["MF"].iloc[i]

df["PMF_sum"] = df["PMF"].rolling(window=20).sum()
df["NMF_sum"] = df["NMF"].rolling(window=20).sum()
df["MFR"] = df["PMF_sum"] / df["NMF_sum"]
df["MFI20"] = 100 - (100 / (1 + df["MFR"]))

# --- RSI (14) --- (EWMA 방식, NaN 최소화)
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI14'] = calc_rsi(df[pt], 14)

# --- Stochastic Oscillator (14, 3, 3) ---
low14 = df['Low'].rolling(window=14).min()
high14 = df['High'].rolling(window=14).max()
df['%K'] = 100 * (df[pt] - low14) / (high14 - low14)
df['%D'] = df['%K'].rolling(window=3).mean()

# --- MACD (12, 26, 9) ---
ema12 = df[pt].ewm(span=12, adjust=False).mean()
ema26 = df[pt].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_hist'] = df['MACD'] - df['MACD_signal']

# --- Squeeze Momentum (LazyBear) ---
# 볼린저밴드
bb_mid = df[pt].rolling(window=20).mean()
bb_std = df[pt].rolling(window=20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
# 켈트너채널
tr = pd.concat([
    df['High'] - df['Low'],
    abs(df['High'] - df[pt].shift()),
    abs(df['Low'] - df[pt].shift())
], axis=1).max(axis=1)
atr = tr.rolling(window=20).mean()
kc_upper = bb_mid + 1.5 * atr
kc_lower = bb_mid - 1.5 * atr
# Squeeze On/Off
squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
df['squeeze_on'] = squeeze_on
# Momentum (간단화 버전)
df['mom'] = df[pt] - df[pt].shift(20)
df['squeeze_mom'] = df['mom'].rolling(window=20).mean()

# --- CCI (Commodity Channel Index, 20일) ---
def calc_cci(df, n=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(window=n).mean()
    md = tp.rolling(window=n).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (tp - ma) / (0.015 * md)
    return cci

df['CCI20'] = calc_cci(df, 20)

# --- OBV (On-Balance Volume) ---
def calc_obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

df['OBV'] = calc_obv(df[pt], df['Volume'])

# --- 캔들패턴 인식 (Hammer, Engulfing 등) ---
def detect_candle_patterns(df):
    patterns = []
    for i in range(len(df)):
        o = df['Open'].iloc[i]
        h = df['High'].iloc[i]
        l = df['Low'].iloc[i]
        c = df['Close'].iloc[i]
        # Hammer
        if (c > o) and ((o - l) > 2 * abs(c - o)) and ((h - c) < 0.3 * (o - l)):
            patterns.append('Hammer')
        # Bullish Engulfing
        elif i > 0 and (df['Close'].iloc[i-1] < df['Open'].iloc[i-1]) and (c > o) and (c > df['Open'].iloc[i-1]) and (o < df['Close'].iloc[i-1]):
            patterns.append('Bullish Engulfing')
        # Bearish Engulfing
        elif i > 0 and (df['Close'].iloc[i-1] > df['Open'].iloc[i-1]) and (c < o) and (c < df['Open'].iloc[i-1]) and (o > df['Close'].iloc[i-1]):
            patterns.append('Bearish Engulfing')
        else:
            patterns.append('')
    return patterns

df['CandlePattern'] = detect_candle_patterns(df)

# 8. 시각화
st.title(f"{ticker} ETF 전략 대시보드")
st.write(f"기간: {period}, 20/50/100/200일 이동평균, 20일 MFI")

fig = go.Figure()
fig.add_trace(go.Candlestick(
  x=df.index,
  open=df["Open"],
  high=df["High"],
  low=df["Low"],
  close=df["Close"],
  name="가격"
))
fig.add_trace(go.Scatter(
  x=df.index, y=df["SMA20"], mode="lines", name="20일 이동평균", line=dict(color="orange")
))
fig.add_trace(go.Scatter(
  x=df.index, y=df["SMA50"], mode="lines", name="50일 이동평균", line=dict(color="blue")
))
fig.add_trace(go.Scatter(
  x=df.index, y=df["SMA100"], mode="lines", name="100일 이동평균", line=dict(color="green")
))
fig.add_trace(go.Scatter(
  x=df.index, y=df["SMA200"], mode="lines", name="200일 이동평균", line=dict(color="red")
))
fig.update_layout(xaxis_rangeslider_visible=False)

# --- Plotly Figure 객체 미리 정의 ---

# 볼린저밴드 차트
bb_fig = go.Figure()
bb_fig.add_trace(go.Scatter(x=df.index, y=bb_upper, name='Upper Band', line=dict(color='red', dash='dot')))
bb_fig.add_trace(go.Scatter(x=df.index, y=bb_mid, name='Middle Band', line=dict(color='blue')))
bb_fig.add_trace(go.Scatter(x=df.index, y=bb_lower, name='Lower Band', line=dict(color='red', dash='dot')))
bb_fig.add_trace(go.Scatter(x=df.index, y=df[pt], name='Price', line=dict(color='black')))
bb_fig.update_layout(showlegend=True)

# MACD 차트
macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')))
macd_fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal', line=dict(color='orange')))
macd_fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Hist', marker_color='green'))
macd_fig.update_layout(showlegend=True, barmode='relative')

# Squeeze Momentum 차트
sq_fig = go.Figure()
sq_fig.add_trace(go.Bar(x=df.index, y=df['squeeze_mom'], name='Squeeze Momentum', marker_color='purple'))
sq_fig.add_trace(go.Scatter(x=df.index, y=[0]*len(df), name='Zero Line', line=dict(color='black', dash='dot')))
sq_fig.add_trace(go.Scatter(x=df.index, y=df['squeeze_on'].astype(int)*df['squeeze_mom'].max(), name='Squeeze On', line=dict(color='red', dash='dot')))
sq_fig.update_layout(showlegend=True)

# Stochastic Oscillator 차트
stoch_fig = go.Figure()
stoch_fig.add_trace(go.Scatter(x=df.index, y=df['%K'], name='%K', line=dict(color='blue')))
stoch_fig.add_trace(go.Scatter(x=df.index, y=df['%D'], name='%D', line=dict(color='orange')))
stoch_fig.update_layout(showlegend=True)

# 거래량 차트
vol_fig = go.Figure()
vol_fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color="#b0b0b0"))
vol_fig.add_trace(go.Scatter(x=df.index, y=df["VOL_MA20"], mode="lines", name="20일 거래량 이동평균", line=dict(color="orange")))
vol_fig.update_layout(showlegend=True)

# 20/50일선 골든/데드크로스 신호 분석
cross_golden = (df['SMA20'] > df['SMA50']) & (df['SMA20'].shift(1) <= df['SMA50'].shift(1))
cross_dead = (df['SMA20'] < df['SMA50']) & (df['SMA20'].shift(1) >= df['SMA50'].shift(1))

# 추세(Trend)
st.markdown('## 📈 추세(Trend)')
st.subheader('이동평균선 & 캔들차트')
st.plotly_chart(fig, use_container_width=True, key="main_price_chart")

st.subheader('Bollinger Bands')
st.plotly_chart(bb_fig, use_container_width=True, key="bollinger_band_chart")
with st.expander("Bollinger Bands 해석 및 신호 설명"):
    price = df[pt].dropna().iloc[-1]
    upper = bb_upper.dropna().iloc[-1]
    lower = bb_lower.dropna().iloc[-1]
    if price > upper:
        st.markdown("**매도 확률 높음**: 가격이 상단 밴드를 돌파했습니다.")
    elif price < lower:
        st.markdown("**매수 확률 높음**: 가격이 하단 밴드 아래에 있습니다.")
    else:
        st.markdown("**중립 확률 높음**: 가격이 밴드 내에 있습니다.")
with st.expander("이동평균선(20/50) 교차 신호 해석"):
    cross_golden = (df['SMA20'] > df['SMA50']) & (df['SMA20'].shift(1) <= df['SMA50'].shift(1))
    cross_dead = (df['SMA20'] < df['SMA50']) & (df['SMA20'].shift(1) >= df['SMA50'].shift(1))
    if cross_golden.iloc[-1]:
        st.markdown("**매수 확률 높음**: 20일선이 50일선을 상향 돌파(골든크로스)했습니다.")
    elif cross_dead.iloc[-1]:
        st.markdown("**매도 확률 높음**: 20일선이 50일선을 하향 돌파(데드크로스)했습니다.")
    else:
        st.markdown("**중립 확률 높음**: 최근 교차 신호가 없습니다.")
st.subheader('MACD')
st.plotly_chart(macd_fig, use_container_width=True, key="macd_chart_1")
with st.expander("MACD 해석 및 신호 설명"):
    macd = df['MACD'].dropna().iloc[-1]
    signal = df['MACD_signal'].dropna().iloc[-1]
    if macd > signal:
        st.markdown("**매수 확률 높음**: MACD가 Signal 위에 있습니다.")
    elif macd < signal:
        st.markdown("**매도 확률 높음**: MACD가 Signal 아래에 있습니다.")
    else:
        st.markdown("**중립 확률 높음**: MACD와 Signal이 거의 같습니다.")

# 모멘텀(Momentum)
st.markdown('## ⚡️ 모멘텀(Momentum)')
st.subheader('MACD Histogram & Squeeze Momentum')
st.plotly_chart(macd_fig, use_container_width=True, key="macd_chart_2")
st.plotly_chart(sq_fig, use_container_width=True, key="squeeze_momentum_chart")
with st.expander("Squeeze Momentum 해석 및 신호 설명"):
    mom = df['squeeze_mom'].dropna().iloc[-1]
    squeeze = df['squeeze_on'].iloc[-1]
    if squeeze:
        st.markdown("**중립 확률 높음**: Squeeze 구간(변동성 축소)입니다. 추세 전환에 주의하세요.")
    elif mom > 0:
        st.markdown("**매수 확률 높음**: 모멘텀이 양수입니다.")
    elif mom < 0:
        st.markdown("**매도 확률 높음**: 모멘텀이 음수입니다.")
    else:
        st.markdown("**중립 확률 높음**: 모멘텀이 0에 가깝습니다.")
st.subheader('Stochastic Oscillator')
st.plotly_chart(stoch_fig, use_container_width=True, key="stochastic_chart")
with st.expander("Stochastic Oscillator 해석 및 신호 설명"):
    k = df['%K'].dropna().iloc[-1]
    d = df['%D'].dropna().iloc[-1]
    if k < 20 and d < 20:
        st.markdown("**매수 확률 높음**: %K, %D 모두 20 이하로 과매도 구간입니다.")
    elif k > 80 and d > 80:
        st.markdown("**매도 확률 높음**: %K, %D 모두 80 이상으로 과매수 구간입니다.")
    else:
        st.markdown("**중립 확률 높음**: Stochastic이 중립 구간(20~80)입니다.")

# 상대강도(Relative Strength)
st.markdown('## 💪 상대강도(Relative Strength)')
st.subheader('RSI(14)')
st.line_chart(df['RSI14'].dropna())
with st.expander("RSI(14) 해석 및 신호 설명"):
    rsi = df['RSI14'].dropna().iloc[-1]
    if rsi < 30:
        st.markdown("**매수 확률 높음**: RSI가 30 이하로 과매도 구간입니다.")
    elif rsi > 70:
        st.markdown("**매도 확률 높음**: RSI가 70 이상으로 과매수 구간입니다.")
    else:
        st.markdown("**중립 확률 높음**: RSI가 중립 구간(30~70)입니다.")
st.subheader('CCI(20)')
st.line_chart(df['CCI20'].dropna())
with st.expander("CCI(20) 해석 및 신호 설명"):
    cci = df['CCI20'].dropna().iloc[-1]
    if cci > 100:
        st.markdown("**매수 확률 높음**: CCI가 100 이상입니다.")
    elif cci < -100:
        st.markdown("**매도 확률 높음**: CCI가 -100 이하입니다.")
    else:
        st.markdown("**중립 확률 높음**: CCI가 -100~100 구간입니다.")

# 거래량(Volume)
st.markdown('## 📊 거래량(Volume)')
st.subheader('거래량 (20일 이동평균 포함)')
vol_fig = go.Figure()
vol_fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="거래량", marker_color="#b0b0b0"))
vol_fig.add_trace(go.Scatter(x=df.index, y=df["VOL_MA20"], mode="lines", name="20일 거래량 이동평균", line=dict(color="orange")))
vol_fig.update_layout(showlegend=True)
st.plotly_chart(vol_fig, use_container_width=True, key="volume_chart")
st.subheader('MFI(20)')
st.line_chart(df["MFI20"])
with st.expander("MFI(20) 해석 및 신호 설명"):
    mfi = df["MFI20"].dropna().iloc[-1]
    if mfi < 20:
        st.markdown("**매수 확률 높음**: MFI가 20 이하로 과매도 구간입니다.")
    elif mfi > 80:
        st.markdown("**매도 확률 높음**: MFI가 80 이상으로 과매수 구간입니다.")
    else:
        st.markdown("**중립 확률 높음**: MFI가 중립 구간(20~80)입니다.")
st.subheader('OBV (On-Balance Volume)')
st.line_chart(df['OBV'])
with st.expander("OBV 해석 및 신호 설명"):
    obv = df['OBV'].iloc[-1]
    obv_prev = df['OBV'].iloc[-2] if len(df['OBV']) > 1 else obv
    if obv > obv_prev:
        st.markdown("**매수 확률 높음**: OBV가 상승 중입니다.")
    elif obv < obv_prev:
        st.markdown("**매도 확률 높음**: OBV가 하락 중입니다.")
    else:
        st.markdown("**중립 확률 높음**: OBV가 변동이 없습니다.")

# 패턴(Pattern)
st.markdown('## 🕯️ 패턴(Pattern)')
st.subheader('캔들패턴(패턴 분석) 자동 인식 결과')
with st.expander("캔들패턴(패턴 분석) 자동 인식 결과"):
    last_pattern = df['CandlePattern'].iloc[-1]
    if last_pattern:
        st.markdown(f"**{last_pattern} 패턴 감지됨**: 최근 캔들에서 {last_pattern} 패턴이 나타났습니다.")
    else:
        st.markdown("최근 캔들에서 뚜렷한 패턴이 감지되지 않았습니다.")

# --------- 카테고리별 신호 종합 함수 ---------
def get_trend_signal():
    signals = []
    cross_golden = (df['SMA20'] > df['SMA50']) & (df['SMA20'].shift(1) <= df['SMA50'].shift(1))
    cross_dead = (df['SMA20'] < df['SMA50']) & (df['SMA20'].shift(1) >= df['SMA50'].shift(1))
    if cross_golden.iloc[-1]:
        signals.append('매수')
    elif cross_dead.iloc[-1]:
        signals.append('매도')
    macd = df['MACD'].dropna().iloc[-1]
    signal = df['MACD_signal'].dropna().iloc[-1]
    if macd > signal:
        signals.append('매수')
    elif macd < signal:
        signals.append('매도')
    else:
        signals.append('중립')
    price = df[pt].dropna().iloc[-1]
    upper = bb_upper.dropna().iloc[-1]
    lower = bb_lower.dropna().iloc[-1]
    if price > upper:
        signals.append('매도')
    elif price < lower:
        signals.append('매수')
    else:
        signals.append('중립')
    if signals.count('매수') > signals.count('매도') and signals.count('매수') > signals.count('중립'):
        return '매수 확률 높음'
    elif signals.count('매도') > signals.count('매수') and signals.count('매도') > signals.count('중립'):
        return '매도 확률 높음'
    else:
        return '중립 확률 높음'

def get_momentum_signal():
    signals = []
    hist = df['MACD_hist'].dropna().iloc[-1]
    if hist > 0:
        signals.append('매수')
    elif hist < 0:
        signals.append('매도')
    else:
        signals.append('중립')
    mom = df['squeeze_mom'].dropna().iloc[-1]
    if mom > 0:
        signals.append('매수')
    elif mom < 0:
        signals.append('매도')
    else:
        signals.append('중립')
    k = df['%K'].dropna().iloc[-1]
    d = df['%D'].dropna().iloc[-1]
    if k < 20 and d < 20:
        signals.append('매수')
    elif k > 80 and d > 80:
        signals.append('매도')
    else:
        signals.append('중립')
    if signals.count('매수') > signals.count('매도') and signals.count('매수') > signals.count('중립'):
        return '매수 확률 높음'
    elif signals.count('매도') > signals.count('매수') and signals.count('매도') > signals.count('중립'):
        return '매도 확률 높음'
    else:
        return '중립 확률 높음'

def get_strength_signal():
    signals = []
    rsi = df['RSI14'].dropna().iloc[-1]
    if rsi < 30:
        signals.append('매수')
    elif rsi > 70:
        signals.append('매도')
    else:
        signals.append('중립')
    cci = df['CCI20'].dropna().iloc[-1]
    if cci > 100:
        signals.append('매수')
    elif cci < -100:
        signals.append('매도')
    else:
        signals.append('중립')
    if signals.count('매수') > signals.count('매도') and signals.count('매수') > signals.count('중립'):
        return '매수 확률 높음'
    elif signals.count('매도') > signals.count('매수') and signals.count('매도') > signals.count('중립'):
        return '매도 확률 높음'
    else:
        return '중립 확률 높음'

def get_volume_signal():
    signals = []
    obv = df['OBV'].iloc[-1]
    obv_prev = df['OBV'].iloc[-2] if len(df['OBV']) > 1 else obv
    if obv > obv_prev:
        signals.append('매수')
    elif obv < obv_prev:
        signals.append('매도')
    else:
        signals.append('중립')
    mfi = df['MFI20'].dropna().iloc[-1]
    if mfi < 20:
        signals.append('매수')
    elif mfi > 80:
        signals.append('매도')
    else:
        signals.append('중립')
    if signals.count('매수') > signals.count('매도') and signals.count('매수') > signals.count('중립'):
        return '매수 확률 높음'
    elif signals.count('매도') > signals.count('매수') and signals.count('매도') > signals.count('중립'):
        return '매도 확률 높음'
    else:
        return '중립 확률 높음'

def get_pattern_signal():
    last_pattern = df['CandlePattern'].iloc[-1]
    if last_pattern in ['Hammer', 'Bullish Engulfing']:
        return '매수 확률 높음'
    elif last_pattern == 'Bearish Engulfing':
        return '매도 확률 높음'
    else:
        return '중립 확률 높음'

# --------- 사이드바에 카테고리별 신호 표시 ---------
st.sidebar.markdown('---')
st.sidebar.header('카테고리별 종합 신호')
st.sidebar.markdown(f"<b>📈 추세(Trend):</b> <span style='color:#007bff;font-weight:bold'>{get_trend_signal()}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<b>⚡️ 모멘텀(Momentum):</b> <span style='color:#28a745;font-weight:bold'>{get_momentum_signal()}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<b>💪 상대강도(Relative Strength):</b> <span style='color:#fd7e14;font-weight:bold'>{get_strength_signal()}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<b>📊 거래량(Volume):</b> <span style='color:#6f42c1;font-weight:bold'>{get_volume_signal()}</span>", unsafe_allow_html=True)
st.sidebar.markdown(f"<b>🕯️ 패턴(Pattern):</b> <span style='color:#e83e8c;font-weight:bold'>{get_pattern_signal()}</span>", unsafe_allow_html=True)
st.sidebar.markdown('---')
