#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║       XAUUSD SIGNAL PRO — Python Backend v3.0           ║
║  Live Data + NumPy/Pandas Technical Analysis Engine     ║
║  Supports: OpenAI · Claude · Gemini · DeepSeek ·        ║
║            Grok · Mistral                               ║
╚══════════════════════════════════════════════════════════╝
"""

import json, http.server, urllib.request, urllib.error, urllib.parse
import ssl, os, re, time
from datetime import datetime

# ── Optional deps (numpy/pandas) ──────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    HAS_NUMPY = True
    print("  [OK] NumPy & Pandas tersedia — analisis teknikal penuh aktif")
except ImportError:
    HAS_NUMPY = False
    print("  [WARN] NumPy/Pandas tidak ditemukan — install: pip install numpy pandas")

# ──────────────────────────────────────────────────────────────────────────────
PORT       = 5000
DATA_CACHE = {}   # {(symbol,tf): (timestamp, df)}
CACHE_TTL  = {
    "M1":60, "M5":300, "M15":900, "M30":1800,
    "H1":3600, "H4":14400, "D1":86400, "W1":604800,
}
YF_INTERVAL = {
    "M1":"1m","M5":"5m","M15":"15m","M30":"30m",
    "H1":"1h","H4":"4h","D1":"1d","W1":"1wk",
}
YF_RANGE = {
    "M1":"1d","M5":"5d","M15":"7d","M30":"14d",
    "H1":"30d","H4":"60d","D1":"1y","W1":"2y",
}

# ──────────────────────────────────────────────────────────────────────────────
#  DATA LAYER
# ──────────────────────────────────────────────────────────────────────────────

def fetch_yahoo(tf):
    interval = YF_INTERVAL.get(tf, "1h")
    rang     = YF_RANGE.get(tf, "30d")
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD=X"
        f"?interval={interval}&range={rang}&includePrePost=false"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; XAUSignal/3.0)",
        "Accept": "application/json",
    }
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=12) as r:
            raw = json.loads(r.read().decode())
    except Exception as e:
        print(f"    [Yahoo] fetch failed: {e}")
        return None
    try:
        res = raw["chart"]["result"][0]
        ts  = res["timestamp"]
        q   = res["indicators"]["quote"][0]
        df  = pd.DataFrame({
            "open":   q["open"],   "high": q["high"],
            "low":    q["low"],    "close": q["close"],
            "volume": q.get("volume", [0]*len(ts)),
        }, index=pd.to_datetime(ts, unit="s", utc=True))
        df.index = df.index.tz_convert("Asia/Jakarta")
        df.dropna(inplace=True)
        if len(df) < 30:
            return None
        print(f"    [Yahoo] OK {len(df)} candles, last={df['close'].iloc[-1]:.2f}")
        return df
    except Exception as e:
        print(f"    [Yahoo] parse error: {e}")
        return None


def fetch_stooq(tf):
    interval_map = {"D1":"d","W1":"w"}
    intv = interval_map.get(tf)
    if not intv:
        return None
    url = f"https://stooq.com/q/d/l/?s=xauusd&i={intv}"
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
            csv_txt = r.read().decode()
        rows = [l.split(",") for l in csv_txt.strip().split("\n")[1:] if l]
        df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume"])
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.set_index("date", inplace=True)
        df.dropna(inplace=True)
        print(f"    [Stooq] OK {len(df)} rows")
        return df if len(df) >= 30 else None
    except Exception as e:
        print(f"    [Stooq] failed: {e}")
        return None


def generate_synthetic(tf, n=200):
    """GBM-based synthetic OHLCV sebagai fallback offline."""
    np.random.seed(int(time.time()) % 10000)
    vol_map = {
        "M1":0.0003,"M5":0.0006,"M15":0.0009,"M30":0.0013,
        "H1":0.0018,"H4":0.0035,"D1":0.0080,"W1":0.0180,
    }
    sig = vol_map.get(tf, 0.0018)
    S0  = 2678.0 + np.random.uniform(-30, 30)
    drift   = np.random.uniform(-0.0001, 0.0002)
    returns = np.random.normal(drift, sig, n)
    closes  = S0 * np.cumprod(1 + returns)
    sf = sig * 0.8
    highs = closes * (1 + np.abs(np.random.normal(0, sf, n)))
    lows  = closes * (1 - np.abs(np.random.normal(0, sf, n)))
    opens = np.roll(closes, 1); opens[0] = S0
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows  = np.minimum(lows,  np.minimum(opens, closes))
    vols  = np.random.lognormal(10, 1.5, n).astype(int)
    freq_map = {
        "M1":"1min","M5":"5min","M15":"15min","M30":"30min",
        "H1":"1h","H4":"4h","D1":"1D","W1":"7D",
    }
    freq = freq_map.get(tf, "1h")
    end  = pd.Timestamp.now(tz="Asia/Jakarta").floor(freq)
    idx  = pd.date_range(end=end, periods=n, freq=freq)
    df = pd.DataFrame({"open":opens,"high":highs,"low":lows,"close":closes,"volume":vols}, index=idx)
    print(f"    [Synthetic] OK {n} candles GBM, last={df['close'].iloc[-1]:.2f}")
    return df


def get_ohlcv(tf):
    key = ("XAUUSD", tf)
    now = time.time()
    if key in DATA_CACHE:
        ts_c, df_c = DATA_CACHE[key]
        if now - ts_c < CACHE_TTL.get(tf, 3600):
            print(f"    [Cache] hit {tf}, age={int(now-ts_c)}s")
            return df_c
    print(f"    [Data] fetching {tf}...")
    df = fetch_yahoo(tf)
    if df is None: df = fetch_stooq(tf)
    if df is None: df = generate_synthetic(tf)
    DATA_CACHE[key] = (now, df)
    return df


# ──────────────────────────────────────────────────────────────────────────────
#  TECHNICAL ANALYSIS ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class TA:
    def __init__(self, df):
        self.df    = df.copy()
        self.close = df["close"].values.astype(float)
        self.high  = df["high"].values.astype(float)
        self.low   = df["low"].values.astype(float)
        self.open  = df["open"].values.astype(float)
        self.vol   = df["volume"].values.astype(float)
        self.n     = len(self.close)

    def sma(self, p): return pd.Series(self.close).rolling(p).mean().values

    def ema(self, p, data=None):
        s = data if data is not None else self.close
        return pd.Series(s).ewm(span=p, adjust=False).mean().values

    def rsi(self, p=14):
        d = pd.Series(self.close).diff()
        g = d.clip(lower=0); l = (-d).clip(lower=0)
        ag = g.ewm(com=p-1, adjust=False).mean()
        al = l.ewm(com=p-1, adjust=False).mean()
        rs = ag / al.replace(0, np.nan)
        return (100 - 100/(1+rs)).values

    def macd(self, fast=12, slow=26, sig=9):
        ml = self.ema(fast) - self.ema(slow)
        sl = pd.Series(ml).ewm(span=sig, adjust=False).mean().values
        return ml, sl, ml - sl

    def bollinger(self, p=20, k=2.0):
        mid = self.sma(p)
        std = pd.Series(self.close).rolling(p).std().values
        up  = mid + k*std; lo = mid - k*std
        bw  = (up-lo)/(mid+1e-10)
        pb  = (self.close-lo)/(up-lo+1e-10)
        return up, mid, lo, bw, pb

    def atr(self, p=14):
        pc = np.roll(self.close,1); pc[0]=self.close[0]
        tr = np.maximum(self.high-self.low,
             np.maximum(np.abs(self.high-pc), np.abs(self.low-pc)))
        return pd.Series(tr).ewm(com=p-1, adjust=False).mean().values

    def stochastic(self, kp=14, dp=3):
        lo = pd.Series(self.low).rolling(kp).min().values
        hi = pd.Series(self.high).rolling(kp).max().values
        k  = 100*(self.close-lo)/(hi-lo+1e-10)
        return k, pd.Series(k).rolling(dp).mean().values

    def cci(self, p=20):
        tp  = (self.high+self.low+self.close)/3
        sma = pd.Series(tp).rolling(p).mean().values
        mad = pd.Series(tp).rolling(p).apply(lambda x: np.mean(np.abs(x-x.mean())), raw=True).values
        return (tp-sma)/(0.015*mad+1e-10)

    def williams_r(self, p=14):
        hi = pd.Series(self.high).rolling(p).max().values
        lo = pd.Series(self.low).rolling(p).min().values
        return -100*(hi-self.close)/(hi-lo+1e-10)

    def adx(self, p=14):
        pc = np.roll(self.close,1); pc[0]=self.close[0]
        ph = np.roll(self.high,1);  ph[0]=self.high[0]
        pl = np.roll(self.low,1);   pl[0]=self.low[0]
        um = self.high-ph; dm = pl-self.low
        pdm = np.where((um>dm)&(um>0), um, 0.0)
        ndm = np.where((dm>um)&(dm>0), dm, 0.0)
        tr  = np.maximum(self.high-self.low,
              np.maximum(np.abs(self.high-pc), np.abs(self.low-pc))).astype(float)
        atr14 = pd.Series(tr).ewm(com=p-1,adjust=False).mean().values
        pdi   = 100*pd.Series(pdm).ewm(com=p-1,adjust=False).mean().values/(atr14+1e-10)
        ndi   = 100*pd.Series(ndm).ewm(com=p-1,adjust=False).mean().values/(atr14+1e-10)
        dx    = 100*np.abs(pdi-ndi)/(pdi+ndi+1e-10)
        return pd.Series(dx).ewm(com=p-1,adjust=False).mean().values, pdi, ndi

    def vwap(self):
        tp  = (self.high+self.low+self.close)/3
        w   = self.vol[-20:]; tv = w.sum()
        return float(np.sum(tp[-20:]*w)/tv) if tv>0 else float(self.close[-1])

    def support_resistance(self, lb=50):
        p  = (self.high[-1]+self.low[-1]+self.close[-1])/3
        r  = self.high[-1]-self.low[-1]
        return {
            "pivot": p,
            "r1": 2*p-self.low[-1],  "s1": 2*p-self.high[-1],
            "r2": p+r,               "s2": p-r,
            "recent_high": float(np.max(self.high[-lb:])),
            "recent_low":  float(np.min(self.low[-lb:])),
        }

    def trend(self):
        e20=self.ema(20); e50=self.ema(50)
        e200=self.ema(min(200,self.n-1))
        p=self.close[-1]
        s=sum([p>e20[-1], p>e50[-1], p>e200[-1], e20[-1]>e50[-1], e50[-1]>e200[-1]])
        d = ("STRONG_BULL" if s>=4 else "BULL" if s==3 else
             "STRONG_BEAR" if s<=1 else "BEAR" if s==2 else "NEUTRAL")
        return d, s, e20[-1], e50[-1], e200[-1]

    def volume_analysis(self):
        if self.vol.sum()==0: return "NORMAL", 1.0
        avg = np.mean(self.vol[-20:])
        r   = self.vol[-1]/(avg+1e-10)
        lbl = ("VERY_HIGH" if r>2 else "HIGH" if r>1.3 else "LOW" if r<0.7 else "NORMAL")
        return lbl, float(r)

    def rsi_divergence(self, rsi_v, p=5):
        if self.n<p*2: return "NONE"
        pn,pp = self.close[-1],self.close[-p]
        rn,rp = rsi_v[-1],rsi_v[-p]
        if any(map(lambda x: x!=x, [rn,rp])): return "NONE"  # NaN check
        if pn<pp and rn>rp: return "BULLISH"
        if pn>pp and rn<rp: return "BEARISH"
        return "NONE"

    def candle_patterns(self):
        pts=[]
        if self.n<2: return pts
        o,h,l,c = self.open,self.high,self.low,self.close
        bd=abs(c-o); wu=h-np.maximum(c,o); wd=np.minimum(c,o)-l
        i=-1
        if bd[i]>0.001:
            if wd[i]>2*bd[i] and wu[i]<bd[i]:
                pts.append("Hammer" if c[i]>o[i] else "Hanging Man")
            if wu[i]>2*bd[i] and wd[i]<bd[i]:
                pts.append("Shooting Star" if c[i]<o[i] else "Inverted Hammer")
        if self.n>=2:
            if c[-2]<o[-2] and c[-1]>o[-1] and c[-1]>o[-2] and o[-1]<c[-2]:
                pts.append("Bullish Engulfing")
            if c[-2]>o[-2] and c[-1]<o[-1] and c[-1]<o[-2] and o[-1]>c[-2]:
                pts.append("Bearish Engulfing")
        if bd[i]<0.15*(h[i]-l[i]+1e-10): pts.append("Doji")
        return pts or ["No pattern"]


# ──────────────────────────────────────────────────────────────────────────────

def run_analysis(tf):
    """Jalankan full TA, return dict semua indikator + scoring."""
    df = get_ohlcv(tf)
    if not HAS_NUMPY: return {"error":"NumPy tidak tersedia"}

    ta   = TA(df)
    price= float(ta.close[-1])
    prev = float(ta.close[-2]) if ta.n>1 else price

    # Compute indicators
    rsi14    = ta.rsi(14);  rv = float(rsi14[-1])
    rsi7     = ta.rsi(7);   r7 = float(rsi7[-1])
    ml,ms,mh = ta.macd()
    mv,msv,mhv = float(ml[-1]), float(ms[-1]), float(mh[-1])
    bu,bm,bl,bw,pb = ta.bollinger()
    bup,blo,bmd,bbw,bpb = float(bu[-1]),float(bl[-1]),float(bm[-1]),float(bw[-1]),float(pb[-1])
    atr14  = ta.atr(14);  av = float(atr14[-1])
    k,d    = ta.stochastic(); sk,sd = float(k[-1]), float(d[-1])
    cci20  = ta.cci();    cv = float(cci20[-1])
    wpr    = ta.williams_r(); wv = float(wpr[-1])
    adx14,pdi,ndi = ta.adx()
    adv,pdv,ndv = float(adx14[-1]),float(pdi[-1]),float(ndi[-1])
    vwap   = ta.vwap()
    sr     = ta.support_resistance()
    tdir,tscore,e20,e50,e200 = ta.trend()
    vlab,vratio = ta.volume_analysis()
    pats   = ta.candle_patterns()
    rdiv   = ta.rsi_divergence(rsi14)
    e8     = float(ta.ema(8)[-1])
    e20v,e50v,e200v = float(e20),float(e50),float(e200)

    # ── Signal scoring ──
    bulls, bears, neutrals = [], [], []

    # RSI
    if rv<30:   bulls.append(f"RSI oversold sangat ekstrem ({rv:.1f} < 30)")
    elif rv<45: bulls.append(f"RSI di zona oversold ({rv:.1f})")
    elif rv>70: bears.append(f"RSI overbought sangat ekstrem ({rv:.1f} > 70)")
    elif rv>55: bears.append(f"RSI di zona overbought ({rv:.1f})")
    else:       neutrals.append(f"RSI netral di {rv:.1f}")

    # MACD
    if mhv>0 and mv>msv:    bulls.append(f"MACD bullish cross, histogram={mhv:.4f}")
    elif mhv<0 and mv<msv:  bears.append(f"MACD bearish cross, histogram={mhv:.4f}")
    else:                   neutrals.append(f"MACD mendekati crossover")

    # EMA
    if price>e20v>e50v:  bulls.append(f"Price di atas EMA20={e20v:.2f} & EMA50={e50v:.2f}")
    elif price<e20v<e50v: bears.append(f"Price di bawah EMA20={e20v:.2f} & EMA50={e50v:.2f}")
    else:                  neutrals.append(f"EMA struktur campuran")

    # EMA 200
    if price>e200v: bulls.append(f"Price di atas EMA200={e200v:.2f} (bull market)")
    else:           bears.append(f"Price di bawah EMA200={e200v:.2f} (bear market)")

    # Bollinger
    if bpb<0.1:   bulls.append(f"Price menyentuh lower Bollinger ({blo:.2f})")
    elif bpb>0.9: bears.append(f"Price menyentuh upper Bollinger ({bup:.2f})")
    else:         neutrals.append(f"Price dalam BB tengah (%B={bpb:.2f})")

    # Stochastic
    if sk<20 and sd<20:  bulls.append(f"Stochastic oversold K={sk:.1f} D={sd:.1f}")
    elif sk>80 and sd>80: bears.append(f"Stochastic overbought K={sk:.1f} D={sd:.1f}")
    else:                  neutrals.append(f"Stochastic K={sk:.1f} D={sd:.1f}")

    # ADX
    if adv>25:
        if pdv>ndv: bulls.append(f"ADX trending kuat ({adv:.1f}), +DI>{pdv:.1f} dominan")
        else:       bears.append(f"ADX trending kuat ({adv:.1f}), -DI>{ndv:.1f} dominan")
    else:           neutrals.append(f"ADX lemah ({adv:.1f}), pasar sideways/ranging")

    # CCI
    if cv<-100:  bulls.append(f"CCI oversold ({cv:.0f})")
    elif cv>100: bears.append(f"CCI overbought ({cv:.0f})")

    # Williams %R
    if wv<-80:   bulls.append(f"Williams %R oversold ({wv:.1f})")
    elif wv>-20: bears.append(f"Williams %R overbought ({wv:.1f})")

    # VWAP
    if price>vwap: bulls.append(f"Price di atas VWAP={vwap:.2f} (bullish bias)")
    else:          bears.append(f"Price di bawah VWAP={vwap:.2f} (bearish bias)")

    # Divergence
    if rdiv=="BULLISH": bulls.append("RSI Bullish Divergence terdeteksi")
    elif rdiv=="BEARISH": bears.append("RSI Bearish Divergence terdeteksi")

    # Patterns
    for p in pats:
        if p in ("Hammer","Bullish Engulfing","Inverted Hammer"): bulls.append(f"Pola candle bullish: {p}")
        elif p in ("Hanging Man","Shooting Star","Bearish Engulfing"): bears.append(f"Pola candle bearish: {p}")

    # Volume
    if vlab in ("HIGH","VERY_HIGH"):
        dom = "BULL" if price>prev else "BEAR"
        (bulls if dom=="BULL" else bears).append(
            f"Volume tinggi ({vratio:.1f}x rata-rata), konfirmasi {dom}")

    # Trend
    if "BULL" in tdir: bulls.append(f"Trend {tdir} (skor {tscore}/5)")
    elif "BEAR" in tdir: bears.append(f"Trend {tdir} (skor {tscore}/5)")

    # ── Bias & confidence ──
    total = len(bulls)+len(bears)
    bp = len(bulls)/total*100 if total else 50
    bias = "BUY" if bp>50 else "SELL"
    conf = max(bp, 100-bp)

    # ── ATR-based levels ──
    atr = av if av>0.5 else 5.0
    if bias=="BUY":
        entry = round(price+atr*0.05, 2)
        sl    = round(price-atr*1.5, 2)
        tp1   = round(price+atr*2.0, 2)
        tp2   = round(price+atr*3.0, 2)
    else:
        entry = round(price-atr*0.05, 2)
        sl    = round(price+atr*1.5, 2)
        tp1   = round(price-atr*2.0, 2)
        tp2   = round(price-atr*3.0, 2)

    risk   = abs(entry-sl)
    reward = abs(tp1-entry)
    rr     = round(reward/risk, 2) if risk else 2.0

    return {
        "price": round(price,2), "price_change": round(price-prev,2),
        "price_change_pct": round((price-prev)/prev*100,3) if prev else 0,
        "tf": tf, "candles": ta.n,
        # levels
        "entry":entry, "tp1":tp1, "tp2":tp2, "sl":sl, "rr":rr, "atr":round(atr,3),
        # indicators
        "rsi14":round(rv,2), "rsi7":round(r7,2),
        "macd_line":round(mv,4), "macd_signal":round(msv,4), "macd_hist":round(mhv,4),
        "bb_upper":round(bup,2), "bb_mid":round(bmd,2), "bb_lower":round(blo,2),
        "bb_bw":round(bbw*100,2), "bb_pct_b":round(bpb,3),
        "stoch_k":round(sk,2), "stoch_d":round(sd,2),
        "cci":round(cv,2), "wpr":round(wv,2),
        "adx":round(adv,2), "plus_di":round(pdv,2), "minus_di":round(ndv,2),
        "ema8":round(e8,2), "ema20":round(e20v,2),
        "ema50":round(e50v,2), "ema200":round(e200v,2),
        "vwap":round(vwap,2),
        # SR
        "pivot":round(sr["pivot"],2),
        "r1":round(sr["r1"],2), "s1":round(sr["s1"],2),
        "r2":round(sr["r2"],2), "s2":round(sr["s2"],2),
        "recent_high":round(sr["recent_high"],2), "recent_low":round(sr["recent_low"],2),
        # scoring
        "bull_signals":bulls, "bear_signals":bears, "neutral_signals":neutrals,
        "bias":bias, "confidence_raw":round(conf,1),
        "trend":tdir, "trend_score":tscore,
        "vol_label":vlab, "vol_ratio":round(vratio,2),
        "rsi_div":rdiv, "candle_patterns":pats,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  AI PROVIDERS
# ──────────────────────────────────────────────────────────────────────────────

AI_PROVIDERS = {
    "gpt4":    {"name":"GPT-4o",            "url":"https://api.openai.com/v1/chat/completions",         "model":"gpt-4o",                     "auth_header":"Authorization","auth_prefix":"Bearer ","json_mode":True},
    "claude":  {"name":"Claude 3.5 Sonnet", "url":"https://api.anthropic.com/v1/messages",              "model":"claude-3-5-sonnet-20241022",  "extra_headers":{"anthropic-version":"2023-06-01"}},
    "gemini":  {"name":"Gemini 1.5 Pro",    "url":"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent","model":"gemini-1.5-pro","auth_type":"query"},
    "deepseek":{"name":"DeepSeek R1",       "url":"https://api.deepseek.com/v1/chat/completions",       "model":"deepseek-reasoner",           "auth_header":"Authorization","auth_prefix":"Bearer ","json_mode":True},
    "grok":    {"name":"Grok-2",            "url":"https://api.x.ai/v1/chat/completions",               "model":"grok-2-latest",               "auth_header":"Authorization","auth_prefix":"Bearer "},
    "mistral": {"name":"Mistral Large",     "url":"https://api.mistral.ai/v1/chat/completions",         "model":"mistral-large-latest",        "auth_header":"Authorization","auth_prefix":"Bearer "},
}


def build_prompt(ta, strategy, mode, rr_min):
    bulls = "\n".join(f"  [+] {s}" for s in ta["bull_signals"]) or "  (tidak ada)"
    bears = "\n".join(f"  [-] {s}" for s in ta["bear_signals"]) or "  (tidak ada)"
    return f"""Anda adalah analis trading XAUUSD senior. Berikut data teknikal NYATA yang dihitung Python/NumPy:

=== HARGA & DATA ===
Last Price   : {ta['price']} USD | Perubahan: {ta['price_change']:+.2f} ({ta['price_change_pct']:+.3f}%)
Timeframe    : {ta['tf']} | Candles dianalisis: {ta['candles']}

=== MOVING AVERAGES ===
EMA 8/20/50/200 : {ta['ema8']} / {ta['ema20']} / {ta['ema50']} / {ta['ema200']}
VWAP            : {ta['vwap']}
Trend           : {ta['trend']} (skor {ta['trend_score']}/5)

=== OSCILLATORS ===
RSI(14)      : {ta['rsi14']}  | RSI(7): {ta['rsi7']}
MACD Line    : {ta['macd_line']} | Signal: {ta['macd_signal']} | Hist: {ta['macd_hist']}
Stoch K/D    : {ta['stoch_k']} / {ta['stoch_d']}
CCI(20)      : {ta['cci']}
Williams%R   : {ta['wpr']}
RSI Diverg   : {ta['rsi_div']}

=== VOLATILITAS ===
ATR(14)      : {ta['atr']}
ADX          : {ta['adx']} | +DI: {ta['plus_di']} | -DI: {ta['minus_di']}

=== BOLLINGER BANDS ===
Upper/Mid/Lower : {ta['bb_upper']} / {ta['bb_mid']} / {ta['bb_lower']}
%B              : {ta['bb_pct_b']:.3f} | Bandwidth: {ta['bb_bw']:.2f}%

=== SUPPORT & RESISTANCE ===
Pivot  : {ta['pivot']}  | R1: {ta['r1']}  S1: {ta['s1']}
R2     : {ta['r2']}    | S2: {ta['s2']}
High50c: {ta['recent_high']}  | Low50c: {ta['recent_low']}

=== VOLUME & CANDLE ===
Volume    : {ta['vol_label']} ({ta['vol_ratio']:.1f}x rata-rata)
Patterns  : {', '.join(ta['candle_patterns'])}

=== SINYAL TEKNIKAL ===
BULLISH ({len(ta['bull_signals'])} sinyal):
{bulls}

BEARISH ({len(ta['bear_signals'])} sinyal):
{bears}

BIAS TEKNIKAL: {ta['bias']} | Confidence raw: {ta['confidence_raw']:.1f}%

=== LEVEL ATR-BASED (REFERENSI) ===
Entry  : {ta['entry']} | TP : {ta['tp1']} | SL: {ta['sl']} | R:R ≈ {ta['rr']}

=== PARAMETER TRADING ===
Strategi : {strategy} | Mode : {mode} | Min R:R : 1:{rr_min}

=== INSTRUKSI ===
Berdasarkan data di atas, berikan analisis final dan trading signal.
Anda BOLEH memodifikasi level entry/TP/SL dari ATR-based jika ada alasan S/R yang lebih baik.
Pastikan R:R minimal 1:{rr_min}.

RESPONS HANYA JSON MURNI (tanpa markdown, tanpa komentar):
{{
  "signal": "BUY" atau "SELL",
  "entry": angka_float,
  "tp": angka_float,
  "sl": angka_float,
  "confidence": integer_0_100,
  "rr_ratio": "1:X.X",
  "summary": "Ringkasan 2 kalimat dengan angka spesifik dari data di atas",
  "reasons": [
    "Alasan 1 — sebutkan angka indikator spesifik",
    "Alasan 2 — sebutkan angka indikator spesifik",
    "Alasan 3 — sebutkan angka indikator spesifik",
    "Alasan 4 — sebutkan angka indikator spesifik",
    "Alasan 5 — sebutkan angka indikator spesifik"
  ],
  "indicators": [
    {{"label": "RSI(14): {ta['rsi14']}", "type": "{'bull' if ta['rsi14']<50 else 'bear'}"}},
    {{"label": "MACD Hist: {'Positif' if ta['macd_hist']>0 else 'Negatif'} {ta['macd_hist']:.4f}", "type": "{'bull' if ta['macd_hist']>0 else 'bear'}"}},
    {{"label": "ADX: {ta['adx']:.1f} | +DI {ta['plus_di']:.1f} -DI {ta['minus_di']:.1f}", "type": "{'bull' if ta['plus_di']>ta['minus_di'] else 'bear'}"}},
    {{"label": "Stoch K/D: {ta['stoch_k']:.0f}/{ta['stoch_d']:.0f}", "type": "{'bull' if ta['stoch_k']<50 else 'bear'}"}},
    {{"label": "BB %B: {ta['bb_pct_b']:.2f} | {ta['trend']}", "type": "{'bull' if ta['bb_pct_b']<0.4 else 'bear' if ta['bb_pct_b']>0.6 else 'neutral'}"}},
    {{"label": "Vol: {ta['vol_label']} | {', '.join(ta['candle_patterns'][:1])}", "type": "neutral"}}
  ],
  "market_context": "Konteks makroekonomi gold: DXY, inflasi, Fed policy, sentimen risiko saat ini"
}}"""


def parse_ai(content):
    content = content.strip()
    content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.MULTILINE)
    content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m: return json.loads(m.group())
        raise ValueError(f"Cannot parse JSON: {content[:300]}")


def call_openai_compat(api_key, cfg, prompt):
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role":"system","content":"Analis trading XAUUSD. Respons HANYA JSON valid."},
            {"role":"user","content":prompt},
        ],
        "temperature":0.2, "max_tokens":2000,
    }
    if cfg.get("json_mode"):
        payload["response_format"] = {"type":"json_object"}
    data = json.dumps(payload).encode()
    hdrs = {"Content-Type":"application/json",
            cfg["auth_header"]: cfg["auth_prefix"]+api_key}
    if cfg.get("extra_headers"): hdrs.update(cfg["extra_headers"])
    ctx = ssl.create_default_context()
    req = urllib.request.Request(cfg["url"], data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, context=ctx, timeout=90) as r:
        res = json.loads(r.read().decode())
    return parse_ai(res["choices"][0]["message"]["content"])


def call_anthropic(api_key, cfg, prompt):
    payload = {
        "model":cfg["model"], "max_tokens":2000,
        "system":"Analis trading XAUUSD. Respons HANYA JSON valid tanpa markdown.",
        "messages":[{"role":"user","content":prompt}],
    }
    hdrs = {"Content-Type":"application/json",
            "x-api-key":api_key,"anthropic-version":"2023-06-01"}
    ctx = ssl.create_default_context()
    req = urllib.request.Request(cfg["url"], data=json.dumps(payload).encode(), headers=hdrs, method="POST")
    with urllib.request.urlopen(req, context=ctx, timeout=90) as r:
        res = json.loads(r.read().decode())
    return parse_ai(res["content"][0]["text"])


def call_gemini(api_key, cfg, prompt):
    url = f"{cfg['url']}?key={api_key}"
    payload = {"contents":[{"parts":[{"text":prompt}]}],
               "generationConfig":{"temperature":0.2,"maxOutputTokens":2000}}
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                  headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=ctx, timeout=90) as r:
        res = json.loads(r.read().decode())
    return parse_ai(res["candidates"][0]["content"]["parts"][0]["text"])


def call_ai(ai_id, api_key, tf, strategy, mode, rr_min):
    if ai_id not in AI_PROVIDERS:
        raise ValueError(f"Unknown provider: {ai_id}")
    cfg = AI_PROVIDERS[ai_id]

    print(f"  [TA] Menghitung indikator teknikal {tf}...")
    ta = run_analysis(tf)
    if "error" in ta: raise RuntimeError(ta["error"])
    print(f"  [TA] OK — Bias={ta['bias']} RSI={ta['rsi14']} MACD={ta['macd_hist']:.4f} ADX={ta['adx']:.1f}")

    prompt = build_prompt(ta, strategy, mode, rr_min)

    print(f"  [AI] Calling {cfg['name']}...")
    if ai_id=="claude":
        result = call_anthropic(api_key, cfg, prompt)
    elif ai_id=="gemini":
        result = call_gemini(api_key, cfg, prompt)
    else:
        result = call_openai_compat(api_key, cfg, prompt)

    # Inject raw TA data ke response
    result["ta_data"] = {
        "price":ta["price"], "rsi14":ta["rsi14"], "macd_hist":ta["macd_hist"],
        "adx":ta["adx"], "atr14":ta["atr"], "ema20":ta["ema20"],
        "ema50":ta["ema50"], "ema200":ta["ema200"], "bb_upper":ta["bb_upper"],
        "bb_lower":ta["bb_lower"], "stoch_k":ta["stoch_k"], "vwap":ta["vwap"],
        "trend":ta["trend"], "vol_label":ta["vol_label"],
        "patterns":ta["candle_patterns"],
        "r1":ta["r1"],"s1":ta["s1"],"r2":ta["r2"],"s2":ta["s2"],
        "bull_count":len(ta["bull_signals"]),
        "bear_count":len(ta["bear_signals"]),
    }
    result["candles"] = ta["candles"]
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  HTTP SERVER
# ──────────────────────────────────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {fmt % args}")

    def cors(self):
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Access-Control-Allow-Methods","GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers","Content-Type")

    def do_OPTIONS(self):
        self.send_response(200); self.cors(); self.end_headers()

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/","/index.html"):
            self._file("static/index.html","text/html")
        elif p=="/health":
            self._json({"status":"ok","version":"3.0","numpy":HAS_NUMPY,
                        "cache":len(DATA_CACHE),"time":datetime.now().isoformat()})
        elif p=="/api/price":
            self._price()
        elif p=="/api/ta":
            qs = urllib.parse.parse_qs(self.path.split("?",1)[-1]) if "?" in self.path else {}
            self._ta(qs.get("tf",["H1"])[0])
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path=="/api/signal":     self._signal()
        elif self.path=="/api/validate": self._validate()
        elif self.path=="/api/clear-cache":
            DATA_CACHE.clear(); self._json({"cleared":True})
        else:
            self.send_response(404); self.end_headers()

    def _body(self):
        n = int(self.headers.get("Content-Length",0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _price(self):
        try:
            df = get_ohlcv("M5")
            c  = float(df["close"].iloc[-1])
            p  = float(df["close"].iloc[-2]) if len(df)>1 else c
            self._json({"price":round(c,2),"change":round(c-p,2),
                        "change_pct":round((c-p)/p*100,3) if p else 0,
                        "high":round(float(df["high"].max()),2),
                        "low":round(float(df["low"].min()),2)})
        except Exception as e:
            self._json({"error":str(e)},500)

    def _ta(self, tf):
        try:
            self._json(run_analysis(tf))
        except Exception as e:
            self._json({"error":str(e)},500)

    def _signal(self):
        try:
            b = self._body()
            ai_id   = b.get("ai_id","")
            api_key = b.get("api_key","").strip()
            tf      = b.get("timeframe","H1")
            strategy= b.get("strategy","technical")
            mode    = b.get("mode","moderate")
            rr_min  = b.get("rr_min","2")

            if not ai_id or not api_key:
                self._json({"error":"ai_id dan api_key diperlukan"},400); return

            print(f"\n{'─'*52}")
            print(f"  📡 Signal: AI={ai_id} TF={tf} Strategy={strategy} Mode={mode}")

            res = call_ai(ai_id, api_key, tf, strategy, mode, rr_min)
            res["ai_name"]        = AI_PROVIDERS[ai_id]["name"]
            res["timeframe"]      = tf
            res["generated_at"]   = datetime.now().strftime("%H:%M WIB")
            res["generated_date"] = datetime.now().strftime("%d %b %Y")

            print(f"  ✅ {res.get('signal','?')} entry={res.get('entry','?')} "
                  f"tp={res.get('tp','?')} sl={res.get('sl','?')} "
                  f"conf={res.get('confidence','?')}%")
            self._json(res)

        except urllib.error.HTTPError as e:
            eb = e.read().decode()
            print(f"  ✗ HTTP {e.code}: {eb[:200]}")
            try:
                ed = json.loads(eb)
                msg = ed.get("error",{}).get("message",eb[:150]) if isinstance(ed.get("error"),dict) else str(ed.get("error",e))
            except: msg = f"HTTP {e.code}: {eb[:150]}"
            self._json({"error":f"API Error: {msg}"},502)

        except urllib.error.URLError as e:
            print(f"  ✗ Network: {e}")
            self._json({"error":f"Koneksi gagal: {e.reason}"},503)

        except json.JSONDecodeError:
            self._json({"error":"Respons AI bukan JSON valid. Coba lagi."},500)

        except Exception as e:
            import traceback; traceback.print_exc()
            self._json({"error":str(e)},500)

    def _validate(self):
        try:
            b = self._body()
            ai,k = b.get("ai_id",""), b.get("api_key","").strip()
            fns = {
                "gpt4":    lambda x: x.startswith("sk-") and len(x)>20,
                "claude":  lambda x: x.startswith("sk-ant-") and len(x)>30,
                "gemini":  lambda x: len(x)>20,
                "deepseek":lambda x: x.startswith("sk-") and len(x)>20,
                "grok":    lambda x: x.startswith("xai-") and len(x)>20,
                "mistral": lambda x: len(x)>20,
            }
            ok = bool(k) and fns.get(ai, lambda x: len(x)>10)(k)
            self._json({"valid":ok,"message":"Format valid ✓" if ok else "Format API Key tidak sesuai"})
        except Exception as e:
            self._json({"valid":False,"message":str(e)},500)

    def _file(self, rel, ct):
        full = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)
        try:
            body = open(full,"rb").read()
            self.send_response(200)
            self.send_header("Content-Type",f"{ct}; charset=utf-8")
            self.cors()
            self.send_header("Content-Length",len(body))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_response(404); self.end_headers()

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.cors()
        self.send_header("Content-Length",len(body))
        self.end_headers()
        self.wfile.write(body)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    np_ver = ""
    if HAS_NUMPY:
        import numpy as _np, pandas as _pd
        np_ver = f"NumPy {_np.__version__} / Pandas {_pd.__version__}"
    else:
        np_ver = "TIDAK TERSEDIA — pip install numpy pandas"

    print(f"""
╔═══════════════════════════════════════════════════════════╗
║      🥇  XAUUSD SIGNAL PRO — Python Backend v3.0         ║
╠═══════════════════════════════════════════════════════════╣
║  Web App : http://localhost:{PORT}                        ║
║  Health  : http://localhost:{PORT}/health                 ║
║  Price   : http://localhost:{PORT}/api/price              ║
║  TA Data : http://localhost:{PORT}/api/ta?tf=H1           ║
╠═══════════════════════════════════════════════════════════╣
║  DATA ENGINE                                             ║
║  1. Yahoo Finance (live OHLCV via raw API)               ║
║  2. Stooq fallback (daily/weekly)                        ║
║  3. NumPy GBM synthetic (offline fallback)               ║
╠═══════════════════════════════════════════════════════════╣
║  INDICATORS (NumPy/Pandas)                               ║
║  RSI(7,14) · MACD · Bollinger · ATR · ADX/DI            ║
║  Stochastic · CCI · Williams%R · VWAP                   ║
║  EMA(8/20/50/200) · Pivot Points · Candle Patterns      ║
║  RSI Divergence · Volume Analysis · Trend Score         ║
╠═══════════════════════════════════════════════════════════╣
║  AI  : GPT-4o · Claude 3.5 · Gemini · DeepSeek          ║
║        Grok-2 · Mistral Large                            ║
╠═══════════════════════════════════════════════════════════╣
║  {np_ver:<55} ║
╚═══════════════════════════════════════════════════════════╝
  Ctrl+C untuk stop
""")
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  👋 Server dihentikan.")
        srv.server_close()


if __name__ == "__main__":
    main()
