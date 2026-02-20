import yfinance as yf
import pandas as pd
from openai import OpenAI
import os
from datetime import datetime, timedelta
import json
import time
from datetime import timezone
import threading
import telebot
from flask import Flask, jsonify

# ─────────────────────────────────────────
#  Config
# ─────────────────────────────────────────
SYMBOL             = "GC=F"       # Gold Futures (XAUUSD proxy)
INTERVAL           = "15m"
CANDLE_LIMIT       = 60
SIGNAL_INTERVAL    = 15 * 60      # Kirim sinyal setiap 15 menit (detik)
FLASK_PORT         = int(os.getenv("PORT", 5000))

DEEPSEEK_API_KEY   = "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"
TELEGRAM_BOT_TOKEN = "8142186242:AAHB9Z05A-QZ6W5G2Mbzr0XR_8FGgSDCtAY"
TELEGRAM_CHAT_ID   = -1003899518610

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# ─────────────────────────────────────────
#  State (thread-safe)
# ─────────────────────────────────────────
START_TIME = datetime.now(timezone.utc)
bot_status = {
    "status"        : "starting",   # starting | online | error
    "last_signal_at": None,
    "last_price"    : None,
    "signals_sent"  : 0,
    "last_error"    : None,
}
state_lock = threading.Lock()


# ─────────────────────────────────────────
#  Flask — Status Page
# ─────────────────────────────────────────
app = Flask(__name__)

def get_uptime() -> str:
    delta   = datetime.now(timezone.utc) - START_TIME
    days    = delta.days
    hours   = delta.seconds // 3600
    minutes = (delta.seconds % 3600) // 60
    seconds = delta.seconds % 60
    parts   = []
    if days:    parts.append(f"{days}d")
    if hours:   parts.append(f"{hours}h")
    if minutes: parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)

@app.route("/")
def index():
    with state_lock:
        status      = bot_status["status"]
        last_signal = bot_status["last_signal_at"] or "—"
        last_price  = f"${bot_status['last_price']:,.2f}" if bot_status["last_price"] else "—"
        signals_cnt = bot_status["signals_sent"]
        last_error  = bot_status["last_error"] or "None"

    color_map = {"online": "#00e676", "error": "#ff1744", "starting": "#ffea00"}
    icon_map  = {"online": "🟢", "error": "🔴", "starting": "🟡"}
    color     = color_map.get(status, "#ffffff")
    icon      = icon_map.get(status, "⚪")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <meta http-equiv="refresh" content="30"/>
  <title>XAUUSD Bot Status</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{
      background: #0d1117;
      color: #e6edf3;
      font-family: 'Segoe UI', system-ui, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }}
    .card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 16px;
      padding: 40px 48px;
      max-width: 560px;
      width: 100%;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }}
    .header {{
      text-align: center;
      margin-bottom: 32px;
    }}
    .header h1 {{
      font-size: 1.8rem;
      font-weight: 700;
      letter-spacing: 1px;
    }}
    .header p {{
      color: #8b949e;
      margin-top: 6px;
      font-size: 0.9rem;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 999px;
      padding: 6px 20px;
      font-size: 1rem;
      font-weight: 600;
      color: {color};
      margin: 14px auto 0;
    }}
    .dot {{
      width: 10px; height: 10px;
      border-radius: 50%;
      background: {color};
      box-shadow: 0 0 8px {color};
      animation: pulse 1.5s infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50%       {{ opacity: 0.3; }}
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-top: 28px;
    }}
    .stat {{
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 16px 20px;
    }}
    .stat-label {{
      font-size: 0.72rem;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      margin-bottom: 6px;
    }}
    .stat-value {{
      font-size: 1.05rem;
      font-weight: 600;
      color: #e6edf3;
    }}
    .full-width {{ grid-column: 1 / -1; }}
    .error-box {{
      background: #2d1215;
      border: 1px solid #f8514966;
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 0.83rem;
      color: #ff7b72;
      margin-top: 20px;
      word-break: break-all;
    }}
    .footer {{
      text-align: center;
      color: #484f58;
      font-size: 0.76rem;
      margin-top: 28px;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <h1>📊 XAUUSD AI Bot</h1>
      <p>Powered by DeepSeek Reasoner + Yahoo Finance</p>
      <div class="badge">
        <span class="dot"></span>
        {icon} {status.upper()}
      </div>
    </div>

    <div class="grid">
      <div class="stat">
        <div class="stat-label">⏱ Uptime</div>
        <div class="stat-value">{get_uptime()}</div>
      </div>
      <div class="stat">
        <div class="stat-label">📨 Signals Sent</div>
        <div class="stat-value">{signals_cnt}</div>
      </div>
      <div class="stat">
        <div class="stat-label">💰 Last Price</div>
        <div class="stat-value">{last_price}</div>
      </div>
      <div class="stat">
        <div class="stat-label">🕒 Last Signal</div>
        <div class="stat-value" style="font-size:0.88rem">{last_signal}</div>
      </div>
      <div class="stat full-width">
        <div class="stat-label">📡 Telegram Channel</div>
        <div class="stat-value">{TELEGRAM_CHAT_ID}</div>
      </div>
    </div>

    {"<div class='error-box'>⚠️ Last Error: " + last_error + "</div>" if last_error != "None" else ""}

    <div class="footer">
      🔄 Auto-refresh setiap 30 detik &nbsp;|&nbsp;
      🚀 Started {START_TIME.strftime('%Y-%m-%d %H:%M:%S')} UTC
    </div>
  </div>
</body>
</html>"""
    return html


@app.route("/health")
def health():
    """Endpoint JSON untuk health check / monitoring eksternal."""
    with state_lock:
        return jsonify({
            "status"        : bot_status["status"],
            "uptime"        : get_uptime(),
            "signals_sent"  : bot_status["signals_sent"],
            "last_signal_at": bot_status["last_signal_at"],
            "last_price"    : bot_status["last_price"],
            "last_error"    : bot_status["last_error"],
            "started_at"    : START_TIME.strftime("%Y-%m-%d %H:%M:%S UTC"),
        })


# ─────────────────────────────────────────
#  Fetch Gold Data
# ─────────────────────────────────────────
def fetch_gold_data():
    print("📡 Fetching gold data from Yahoo Finance...")
    ticker  = yf.Ticker(SYMBOL)
    periods = ["60d", "30d", "7d", "5d"]

    for attempt in range(3):
        for period in periods:
            try:
                df = ticker.history(period=period, interval=INTERVAL, prepost=False, actions=False)
                if not df.empty and len(df) >= CANDLE_LIMIT:
                    df.index = df.index.tz_localize(None)
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    df = df.tail(CANDLE_LIMIT)

                    latest_time     = df['timestamp'].iloc[-1]
                    now             = datetime.now(timezone.utc)
                    latest_time_utc = (
                        latest_time.replace(tzinfo=timezone.utc)
                        if latest_time.tzinfo is None
                        else latest_time.astimezone(timezone.utc)
                    )

                    print(f"✅ Rows: {len(df)}, Latest: {latest_time}, Close: {df['close'].iloc[-1]:.2f}")
                    if (now - latest_time_utc).days > 1:
                        print("⚠️  Using historical data (market closed)")
                    return df
                time.sleep(3)
            except Exception as e:
                print(f"Attempt {attempt+1}, period {period} failed: {e}")
                time.sleep(5)

    raise ValueError("Failed to fetch enough data.")


# ─────────────────────────────────────────
#  Format Telegram Message
# ─────────────────────────────────────────
ACTION_EMOJI = {
    "buy":  "🟢 BUY",
    "sell": "🔴 SELL",
    "hold": "🟡 HOLD",
}

def confidence_bar(conf: int) -> str:
    filled = round(conf / 10)
    return "█" * filled + "░" * (10 - filled)

def format_message(ai: dict, current_close: float, ts: str) -> str:
    st_action = ACTION_EMOJI.get(ai["short_term_action"].lower(), ai["short_term_action"].upper())
    lt_action = ACTION_EMOJI.get(ai["long_term_action"].lower(), ai["long_term_action"].upper())
    conf      = int(ai["confidence"])

    return (
        "╔══════════════════════════╗\n"
        "       📊 *XAUUSD AI SIGNAL*\n"
        "╚══════════════════════════╝\n\n"
        f"🕒 *Time (UTC):*  `{ts}`\n"
        f"💰 *Current Price:*  `${current_close:,.2f}`\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚡ *SHORT TERM SIGNAL*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📌 Action      : *{st_action}*\n"
        f"  🎯 Take Profit : `${ai['short_term_tp']:,.2f}`\n"
        f"  🛡️ Stop Loss   : `${ai['short_term_sl']:,.2f}`\n"
        f"  💬 Reason      : _{ai['short_term_reason']}_\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🔭 *LONG TERM SIGNAL*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  📌 Action      : *{lt_action}*\n"
        f"  🎯 Take Profit : `${ai['long_term_tp']:,.2f}`\n"
        f"  🛡️ Stop Loss   : `${ai['long_term_sl']:,.2f}`\n"
        f"  💬 Reason      : _{ai['long_term_reason']}_\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🔮 *PRICE PROJECTION*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  ⏱️ After 15 min  : `${ai['price_after_15m']:,.2f}`\n"
        f"  ⏱️ After 1 hour  : `${ai['price_after_1h']:,.2f}`\n"
        f"  ⏱️ After 4 hours : `${ai['price_after_4h']:,.2f}`\n"
        f"  ⏱️ After 1 day   : `${ai['price_after_1d']:,.2f}`\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 *AI Confidence:* `{conf}%`\n"
        f"  `[{confidence_bar(conf)}]`\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "⚠️ _This is AI-generated analysis, not financial advice._"
    )


# ─────────────────────────────────────────
#  Send to Telegram
# ─────────────────────────────────────────
def send_to_telegram(message: str):
    bot.send_message(
        chat_id    = TELEGRAM_CHAT_ID,
        text       = message,
        parse_mode = "Markdown"
    )
    print("✅ Signal sent to Telegram!")


# ─────────────────────────────────────────
#  Bot Loop — berjalan di thread background
# ─────────────────────────────────────────
def bot_loop():
    with state_lock:
        bot_status["status"] = "online"

    print(f"🤖 Bot loop started — interval: {SIGNAL_INTERVAL // 60} menit")

    while True:
        try:
            df            = fetch_gold_data()
            data_str      = df.to_string(index=False)
            current_close = df['close'].iloc[-1]

            prompt = f"""You are a world-class ICT/SMC institutional trader with 15+ years experience trading XAUUSD.

Current price: {current_close:.2f}

Here is the latest 15m chart data (last {len(df)} candles):

{data_str}

Perform deep multi-timeframe analysis using:
- Market Structure (BOS, CHOCH, HH/HL, LH/LL)
- Order Blocks, Breaker Blocks, Mitigation Blocks
- Fair Value Gaps (FVGs) and Imbalances
- Liquidity Sweeps, Judas Swings, Stop Hunts
- Displacement and strong impulsive moves
- Key Support & Resistance zones
- Volume confirmation and momentum

Rules:
- Be extremely selective — only high-conviction signals (minimum 70%)
- Prioritize London/NY session behavior if active
- Short-term = next 15–60 minutes
- Long-term = next 2–8 hours

Then provide realistic price projections at these exact horizons:
- After 15 minutes
- After 1 hour
- After 4 hours
- After 1 day

Output ONLY valid JSON — nothing else:
{{
  "short_term_action": "buy" or "sell" or "hold",
  "short_term_tp": float,
  "short_term_sl": float,
  "short_term_reason": "short powerful sentence",
  "long_term_action": "buy" or "sell" or "hold",
  "long_term_tp": float,
  "long_term_sl": float,
  "long_term_reason": "short powerful sentence",
  "price_after_15m": float,
  "price_after_1h": float,
  "price_after_4h": float,
  "price_after_1d": float,
  "confidence": 0-100
}}
"""

            client   = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model           = "deepseek-reasoner",
                messages        = [{"role": "user", "content": prompt}],
                response_format = {"type": "json_object"},
                temperature     = 0.2
            )

            ai  = json.loads(response.choices[0].message.content)
            ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            msg = format_message(ai, current_close, ts)

            send_to_telegram(msg)

            with state_lock:
                bot_status["status"]         = "online"
                bot_status["last_signal_at"] = ts
                bot_status["last_price"]     = current_close
                bot_status["signals_sent"]  += 1
                bot_status["last_error"]     = None

        except Exception as e:
            err = str(e)
            print(f"❌ Bot error: {err}")
            with state_lock:
                bot_status["status"]     = "error"
                bot_status["last_error"] = err

            try:
                bot.send_message(
                    chat_id    = TELEGRAM_CHAT_ID,
                    text       = f"⚠️ *XAUUSD Bot Error*\n\n`{err}`",
                    parse_mode = "Markdown"
                )
            except Exception:
                pass

        print(f"💤 Sleeping {SIGNAL_INTERVAL // 60} menit...\n")
        time.sleep(SIGNAL_INTERVAL)


# ─────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Jalankan bot loop di background thread
    threading.Thread(target=bot_loop, daemon=True).start()

    # Jalankan Flask di main thread
    print(f"🌐 Status page  →  http://0.0.0.0:{FLASK_PORT}")
    print(f"🔍 Health check →  http://0.0.0.0:{FLASK_PORT}/health")
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False)
