import os
import re
import requests
import yfinance as yf
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for

app = Flask(__name__)
app.secret_key = 'rahasia'  # Ganti dengan kunci rahasia yang aman

# ================== KONFIGURASI AI ==================
AI_CONFIG = {
    'deepseek': {
        'name': 'DeepSeek',
        'endpoint': 'https://api.deepseek.com/v1/chat/completions',
        'model': 'deepseek-chat',
        'headers': lambda api_key: {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        'payload': lambda prompt, model: {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 300
        }
    },
    'claude': {
        'name': 'Claude (Anthropic)',
        'endpoint': 'https://api.anthropic.com/v1/messages',
        'model': 'claude-3-haiku-20240307',
        'headers': lambda api_key: {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        },
        'payload': lambda prompt, model: {
            'model': model,
            'max_tokens': 300,
            'messages': [{'role': 'user', 'content': prompt}]
        }
    },
    'grok': {
        'name': 'Grok (xAI)',
        'endpoint': 'https://api.x.ai/v1/chat/completions',  # endpoint asumsi, bisa disesuaikan
        'model': 'grok-1',
        'headers': lambda api_key: {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        'payload': lambda prompt, model: {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 300
        }
    },
    'openai': {
        'name': 'OpenAI',
        'endpoint': 'https://api.openai.com/v1/chat/completions',
        'model': 'gpt-3.5-turbo',
        'headers': lambda api_key: {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        },
        'payload': lambda prompt, model: {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0.7,
            'max_tokens': 300
        }
    }
}

# ================== FUNGSI AMBIL DATA XAUUSD ==================
def get_xauusd_data():
    """Mengambil data terkini XAUUSD dari Yahoo Finance"""
    ticker = yf.Ticker("XAUUSD=X")  # Spot XAUUSD
    data = ticker.history(period="5d", interval="1d")  # 5 hari terakhir
    if data.empty:
        # Fallback ke gold futures jika spot tidak tersedia
        ticker = yf.Ticker("GC=F")
        data = ticker.history(period="5d", interval="1d")
    
    if data.empty:
        return None
    
    latest = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else latest
    return {
        'date': latest.name.strftime('%Y-%m-%d'),
        'open': latest['Open'],
        'high': latest['High'],
        'low': latest['Low'],
        'close': latest['Close'],
        'prev_close': prev['Close'],
        'data': data
    }

# ================== AI SEDERHANA (MOVING AVERAGE CROSSOVER) ==================
def simple_ai_analysis(data_dict):
    df = data_dict['data']
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    last_ma50 = df['MA50'].iloc[-1]
    last_ma200 = df['MA200'].iloc[-1]
    prev_ma50 = df['MA50'].iloc[-2] if len(df) > 1 else last_ma50
    prev_ma200 = df['MA200'].iloc[-2] if len(df) > 1 else last_ma200

    current_price = data_dict['close']
    
    if prev_ma50 <= prev_ma200 and last_ma50 > last_ma200:
        signal = "BUY"
        entry = round(current_price, 2)
        tp = round(entry * 1.01, 2)    # +1%
        sl = round(entry * 0.995, 2)   # -0.5%
        reason = "Golden cross: MA50 melintasi di atas MA200, mengindikasikan tren bullish."
    elif prev_ma50 >= prev_ma200 and last_ma50 < last_ma200:
        signal = "SELL"
        entry = round(current_price, 2)
        tp = round(entry * 0.99, 2)    # -1%
        sl = round(entry * 1.005, 2)   # +0.5%
        reason = "Death cross: MA50 melintasi di bawah MA200, mengindikasikan tren bearish."
    else:
        signal = "HOLD"
        entry = "-"
        tp = "-"
        sl = "-"
        reason = "Tidak ada crossover signifikan. Posisi moving average masih sejajar atau konsolidasi."
    
    return {
        'signal': signal,
        'entry': entry,
        'tp': tp,
        'sl': sl,
        'reason': reason
    }

# ================== FUNGSI MEMANGGIL API AI ==================
def call_ai_api(ai_choice, api_key, prompt):
    """Memanggil API AI berdasarkan pilihan dan mengembalikan teks respons"""
    config = AI_CONFIG.get(ai_choice)
    if not config:
        return None, "Pilihan AI tidak valid"
    
    headers = config['headers'](api_key)
    payload = config['payload'](prompt, config['model'])
    
    try:
        response = requests.post(
            config['endpoint'],
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # Ekstrak teks dari respons sesuai struktur masing-masing API
        if ai_choice == 'claude':
            # Claude: content adalah list of objects dengan field text
            content_blocks = data.get('content', [])
            if content_blocks and isinstance(content_blocks, list):
                return content_blocks[0].get('text', ''), None
            else:
                return None, "Respons Claude tidak mengandung konten teks"
        else:
            # OpenAI, DeepSeek, Grok: struktur choices[0].message.content
            choices = data.get('choices', [])
            if choices and len(choices) > 0:
                message = choices[0].get('message', {})
                return message.get('content', ''), None
            else:
                return None, "Respons AI tidak mengandung pilihan yang valid"
    
    except requests.exceptions.Timeout:
        return None, "Permintaan timeout. Coba lagi nanti."
    except requests.exceptions.RequestException as e:
        return None, f"Kesalahan koneksi: {str(e)}"
    except ValueError as e:
        return None, f"Kesalahan parsing JSON: {str(e)}"
    except Exception as e:
        return None, f"Kesalahan tidak terduga: {str(e)}"

# ================== PARSING RESPONS AI KE FORMAT SINYAL ==================
def parse_ai_response(text):
    """Mengubah teks respons AI menjadi dictionary sinyal"""
    # Regex untuk menangkap field
    signal_match = re.search(r"Sinyal:\s*(BUY|SELL|HOLD)", text, re.IGNORECASE)
    entry_match = re.search(r"Entry:\s*([\d,.]+)", text)
    tp_match = re.search(r"TP:\s*([\d,.]+)", text)
    sl_match = re.search(r"SL:\s*([\d,.]+)", text)
    reason_match = re.search(r"Alasan:\s*(.+)", text, re.DOTALL)
    
    signal = signal_match.group(1).upper() if signal_match else "HOLD"
    entry = entry_match.group(1).replace(',', '') if entry_match else "-"
    tp = tp_match.group(1).replace(',', '') if tp_match else "-"
    sl = sl_match.group(1).replace(',', '') if sl_match else "-"
    reason = reason_match.group(1).strip() if reason_match else "Tidak ada alasan yang diberikan."
    
    return {
        'signal': signal,
        'entry': entry,
        'tp': tp,
        'sl': sl,
        'reason': reason
    }

# ================== ROUTES ==================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ai_choice = request.form.get('ai_choice')
        api_key = request.form.get('api_key', '').strip()
        
        # Validasi input
        if not ai_choice:
            flash('Pilih AI terlebih dahulu.')
            return redirect(url_for('index'))
        
        # AI Sederhana tidak perlu API key
        if ai_choice != 'simple' and not api_key:
            flash('API Key diperlukan untuk AI yang dipilih.')
            return redirect(url_for('index'))
        
        # Ambil data XAUUSD
        data = get_xauusd_data()
        if data is None:
            flash('Gagal mengambil data harga. Coba lagi nanti.')
            return redirect(url_for('index'))
        
        # Jika AI Sederhana
        if ai_choice == 'simple':
            result = simple_ai_analysis(data)
            return render_template('result.html', result=result, ai_name="AI Sederhana (MA Crossover)")
        
        # Siapkan prompt untuk AI
        price_data = f"""
        Tanggal: {data['date']}
        Open: {data['open']:.2f}
        High: {data['high']:.2f}
        Low: {data['low']:.2f}
        Close: {data['close']:.2f}
        Previous Close: {data['prev_close']:.2f}
        """
        
        prompt = f"""
        Anda adalah seorang analis trading forex yang ahli. Berdasarkan data harga XAUUSD (Emas) berikut:
        {price_data}
        
        Berikan rekomendasi trading yang mencakup:
        - Sinyal (BUY, SELL, atau HOLD)
        - Harga Entry
        - Take Profit (TP)
        - Stop Loss (SL)
        - Alasan analisis singkat (berdasarkan teknikal dan/atau fundamental)
        
        Format jawaban:
        Sinyal: [BUY/SELL/HOLD]
        Entry: [angka]
        TP: [angka]
        SL: [angka]
        Alasan: [penjelasan]
        
        Pastikan angka entry, TP, SL dalam format desimal (contoh: 1950.50).
        """
        
        # Panggil API AI
        response_text, error = call_ai_api(ai_choice, api_key, prompt)
        if error:
            flash(f'Kesalahan dari {AI_CONFIG[ai_choice]["name"]}: {error}')
            return redirect(url_for('index'))
        
        if not response_text:
            flash('Respons AI kosong.')
            return redirect(url_for('index'))
        
        # Parsing respons
        result = parse_ai_response(response_text)
        return render_template('result.html', result=result, ai_name=AI_CONFIG[ai_choice]["name"])
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)