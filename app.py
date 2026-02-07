from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import os
from datetime import datetime, timedelta

app = Flask(__name__)

DATA_FILE = 'data.txt'
SALDO_FILE = 'saldo.txt'

def load_data():
    """Load data from text file - start with empty data"""
    if not os.path.exists(DATA_FILE):
        # Start with empty data
        default_data = []
        save_data(default_data)
        return default_data
    
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Ensure all IDs are numbers and sequential starting from 0
            for i, item in enumerate(data):
                item['id'] = i
            
            return data
    except:
        return []

def save_data(data):
    """Save data to text file"""
    # Sort data by ID to ensure consistency
    data.sort(key=lambda x: x['id'])
    
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_saldo():
    """Load saldo history from text file - start with 0"""
    if not os.path.exists(SALDO_FILE):
        # Start with 0 balance
        default_saldo = {
            "saldo_awal": 0,
            "transaksi": [
                {"tanggal": datetime.now().strftime("%Y-%m-%d"), "saldo": 0}
            ]
        }
        save_saldo(default_saldo)
        return default_saldo
    
    try:
        with open(SALDO_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {"saldo_awal": 0, "transaksi": [{"tanggal": datetime.now().strftime("%Y-%m-%d"), "saldo": 0}]}

def save_saldo(saldo_data):
    """Save saldo history to text file"""
    with open(SALDO_FILE, 'w', encoding='utf-8') as f:
        json.dump(saldo_data, f, indent=2)

def calculate_profit_percentage(modal, jumlah_profit):
    """Calculate profit percentage"""
    if modal == 0:
        return 0
    return round((jumlah_profit / modal) * 100, 2)

def calculate_total_saldo():
    """Calculate total current saldo"""
    data = load_data()
    saldo_data = load_saldo()
    
    total_profit = sum(item['jumlah_profit'] for item in data)
    saldo_akhir = saldo_data['saldo_awal'] + total_profit
    
    # Update saldo history with latest date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Check if today's entry exists
    today_exists = False
    for i, trans in enumerate(saldo_data['transaksi']):
        if trans['tanggal'] == today:
            saldo_data['transaksi'][i]['saldo'] = saldo_akhir
            today_exists = True
            break
    
    if not today_exists:
        # Add today's saldo
        saldo_data['transaksi'].append({
            "tanggal": today,
            "saldo": saldo_akhir
        })
    
    # Keep only last 30 days of history
    saldo_data['transaksi'] = saldo_data['transaksi'][-30:]
    
    save_saldo(saldo_data)
    
    return saldo_akhir

def get_saldo_history():
    """Get saldo history for chart"""
    saldo_data = load_saldo()
    return saldo_data['transaksi']

def generate_chart_data():
    """Generate data for saldo chart"""
    history = get_saldo_history()
    
    # Sort by date
    history.sort(key=lambda x: x['tanggal'])
    
    # Get last 10 entries
    last_entries = history[-10:] if len(history) > 10 else history
    
    labels = [item['tanggal'] for item in last_entries]
    data = [item['saldo'] for item in last_entries]
    
    return labels, data

@app.route('/')
def index():
    """Render main dashboard"""
    data = load_data()
    
    # Calculate summary statistics
    total_modal = sum(item['modal'] for item in data)
    total_profit = sum(item['jumlah_profit'] for item in data)
    total_profit_percentage = calculate_profit_percentage(total_modal, total_profit) if total_modal > 0 else 0
    
    # Calculate total saldo
    total_saldo = calculate_total_saldo()
    
    # Get chart data
    chart_labels, chart_data = generate_chart_data()
    
    # Calculate saldo change
    saldo_data = load_saldo()
    if len(saldo_data['transaksi']) >= 2:
        last_saldo = saldo_data['transaksi'][-1]['saldo']
        prev_saldo = saldo_data['transaksi'][-2]['saldo'] if len(saldo_data['transaksi']) > 1 else saldo_data['saldo_awal']
        saldo_change = last_saldo - prev_saldo
        saldo_change_percentage = round((saldo_change / prev_saldo) * 100, 2) if prev_saldo > 0 else 0
    else:
        saldo_change = 0
        saldo_change_percentage = 0
    
    # Get profitable trades count
    profitable_trades = len([item for item in data if item['status'] == 'profit'])
    
    return render_template('index.html', 
                         data=data, 
                         total_modal=total_modal,
                         total_profit=total_profit,
                         total_profit_percentage=total_profit_percentage,
                         total_saldo=total_saldo,
                         saldo_change=saldo_change,
                         saldo_change_percentage=saldo_change_percentage,
                         chart_labels=chart_labels,
                         chart_data=chart_data,
                         profitable_trades=profitable_trades,
                         now=datetime.now().strftime("%Y-%m-%d %H:%M"))

# ============== API ENDPOINTS ==============

@app.route('/api/saldo', methods=['GET'])
def get_saldo():
    """API to get saldo data"""
    try:
        total_saldo = calculate_total_saldo()
        chart_labels, chart_data = generate_chart_data()
        
        return jsonify({
            "success": True,
            "total_saldo": total_saldo,
            "chart_labels": chart_labels,
            "chart_data": chart_data
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/saldo/update', methods=['POST'])
def update_saldo():
    """API to update saldo awal"""
    try:
        data = request.get_json()
        if not data or 'saldo_awal' not in data:
            return jsonify({"success": False, "message": "Missing saldo_awal parameter"}), 400
        
        saldo_data = load_saldo()
        saldo_data['saldo_awal'] = float(data['saldo_awal'])
        
        # Update current saldo in transactions
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate total profit from trades
        trades = load_data()
        total_profit = sum(trade['jumlah_profit'] for trade in trades)
        current_saldo = saldo_data['saldo_awal'] + total_profit
        
        # Update or add today's transaction
        today_exists = False
        for i, trans in enumerate(saldo_data['transaksi']):
            if trans['tanggal'] == today:
                saldo_data['transaksi'][i]['saldo'] = current_saldo
                today_exists = True
                break
        
        if not today_exists:
            saldo_data['transaksi'].append({
                "tanggal": today,
                "saldo": current_saldo
            })
        
        save_saldo(saldo_data)
        
        return jsonify({
            "success": True, 
            "saldo_awal": saldo_data['saldo_awal'],
            "current_saldo": current_saldo
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/trades', methods=['GET'])
def get_trades():
    """API to get all trades"""
    try:
        data = load_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/trades/<int:trade_id>', methods=['GET'])
def get_trade(trade_id):
    """API to get single trade by ID"""
    try:
        data = load_data()
        
        # Convert trade_id to integer and ensure it's within bounds
        trade_id = int(trade_id)
        if trade_id < 0 or trade_id >= len(data):
            return jsonify({"success": False, "message": "Trade not found"}), 404
        
        for trade in data:
            if trade['id'] == trade_id:
                return jsonify(trade)
        
        return jsonify({"success": False, "message": "Trade not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/trades/add', methods=['POST'])
def add_trade():
    """API to add new trade"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['nama_koin', 'modal', 'status', 'jumlah_profit']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "message": f"Missing field: {field}"}), 400
        
        trades = load_data()
        
        # Generate new ID starting from 0
        if trades:
            new_id = max([item['id'] for item in trades]) + 1
        else:
            new_id = 0
        
        new_trade = {
            "id": new_id,
            "nama_koin": str(data['nama_koin']).upper().strip(),
            "modal": float(data['modal']),
            "status": str(data['status']).lower(),
            "jumlah_profit": float(data['jumlah_profit']),
            "persen_profit": calculate_profit_percentage(float(data['modal']), float(data['jumlah_profit'])),
            "tanggal": datetime.now().strftime("%Y-%m-%d")
        }
        
        # Validate status
        if new_trade['status'] not in ['profit', 'loss']:
            return jsonify({"success": False, "message": "Status must be either 'profit' or 'loss'"}), 400
        
        trades.append(new_trade)
        save_data(trades)
        
        # Update saldo history
        update_saldo_history()
        
        return jsonify({"success": True, "trade": new_trade})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

def update_saldo_history():
    """Update saldo history with current total"""
    try:
        saldo_data = load_saldo()
        trades = load_data()
        
        total_profit = sum(item['jumlah_profit'] for item in trades)
        current_saldo = saldo_data['saldo_awal'] + total_profit
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Check if today's entry exists
        today_exists = False
        for i, trans in enumerate(saldo_data['transaksi']):
            if trans['tanggal'] == today:
                saldo_data['transaksi'][i]['saldo'] = current_saldo
                today_exists = True
                break
        
        if not today_exists:
            saldo_data['transaksi'].append({
                "tanggal": today,
                "saldo": current_saldo
            })
        
        # Keep only last 30 days of history
        saldo_data['transaksi'] = saldo_data['transaksi'][-30:]
        
        save_saldo(saldo_data)
    except Exception as e:
        print(f"Error updating saldo history: {e}")

@app.route('/api/trades/<int:trade_id>', methods=['PUT'])
def update_trade(trade_id):
    """API to update trade"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['nama_koin', 'modal', 'status', 'jumlah_profit']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "message": f"Missing field: {field}"}), 400
        
        trades = load_data()
        
        # Convert trade_id to integer
        trade_id = int(trade_id)
        
        for i, trade in enumerate(trades):
            if trade['id'] == trade_id:
                updated_trade = {
                    "id": trade_id,
                    "nama_koin": str(data['nama_koin']).upper().strip(),
                    "modal": float(data['modal']),
                    "status": str(data['status']).lower(),
                    "jumlah_profit": float(data['jumlah_profit']),
                    "persen_profit": calculate_profit_percentage(float(data['modal']), float(data['jumlah_profit'])),
                    "tanggal": trade.get('tanggal', datetime.now().strftime("%Y-%m-%d"))
                }
                
                # Validate status
                if updated_trade['status'] not in ['profit', 'loss']:
                    return jsonify({"success": False, "message": "Status must be either 'profit' or 'loss'"}), 400
                
                trades[i] = updated_trade
                save_data(trades)
                
                # Update saldo history
                update_saldo_history()
                
                return jsonify({"success": True, "trade": updated_trade})
        
        return jsonify({"success": False, "message": "Trade not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/trades/<int:trade_id>', methods=['DELETE'])
def delete_trade(trade_id):
    """API to delete trade"""
    try:
        trades = load_data()
        
        # Convert trade_id to integer
        trade_id = int(trade_id)
        
        # Find and remove the trade
        trade_found = False
        for i, trade in enumerate(trades):
            if trade['id'] == trade_id:
                del trades[i]
                trade_found = True
                break
        
        if not trade_found:
            return jsonify({"success": False, "message": "Trade not found"}), 404
        
        # Reassign IDs starting from 0 to maintain sequence
        for i, trade in enumerate(trades):
            trade['id'] = i
        
        save_data(trades)
        
        # Update saldo history
        update_saldo_history()
        
        return jsonify({"success": True, "message": "Trade deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_data():
    """API to reset all data to empty"""
    try:
        # Reset trades data to empty
        save_data([])
        
        # Reset saldo data to 0
        default_saldo = {
            "saldo_awal": 0,
            "transaksi": [
                {"tanggal": datetime.now().strftime("%Y-%m-%d"), "saldo": 0}
            ]
        }
        
        save_saldo(default_saldo)
        
        return jsonify({"success": True, "message": "All data reset to empty"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')