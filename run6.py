import pandas as pd
import requests
import io
import os
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

def get_fred_data(series_ids):
    """Direct CSV fetch from FRED with browser-like headers."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    data_frames = {}
    for sid in series_ids:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
                # Ensure a clean single column named 'value'
                df = df.iloc[:, [0]].rename(columns={df.columns[0]: 'value'})
                data_frames[sid] = df
        except Exception:
            continue 
    return data_frames


def run_macro_analysis():
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # --- 1. DATA FETCHING (FRED) ---
    headers = {'User-Agent': 'Mozilla/5.0'}
    fred_series = {"GDP": "GDP", "T10Y2Y": "T10Y2Y"}
    fred_data = {}

    for name, sid in fred_series.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
                df = df.iloc[:, [0]].rename(columns={df.columns[0]: 'value'})
                fred_data[name] = df
        except Exception as e:
            return f"Error fetching {name} from FRED: {e}"

    # --- 2. DATA FETCHING (YAHOO FINANCE) ---
    try:
        # VTI = Total Market, ^GSPC = S&P 500, GC=F = Gold
        raw_data = yf.download(["VTI", "^GSPC", "GC=F"], period="15y", progress=False)
        
        # Robust column extraction (Handles both MultiIndex and Flat columns)
        if isinstance(raw_data.columns, pd.MultiIndex):
            closes = raw_data['Close']
        else:
            closes = raw_data[['Close']]
            
        # THE FIX: Synchronize dates and fill gaps to prevent 'NaN' in ratio
        closes = closes.ffill().dropna()
        
        if closes.empty:
            return "Error: No overlapping price data found."

        vti_series = closes['VTI']
        sp_price = float(closes['^GSPC'].iloc[-1])
        gold_price = float(closes['GC=F'].iloc[-1])
        
    except Exception as e:
        return f"Error fetching Yahoo Finance data: {e}"

    # --- 3. METRIC CALCULATIONS ---
    # A. S&P / Gold Ratio
    ratio = sp_price / gold_price

    # B. Buffett Indicator Z-Score
    gdp_daily = fred_data['GDP'].resample('D').ffill()
    # Align VTI (Market Cap proxy) with GDP
    common_idx = vti_series.index.intersection(gdp_daily.index)
    buffett_series = vti_series.loc[common_idx] / gdp_daily.loc[common_idx, 'value']
    
    # Use a 10-year rolling window (approx 2520 trading days)
    # min_periods allows it to work even if full 10 years isn't available
    current_val = buffett_series.iloc[-1]
    rolling_mean = buffett_series.rolling(window=2520, min_periods=500).mean().iloc[-1]
    rolling_std = buffett_series.rolling(window=2520, min_periods=500).std().iloc[-1]
    z_score = (current_val - rolling_mean) / rolling_std

    # C. Yield Curve Spread
    yield_spread = float(fred_data['T10Y2Y']['value'].iloc[-1])

    # --- 4. DECISION MATRIX ---
    # Priority 1: Extreme Valuation (Bubble)
    # Priority 2: Monetary/Recession Risk (Yield Curve)
    # Priority 3: Hard Asset Preference (S&P/Gold Ratio)
    
    if z_score > 2.0:
        regime, alloc = "EXCESSIVE BUBBLE", [20, 30, 50] # Equity, Gold, Cash
    elif yield_spread < 0:
        regime, alloc = "RECESSION WARNING", [10, 50, 40]
    elif ratio < 1.0:
        regime, alloc = "VALUE RECOVERY", [70, 20, 10]
    elif 1.8 < ratio <= 2.2:
        regime, alloc = "LATE CYCLE", [40, 20, 40]
    else:
        regime, alloc = "EARLY/MID CYCLE GROWTH", [60, 20, 20]

    # --- 5. FORMAT OUTPUT ---
    report = (
        f"\n{'='*55}\n"
        f"        2026 GLOBAL MACRO DASHBOARD ({current_date})\n"
        f"{'='*55}\n"
        f" [VALUATION] Buffett Z-Score:  {z_score:.2f}  (Extreme: > 2.0)\n"
        f" [LIQUIDITY] 10Y-2Y Spread:    {yield_spread:.2f}% (Inversion: < 0.0)\n"
        f" [CURRENCY]  S&P/Gold Ratio:   {ratio:.2f}  (Mean: ~1.5)\n"
        f"-------------------------------------------------------\n"
        f" MARKET REGIME: **{regime}**\n"
        f" RECOMMENDED ALLOCATION:\n"
        f"   > Equities (VTI/SPY):  {alloc[0]}%\n"
        f"   > Gold (Physical/GLD): {alloc[1]}%\n"
        f"   > Cash (T-Bills/USD):  {alloc[2]}%\n"
        f"{'='*55}\n"
        f"Note: Z-score is normalized against a 10-year rolling mean.\n"
    )
    
    return report

def send_telegram_message(message):
    """
    Sends a message to a Telegram chat using a bot.
    Reads the Bot Token and Chat ID from environment variables for security.
    """
    # Get secrets from environment variables
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("ERROR: Telegram Bot Token or Chat ID not found in environment variables.")
        print("Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to send a message.")
        # Fallback to printing the message to the console if secrets are not set
        print("\n--- Stock Analysis Results ---\n")
        print(message)
        return

    # Construct the API URL and send the request
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'Markdown'
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        print("Successfully sent message to Telegram.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send message to Telegram: {e}")


if __name__ == "__main__":
    # Returning and printing in a single line
    analysis = run_macro_analysis()
    send_telegram_message(analysis)
    print(analysis)