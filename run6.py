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
    
    # 1. FETCH ECONOMIC DATA (FRED)
    # GDP (Quarterly), T10Y2Y (Daily Yield Spread)
    fred_data = get_fred_data(["GDP", "T10Y2Y"])
    if "GDP" not in fred_data or "T10Y2Y" not in fred_data:
        return "Error: Could not connect to FRED for GDP or Yield Spread."

    # 2. FETCH MARKET DATA (Yahoo)
    try:
        # Use VTI (Total Stock Market) as it's more reliable than ^W5000 index
        # We also download S&P 500 and Gold
        tickers = yf.download(["VTI", "^GSPC", "GC=F"], period="10y", interval="1d", progress=False)
        
        # FIX: Flatten MultiIndex columns (e.g., ('Close', 'VTI') -> 'Close')
        # This is the most common reason scripts fail in 2025/2026
        if isinstance(tickers.columns, pd.MultiIndex):
            # We want the 'Close' prices for our calculation
            closes = tickers['Close']
        else:
            closes = tickers[['Close']]

        if closes.empty:
            return "Error: No price data returned from Yahoo Finance."
            
        mkt_proxy = closes['VTI'].dropna()
        sp_price = closes['^GSPC'].iloc[-1]
        gold_price = closes['GC=F'].iloc[-1]
    except Exception as e:
        return f"Error fetching market data: {e}"

    # 3. CALCULATE METRICS
    # A. S&P/Gold Ratio
    ratio = sp_price / gold_price

    # B. Buffett Indicator Z-Score (Total Market / GDP)
    gdp_daily = fred_data['GDP'].resample('D').ffill()
    
    # Align dates for overlap
    common_idx = mkt_proxy.index.intersection(gdp_daily.index)
    buffett_series = mkt_proxy.loc[common_idx] / gdp_daily.loc[common_idx, 'value']
    
    if len(buffett_series) < 500:
        return "Error: Insufficient historical overlap for Z-Score calculation."

    current_val = buffett_series.iloc[-1]
    rolling_mean = buffett_series.rolling(window=2520, min_periods=500).mean().iloc[-1]
    rolling_std = buffett_series.rolling(window=2520, min_periods=500).std().iloc[-1]
    z_score = (current_val - rolling_mean) / rolling_std

    # C. Yield Curve
    yield_spread = fred_data['T10Y2Y']['value'].iloc[-1]

    # 4. DECISION MATRIX
    if z_score > 2.0:
        regime, alloc = "EXCESSIVE BUBBLE", [20, 30, 50]
    elif yield_spread < 0:
        regime, alloc = "RECESSION WARNING", [10, 50, 40]
    elif ratio < 1.0:
        regime, alloc = "VALUE RECOVERY", [70, 20, 10]
    else:
        regime, alloc = "GROWTH / MOMENTUM", [60, 20, 20]

    # 5. ASSEMBLE MULTI-LINE REPORT
    report = (
        f"\n{'='*55}\n"
        f"        2026 GLOBAL MACRO DASHBOARD ({current_date})\n"
        f"{'='*55}\n"
        f" [VALUATION] Buffett Z-Score:  {z_score:.2f}  (Extreme: > 2.0)\n"
        f" [LIQUIDITY] 10Y-2Y Spread:    {yield_spread:.2f}% (Warning: < 0.0)\n"
        f" [CURRENCY]  S&P/Gold Ratio:   {ratio:.2f}  (Neutral: ~1.5)\n"
        f"-------------------------------------------------------\n"
        f" MARKET REGIME: **{regime}**\n"
        f" RECOMMENDED ALLOCATION:\n"
        f"   > Equities (VTI/SPY):  {alloc[0]}%\n"
        f"   > Gold (Physical/GLD): {alloc[1]}%\n"
        f"   > Cash (T-Bills/USD):  {alloc[2]}%\n"
        f"{'='*55}\n"
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