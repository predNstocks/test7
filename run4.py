import pandas as pd
import requests
import io
import os
import yfinance as yf
from datetime import datetime

def get_fred_safe(series_id):
    """Fetches data from FRED using a browser header to avoid 404s."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    df = pd.read_csv(io.StringIO(response.text), index_col=0, parse_dates=True)
    return df.iloc[:, 0]

def run_analysis():
    # 1. Fetch market data
    sp_data = yf.Ticker("^GSPC").history(period="5d")['Close']
    gold_data = yf.Ticker("GC=F").history(period="5d")['Close']
    
    # 2. Extract scalars (Safe extraction using .iloc[-1])
    current_sp = float(sp_data.iloc[-1])
    current_gold = float(gold_data.iloc[-1])
    ratio = current_sp / current_gold
    
    # 3. Macro logic - Decision Matrix
    if ratio > 2.2:
        regime, alloc = "EQUITY BUBBLE", [10, 40, 50]
    elif ratio < 0.8:
        regime, alloc = "MONETARY CRISIS", [20, 70, 10]
    elif 1.6 < ratio <= 2.2:
        regime, alloc = "LATE CYCLE", [40, 20, 40]
    else:
        regime, alloc = "EARLY/MID CYCLE", [60, 20, 20]

    # 4. Construct the multi-line string
    report = (
        f"{'='*40}\n"
        f" GLOBAL MACRO REPORT: {datetime.now().strftime('%Y-%m-%d')}\n"
        f"{'='*40}\n"
        f" MARKET PRICES:\n"
        f"  S&P 500 Index:   {current_sp:,.2f}\n"
        f"  Gold (per oz):   ${current_gold:,.2f}\n"
        f"  S&P/Gold Ratio:  {ratio:.2f}\n"
        f"\n REGIME ANALYSIS:\n"
        f"  Current State:   **{regime}**\n"
        f"\n TARGET ALLOCATION:\n"
        f"  [+] S&P 500:     {alloc[0]}%\n"
        f"  [+] Gold:        {alloc[1]}%\n"
        f"  [+] Cash:        {alloc[2]}%\n"
        f"{'='*40}"
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


def run_analysis():
    print("--- 2026 Global Macro Snapshot ---")
    
    # 1. FETCH DATA (Using Ticker.history for guaranteed Series output)
    print("Syncing market prices...")
    sp_ticker = yf.Ticker("^GSPC")
    gold_ticker = yf.Ticker("GC=F")
    
    # Get last 5 days to ensure we have data even if market is closed today
    sp_data = sp_ticker.history(period="5d")['Close']
    gold_data = gold_ticker.history(period="5d")['Close']
    
    # 2. EXTRACT SCALARS SAFELY
    # .iloc[-1] on a Series is a number. We add float() just to be sure.
    current_sp = float(sp_data.iloc[-1])
    current_gold = float(gold_data.iloc[-1])
    ratio = current_sp / current_gold
    
    # 3. MACRO INDICATORS
    # Buffett Indicator logic: Market Cap / GDP
    # Since we can't get total Market Cap easily, we use S&P 500 as a proxy
    print("Fetching GDP data...")
    gdp_series = get_fred_safe("GDP")
    current_gdp = float(gdp_series.iloc[-1])
    
    # 4. THE DECISION MATRIX (2026 Calibration)
    # We use the S&P/Gold ratio as the 'Valuation Compass'
    if ratio > 2.2:
        regime, alloc = "EQUITY BUBBLE", [10, 40, 50] # 10% SPY, 40% Gold, 50% Cash
    elif ratio < 0.8:
        regime, alloc = "MONETARY CRISIS", [20, 70, 10] # Hard Assets heavy
    elif 1.6 < ratio <= 2.2:
        regime, alloc = "LATE CYCLE", [40, 20, 40] # Defensive Growth
    else:
        regime, alloc = "EARLY/MID CYCLE", [60, 20, 20] # Standard Bull

    # 5. OUTPUT
    print(f"\nRESULTS FOR {datetime.now().strftime('%Y-%m-%d')}")
    print(f"S&P 500:        {current_sp:,.2f}")
    print(f"Gold:           ${current_gold:,.2f}")
    print(f"S&P/Gold Ratio: {ratio:.2f}")
    print(f"Current Regime: **{regime}**")
    
    print("\n" + "="*30)
    print(" RECOMMENDED ALLOCATION ")
    print("="*30)
    print(f" S&P 500:  {alloc[0]}%")
    print(f" Gold:     {alloc[1]}%")
    print(f" Cash:     {alloc[2]}%")
    print("="*30)

if __name__ == "__main__":
    analysis = run_analysis()
    send_telegram_message(analysis)
    print(analysis)