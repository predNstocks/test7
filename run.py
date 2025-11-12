import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime
import json 

# --- Configuration ---
# Easily modify this list to analyze different stock tickers.
TICKERS = ["SPY", "GLD", "QQQ", "GOOGL", "NVDA", "AAPL", "MSFT", "AMZN", "META", "AMD", "NFLX", "MU", "INTC"]

class Globals:
    pe_ratio_sp500 = 25.0
    n_ratio_sp500 = 1.0

from typing import Dict, Tuple

def long_term_investment_score(ticker_symbol: str, base_daily: float = 10.0) -> Dict:
    t = yf.Ticker(ticker_symbol)
    info = t.info
    try:
        current_price = info.get('currentPrice') or info['regularMarketPrice']

        # === SAFE ALL-TIME HIGH (handles splits & old tickers) ===
        hist = t.history(period="5y", auto_adjust=True)
        if len(hist) > 500:
            ath = hist['High'].max()
            ath_date = hist['High'].idxmax().strftime("%Y-%m-%d")
        else:
            ath = info.get('fiftyTwoWeekHigh', current_price) * 1.3
            ath_date = "recent"
        drawdown_pct = (ath - current_price) / ath * 100

        # === Key Metrics ===
        forward_pe = info.get('forwardPE')
        trailing_pe = info.get('trailingPE')
        peg = info.get('pegRatio')
        if peg is None or peg <= 0:
            growth = (info.get('earningsGrowth') or 0.1) * 100
            pe_use = forward_pe or trailing_pe or 20
            peg = pe_use / growth if growth > 0 else 99

        roic_proxy = info.get('returnOnEquity') or info.get('returnOnAssets')
        fcf = info.get('freeCashflow')
        market_cap = info.get('marketCap')
        fcf_yield = fcf / market_cap * 100 if fcf and market_cap and fcf > 0 else None

        debt_to_equity = info.get('debtToEquity')
        revenue_growth = info.get('revenueGrowth', 0) * 100
        earnings_growth = info.get('earningsGrowth', 0) * 100
        forward_trailing_ratio = forward_pe / trailing_pe if forward_pe and trailing_pe else 1.0

        # === SCORE CALCULATION (same as v2 aggressive drawdown) ===
        score = 0.0

        # ROIC/ROE
        #if roic_proxy > 0.25:           score += 25
        #elif roic_proxy > 0.18:         score += 20
        #elif roic_proxy > 0.10:         score += 10
        score += 100 * roic_proxy

        # FCF Yield
        #if fcf_yield and fcf_yield > 6: score += 20
        #elif fcf_yield and fcf_yield > 4: score += 12
        #elif fcf_yield and fcf_yield > 2: score += 6
        score += 3 * fcf_yield

        # PEG
        if peg < 1.0:                   score += 15
        elif peg < 1.3:                 score += 10
        elif peg < 1.8:                 score += 5
        elif peg > 3.0:                 score -= 8

        # Balance Sheet
        if debt_to_equity and debt_to_equity < 50:  score += 10
        elif debt_to_equity and debt_to_equity < 100: score += 5

        # Growth Outlook
        if forward_trailing_ratio < 0.9 and earnings_growth > 15: score += 10
        elif forward_trailing_ratio < 1.0 and earnings_growth > 8: score += 6

        # DRAWDOWN FROM ATH (AGGRESSIVE REWARD)
        if drawdown_pct > 70:           score += 40
        elif drawdown_pct > 60:         score += 35
        elif drawdown_pct > 50:         score += 30
        elif drawdown_pct > 40:         score += 22
        elif drawdown_pct > 30:         score += 15
        elif drawdown_pct > 20:         score += 8
        elif drawdown_pct > 10:         score += 3

        score = max(0, min(100, score))
        multiplier = score / 100 * 6.0
        daily_amount = base_daily * multiplier
        daily_amount = min(daily_amount, 100.0)
        if score < 40: daily_amount = 0.0

        # === COLOR-CODED METRICS WITH EMOJIS ===
        def colorize(value, thresholds_good, thresholds_ok=None, higher_better=True):
            if value is None:
                return "Gray: N/A"
            if higher_better:
                if thresholds_ok and value >= thresholds_ok[1]: return "🟢"
                elif thresholds_ok and value >= thresholds_ok[0]: return "🟡"
                elif value >= thresholds_good: return "🟢"
                elif value >= thresholds_good * 0.7: return "🟡"
                elif value >= thresholds_good * 0.4: return "🟠"
                else: return "⛔"
            else:  # lower better
                if value <= thresholds_good: return "🟢"
                elif value <= thresholds_good * 1.5: return "🟡"
                elif value <= thresholds_good * 3: return "🟠"
                else: return "⛔"

        metrics = {
            f"Score ({score}/100)": f"{'🟢' if score >= 85 else '🟡' if score >= 65 else '🟠' if score >= 45 else '⛔'}",
            f"Daily Invest": f"${daily_amount:.0f}",
            "": "",
            "Drawdown from ATH": f"{drawdown_pct:.1f}% → {colorize(drawdown_pct, 50, (30, 70))}",
            "PEG Ratio": f"{peg:.2f} → {colorize(peg, 1.0, (1.0, 1.3), higher_better=False)}",
            "FCF Yield": f"{fcf_yield:.2f}% → {colorize(fcf_yield, 5, (3, 7))}" if fcf_yield else "Gray: N/A",
            "ROE/ROIC": f"{roic_proxy:.1%} → {colorize(roic_proxy, 0.18, (0.12, 0.25))}",
            "Debt/Equity": f"{debt_to_equity:.0f}% → {colorize(debt_to_equity or 0, 50, (0, 100), higher_better=False)}" if debt_to_equity else "Yellow: OK",
            "Earnings Growth": f"{earnings_growth:+.1f}% → {colorize(earnings_growth, 15, (8, 25))}",
            "Forward/Trailing PE": f"{forward_trailing_ratio:.2f} → {colorize(forward_trailing_ratio, 0.9, (0.9, 1.0), higher_better=False)}",
        }

        return json.dumps({
            "ticker": ticker_symbol.upper(),
            "price": f"${current_price:.2f}",
            "ath_date": ath_date,
            "score": round(score, 1),
            "daily_amount_usd": round(daily_amount, 2),
            "metrics_visual": metrics
        }, indent=True, ensure_ascii=False)

    except Exception as e:
        return f"error: {str(e)}"

def calculate_n_score(ticker_symbol):
    """
    Fetches 5 years of historical data for a given stock ticker,
    calculates the 'n' value, computes the final score, and returns a formatted string.
    
    The formula is: 100 * n^2 / PE
    where n = (5-year high) / (current price)
    
    Returns a formatted string with the results or None on failure.
    """
    try:
        # 1. Get the ticker object
        ticker = yf.Ticker(ticker_symbol)
        
        # 2. Fetch historical data for the last 5 years
        hist_data = ticker.history(period="5y")
        
        if hist_data.empty:
            return f"--- {ticker_symbol}: Could not retrieve historical data. ---\n"

        # 3. Get current price and 5-year high
        current_price = hist_data['Close'].iloc[-1]
        five_year_high = hist_data['High'].max()
        
        if current_price <= 0:
            return f"--- {ticker_symbol}: Invalid current price ({current_price}). ---\n"

        # 4. Calculate 'n'
        n = five_year_high / current_price
        
        # 5. Get the current P/E ratio
        info = ticker.info

        #if ticker_symbol == "NVDA": breakpoint()
        pe_ratio = info.get('trailingPE', 20)
        peg_ratio = info.get('trailingPegRatio', 1.0)
        if peg_ratio is None:
            peg_ratio = 1.0
        if pe_ratio is None or pe_ratio <= 0:
            return f"--- {ticker_symbol}: P/E ratio not available or invalid. ---\n"
        
        if ticker_symbol == "SPY":
            Globals.pe_ratio_sp500 = pe_ratio
            Globals.n_ratio_sp500 = n
        
        
        pe_ratio_sp500 = Globals.pe_ratio_sp500
        n_ratio_sp500 = Globals.n_ratio_sp500
        
        pe_ratio_fw = info.get('forwardPE', pe_ratio)

        if ticker_symbol == "GLD":
            # 6. Calculate the final score            
            final_score = pe_ratio_sp500 * (n ** 3)
        else:
            final_score = ((1.5 / (peg_ratio + 0.5))** 2) * 250.0 / pe_ratio_sp500 * (pe_ratio ** 3) / (pe_ratio_fw ** 3)  * (n ** 2) * n_ratio_sp500


        # 7. Format the results into a string
        result_string = (
            f"----------- {ticker_symbol} ----------\n"
            f"  Current Price:     ${current_price:,.2f}\n"
            f"  5-Year High:       ${five_year_high:,.2f}\n"
            f"  Bakward P/E Ratio: {pe_ratio:.2f}\n"
            f"  Forward P/E Ratio: {pe_ratio_fw:.2f}\n"
            f"  PE/Growth Ratio:   {peg_ratio:.2f}\n"        
            f"  'n' value:         {n:.4f}\n"        
            f"  Final Score:       {final_score:.2f}\n"
        )
        return result_string

    except Exception as e:
        return f"--- An error occurred while processing {ticker_symbol}: {e} ---\n"

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


def main():
    """
    Main function to iterate through tickers, collect results, and send them.
    """
    print("Starting stock analysis...")
    
    results = [f"*Daily Stock Analysis - {datetime.now().strftime('%Y-%m-%d')}*"]
    
    for ticker in TICKERS:
        print(f"Processing: {ticker}...")
        result = calculate_n_score(ticker)
        result += long_term_investment_score(ticker)
        if result:
            results.append(result)
        
    print("Analysis complete. Sending to Telegram...")
    
    # Join all individual results into one message
    final_message = "\n".join(results)
    send_telegram_message(final_message)

if __name__ == "__main__":
    main()

