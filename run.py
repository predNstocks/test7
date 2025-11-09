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

import numpy as np
from typing import Dict, Tuple

def long_term_investment_score(ticker_symbol: str, base_daily: float = 10.0) -> Dict:
    """
    Returns a score (0-100) and recommended daily DCA amount for long-term holding.
    Higher score = stronger long-term buy = higher daily investment.
    """
    t = yf.Ticker(ticker_symbol)
    info = t.info

    # --- Fetch key data ---
    try:
        current_price = info['currentPrice'] or info['regularMarketPrice']
        ath = max(t.history(period="5y")['High'])
        drawdown_from_ath = (ath - current_price) / ath * 100

        forward_pe = info.get('forwardPE')
        trailing_pe = info.get('trailingPE')
        peg = info.get('pegRatio')
        if peg is None or peg <= 0:
            # Manual PEG fallback
            growth = info.get('earningsGrowth', 0.1) * 100
            pe_for_peg = forward_pe or trailing_pe
            peg = pe_for_peg / growth if growth > 0 and pe_for_peg else 99

        roic = info.get('returnOnEquity')  # proxy if ROIC missing
        fcf_yield = info.get('freeCashflow') / info.get('marketCap') * 100 if info.get('freeCashflow') and info.get('marketCap') else None
        debt_to_ebitda = info.get('debtToEquity') / 100  # rough proxy
        revenue_growth = info.get('revenueGrowth', 0) * 100
        eps_growth = info.get('earningsGrowth', 0) * 100

        forward_trailing_ratio = forward_pe / trailing_pe if forward_pe and trailing_pe else 1.0

    except Exception as e:
        return str(e) #{"error": str(e), "score": 0, "daily_amount": 0.0}

    # --- Scoring system (0-100) ---
    score = 0.0

    # 1. Moat proxy (ROIC/ROE) - 25 pts
    if roic > 0.20:   score += 25
    elif roic > 0.15: score += 20
    elif roic > 0.10: score += 10

    # 2. FCF Yield - 20 pts
    if fcf_yield and fcf_yield > 6:   score += 20
    elif fcf_yield and fcf_yield > 4: score += 12
    elif fcf_yield and fcf_yield > 2: score += 6

    # 3. PEG Ratio - 20 pts
    if peg < 1.0:     score += 20
    elif peg < 1.3:   score += 15
    elif peg < 1.8:   score += 8
    elif peg > 3.0:   score -= 10  # penalty

    # 4. Balance sheet - 10 pts
    if debt_to_ebitda and debt_to_ebitda < 1.0: score += 10
    elif debt_to_ebitda and debt_to_ebitda < 2.0: score += 5

    # 5. Growth outlook (forward vs trailing + revenue) - 10 pts
    if forward_trailing_ratio < 0.9 and revenue_growth > 15: score += 10
    elif forward_trailing_ratio < 1.0 and revenue_growth > 10: score += 6

    # 6. Drawdown from ATH - bonus up to 15 pts (opportunity)
    #if drawdown_from_ath > 50:  score += 15
    #elif drawdown_from_ath > 30: score += 8
    #elif drawdown_from_ath > 15: score += 4
    score += drawdown_from_ath / 2.0

    score = max(0, min(100, score))  # clamp

    # --- Daily amount logic ---
    multiplier = score / 100 * 5.0  # max 5x base
    daily_amount = base_daily * multiplier

    # Cap at reasonable levels
    daily_amount = min(daily_amount, 50.0)
    if score < 30:
        daily_amount = 0.0  # avoid trash

    return json.dumps({
        "ticker": ticker_symbol,
        "score": round(score, 1),
        "interpretation":
            "90+ God-tier compounder | 70-89 Strong buy | 50-69 Decent | <50 Avoid/Speculative",
        "daily_amount_usd": round(daily_amount, 2),
        "key_metrics": {
            "PEG": round(peg, 2),
            "FCF_Yield_%": round(fcf_yield, 2) if fcf_yield else None,
            "ROE": round(roic, 3) if roic else None,
            "Drawdown_from_ATH_%": round(drawdown_from_ath, 1),
            "Forward/Trailing_PE_ratio": round(forward_trailing_ratio, 2),
            "Debt_to_Equity_proxy": round(debt_to_ebitda, 2) if debt_to_ebitda else None
        }
    }, indent=True)

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

