import yfinance as yf
import pandas as pd
import requests
import os
from datetime import datetime

# --- Configuration ---
# Easily modify this list to analyze different stock tickers.
TICKERS = ["SPY", "GLD", "QQQ", "GOOGL", "NVDA", "AAPL", "MSFT", "AMZN", "META", "AMD", "NFLX", "MU"]

class Globals:
    pe_ratio_sp500 = 25.0
    n_ratio_sp500 = 1.0

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
        pe_ratio = info.get('trailingPE', 20)
        
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
            final_score = 250.0 / pe_ratio_sp500 * (pe_ratio_fw ** 2) / (pe_ratio ** 2)  * (n ** 2) * n_ratio_sp500


        # 7. Format the results into a string
        result_string = (
            f"----------- {ticker_symbol} ----------\n"
            f"  Current Price:     ${current_price:,.2f}\n"
            f"  5-Year High:       ${five_year_high:,.2f}\n"
            f"  Bakward P/E Ratio: {pe_ratio:.2f}\n"
            f"  Forward P/E Ratio: {pe_ratio_fw:.2f}\n"
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
        if result:
            results.append(result)
        
    print("Analysis complete. Sending to Telegram...")
    
    # Join all individual results into one message
    final_message = "\n".join(results)
    send_telegram_message(final_message)

if __name__ == "__main__":
    main()

