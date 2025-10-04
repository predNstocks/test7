import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# --- Configuration ---
# Easily modify this list to analyze different stock tickers.
TICKERS = ["SPY", "GOOGL", "AAPL", "MSFT", "AMZN"]

def calculate_n_score(ticker_symbol):
    """
    Fetches 5 years of historical data for a given stock ticker,
    calculates the 'n' value, and computes the final score based on the P/E ratio.
    
    The formula is: 100 * n^2 / PE
    where n = (5-year high) / (current price)
    """
    try:
        print(f"--- Processing: {ticker_symbol} ---")
        
        # 1. Get the ticker object
        ticker = yf.Ticker(ticker_symbol)
        
        # 2. Fetch historical data for the last 5 years
        # Using period="5y" is a reliable way to get 5 years of data from the current date.
        hist_data = ticker.history(period="5y")
        
        if hist_data.empty:
            print(f"Could not retrieve historical data for {ticker_symbol}. Skipping.")
            return

        # 3. Get current price and 5-year high
        # The most recent closing price is used as the current value.
        current_price = hist_data['Close'].iloc[-1]
        five_year_high = hist_data['High'].max()
        
        if current_price <= 0:
            print(f"Invalid current price ({current_price}) for {ticker_symbol}. Skipping.")
            return

        # 4. Calculate 'n'
        n = five_year_high / current_price
        
        # 5. Get the current P/E ratio
        # We fetch general info and look for 'trailingPE'.
        info = ticker.info
        pe_ratio = info.get('trailingPE')
        
        if pe_ratio is None or pe_ratio <= 0:
            print(f"P/E ratio not available or invalid for {ticker_symbol}. Skipping calculation.")
            return
            
        # 6. Calculate the final score
        final_score = 100 * (n ** 2) / pe_ratio
        
        # 7. Print the results
        print(f"  Current Price: {current_price:,.2f}")
        print(f"  5-Year High:   {five_year_high:,.2f}")
        print(f"  P/E Ratio:     {pe_ratio:.2f}")
        print(f"  'n' value (High/Current): {n:.4f}")
        print(f"  Final Score (100 * n^2 / PE): {final_score:.2f}\n")

    except Exception as e:
        print(f"An error occurred while processing {ticker_symbol}: {e}\n")

def main():
    """
    Main function to iterate through the list of tickers and run the calculation.
    """
    print("Starting stock analysis...")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for ticker in TICKERS:
        calculate_n_score(ticker)
        
    print("Analysis complete.")

if __name__ == "__main__":
    main()

