import pandas as pd
import requests
import io
import yfinance as yf
from datetime import datetime
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def get_fred_data(series_ids):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://fred.stlouisfed.org/',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    data_frames = {}
    
    for sid in series_ids:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        try:
            print(f"Attempting direct fetch for {sid}...")
            resp = requests.get(url, headers=headers, timeout=15)
            print(f"  Status: {resp.status_code}")
            
            if resp.status_code == 200:
                df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
                if sid in df.columns:
                    df = df[[sid]].rename(columns={sid: 'value'})
                data_frames[sid] = df
            else:
                print(f"  Direct fetch failed for {sid} ({resp.status_code})")
        except Exception as e:
            print(f"  Exception during direct fetch for {sid}: {e}")
        
        time.sleep(1.5)
    
    return data_frames


def fallback_fred_via_pandas_datareader(series_ids):
    try:
        import pandas_datareader.data as web
        print("Falling back to pandas_datareader for FRED...")
        start = '1940-01-01'
        end = datetime.now().strftime('%Y-%m-%d')
        df_multi = web.DataReader(series_ids, 'fred', start, end)
        data_frames = {}
        for sid in series_ids:
            if sid in df_multi.columns:
                data_frames[sid] = df_multi[[sid]].rename(columns={sid: 'value'})
        return data_frames
    except Exception as e:
        print(f"pandas_datareader fallback failed: {e}")
        return {}


def run_macro_analysis():
    current_date = datetime.now().strftime("%Y-%m-%d")

    fred_series = ["GDP", "T10Y2Y"]
    fred_data = get_fred_data(fred_series)
    
    missing = [s for s in fred_series if s not in fred_data]
    if missing:
        fallback_data = fallback_fred_via_pandas_datareader(missing)
        fred_data.update(fallback_data)
    
    if len(fred_data) < len(fred_series):
        return f"Error: Could not fetch all required FRED data ({', '.join(fred_series)})."

    # Fetch market proxy
    try:
        print("Fetching total US market proxy via yfinance...")
        tickers = ["^FTW5000", "^W5000"]
        hist = pd.DataFrame()
        for t in tickers:
            temp = yf.download(t, period="max", progress=False)
            if not temp.empty:
                hist = temp
                print(f"  Using {t} with {len(hist)} rows")
                break
        if hist.empty:
            return "Error: No data from yfinance for ^FTW5000 or ^W5000."
        
        mkt_cap = hist[['Close']].rename(columns={'Close': 'value'}).rename_axis('DATE')
        print(f"Market data range: {mkt_cap.index.min().date()} to {mkt_cap.index.max().date()}")
    except Exception as e:
        return f"Error fetching market data: {e}"

    try:
        sp_price = yf.Ticker("^GSPC").history(period="5d")['Close'].iloc[-1]
        gold_price = yf.Ticker("GC=F").history(period="5d")['Close'].iloc[-1]
    except Exception as e:
        return f"Error fetching Yahoo prices (S&P/Gold): {e}"

    # Metrics
    ratio = sp_price / gold_price

    # Align series properly
    gdp = fred_data['GDP'].resample('D').ffill().rename_axis('DATE')
    mkt_cap_daily = mkt_cap.resample('D').ffill().rename_axis('DATE')

    # Create common daily index from min to max of both
    start_date = max(gdp.index.min(), mkt_cap_daily.index.min())
    end_date = min(gdp.index.max(), mkt_cap_daily.index.max())
    common_dates = pd.date_range(start_date, end_date, freq='D')

    gdp_aligned = gdp.reindex(common_dates).ffill().bfill()
    mkt_aligned = mkt_cap_daily.reindex(common_dates).ffill().bfill()
    # print(mkt_aligned['value'] )
    # print(gdp_aligned['value'] )
    buffett_series = (mkt_aligned['value'] / gdp_aligned['value'])

    print(f"Buffett series length after alignment: {len(buffett_series)}")
    if len(buffett_series) > 0:
        print(f"Buffett range: {buffett_series.index.min().date()} to {buffett_series.index.max().date()}")

    min_overlap = 252 * 3  # lowered to ~3 years
    if len(buffett_series) < min_overlap:
        return f"Error: Still insufficient overlapping data ({len(buffett_series)} days). Market data likely starts too late relative to GDP."

    # Z-Score with safer window
    current_val = buffett_series.iloc[-1]
    rolling_mean = buffett_series.rolling(window=2520, min_periods=500).mean().iloc[-1]
    rolling_std = buffett_series.rolling(window=2520, min_periods=500).std().iloc[-1]
    
    if pd.isna(rolling_std) or rolling_std == 0:
        z_score = float('nan')
    else:
        z_score = (current_val - rolling_mean) / rolling_std

    yield_spread = fred_data['T10Y2Y']['value'].iloc[-1]

    # Decision engine
    if pd.isna(z_score):
        regime = "INSUFFICIENT HISTORY FOR Z-SCORE"
        alloc = [33, 33, 34]
    elif z_score > 2.0:
        regime = "EXCESSIVE BUBBLE"
        alloc = [20, 30, 50]
    elif yield_spread < 0:
        regime = "RECESSION WARNING"
        alloc = [10, 50, 40]
    elif ratio < 1.0:
        regime = "VALUE RECOVERY"
        alloc = [70, 20, 10]
    else:
        regime = "GROWTH / MOMENTUM"
        alloc = [60, 20, 20]

    # Report
    z_str = f"{z_score:.2f}" if not pd.isna(z_score) else "N/A (limited history)"
    report = (
        f"\n{'='*50}\n"
        f"       GLOBAL MACRO DASHBOARD  ({current_date})\n"
        f"{'='*50}\n"
        f" [VALUATION] Buffett Z-Score:  {z_str} (Target: < 1.0)\n"
        f" [LIQUIDITY] 10Y-2Y Spread:    {yield_spread:.2f}% (Warning: < 0.0)\n"
        f" [CURRENCY]  S&P/Gold Ratio:   {ratio:.2f} (Neutral: ~1.5)\n"
        f"--------------------------------------------------\n"
        f" MARKET REGIME: **{regime}**\n"
        f" RECOMMENDED ALLOCATION:\n"
        f"   > Equities (S&P 500 proxy): {alloc[0]}%\n"
        f"   > GOLD          (Defense):  {alloc[1]}%\n"
        f"   > CASH         (Liquidity): {alloc[2]}%\n"
        f"{'='*50}\n"
        f"Note: Using FT Wilshire 5000 (^FTW5000) as total market proxy. Overlap limited to recent years."
    )
    
    return report


if __name__ == "__main__":
    print(run_macro_analysis())