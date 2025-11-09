import yfinance as yf
import numpy as np
from datetime import datetime
from typing import Dict
import json 

# === EMOJI LEGEND ===
# Green   = Excellent (ideal)
# Yellow  = OK / acceptable
# Orange  = Warning (risky)
# Red     = Forbidden / dangerous

def long_term_investment_score_v3(ticker_symbol: str, base_daily: float = 10.0) -> Dict:
    t = yf.Ticker(ticker_symbol)
    info = t.info
    try:
        current_price = info.get('currentPrice') or info['regularMarketPrice']

        # === SAFE ALL-TIME HIGH (handles splits & old tickers) ===
        hist = t.history(period="10y", auto_adjust=True)
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
        if roic_proxy > 0.25:           score += 25
        elif roic_proxy > 0.18:         score += 20
        elif roic_proxy > 0.10:         score += 10

        # FCF Yield
        if fcf_yield and fcf_yield > 6: score += 20
        elif fcf_yield and fcf_yield > 4: score += 12
        elif fcf_yield and fcf_yield > 2: score += 6

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

# === RUN ON YOUR WATCHLIST ===
tickers = ["META", "NVDA", "GOOGL", "MU", "AMD", "INTC", "AAPL", "TSLA", "MSFT", "SOFI"]

print(f"{'TICKER':<6} {'SCORE':<6} {'DAILY $':<8} VISUAL DASHBOARD\n" + "-"*80)
for sym in tickers:
    result = long_term_investment_score_v3(sym, base_daily=10)

    print(result)