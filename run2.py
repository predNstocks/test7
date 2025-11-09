import yfinance as yf
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
        ath = max(t.history(period="max")['High'])
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
        return {"error": str(e), "score": 0, "daily_amount": 0.0}

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
    if drawdown_from_ath > 50:  score += 15
    elif drawdown_from_ath > 30: score += 8
    elif drawdown_from_ath > 15: score += 4

    score = max(0, min(100, score))  # clamp

    # --- Daily amount logic ---
    multiplier = score / 100 * 5.0  # max 5x base
    daily_amount = base_daily * multiplier

    # Cap at reasonable levels
    daily_amount = min(daily_amount, 50.0)
    if score < 30:
        daily_amount = 0.0  # avoid trash

    return {
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
    }


tickers = ["NVDA", "GOOGL", "META", "AMD", "MU", "INTC", "AAPL", "MSFT"]
for sym in tickers:
    result = long_term_investment_score(sym, base_daily=10.0)
    print(f"\n{sym}: Score {result['score']} → Invest ${result['daily_amount_usd']}/day")
    print(result['key_metrics'])