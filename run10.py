import argparse
import io
import os
import warnings
import logging
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

FRED_SERIES = {
    "GDP": "GDP",
    "T10Y2Y": "T10Y2Y",
    "DGS10": "DGS10",
    "DGS2": "DGS2",
    "T10YIE": "T10YIE",  # 10-Year Breakeven Inflation Rate
}

MARKET_SYMBOLS = ["^GSPC", "GLD", "BTC-USD", "ETH-USD", "VOO", "TLT", "IEF", "BND", "BIL"]

class MacroPortfolioAllocator:
    def __init__(self, climate_risk_premium: float = 0.3, offline: bool = False, fast: bool = False):
        self.climate_risk_premium = climate_risk_premium
        self.offline = offline
        self.fast = fast
        self.progress_log = []
        # Clean global constants
        global FRED_SERIES, MARKET_SYMBOLS
        FRED_SERIES = self._clean_keys(FRED_SERIES)
        MARKET_SYMBOLS = self._clean_keys(MARKET_SYMBOLS)

    def _clean_keys(self, d: dict) -> dict:
        """Recursively strip trailing spaces from dict keys and string values."""
        if isinstance(d, dict):
            return {k.strip(): (self._clean_keys(v) if isinstance(v, (dict, list)) else v.strip() if isinstance(v, str) else v) 
                    for k, v in d.items()}
        elif isinstance(d, list):
            return [self._clean_keys(item) if isinstance(item, (dict, list)) else item.strip() if isinstance(item, str) else item 
                    for item in d]
        return d

    def _normalize_fred_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure FRED DataFrame always has a 'value' column."""
        if df is None or df.empty:
            return df
        if "value" in df.columns:
            return df
        # Rename first column to "value" (handles FRED's series-ID headers)
        return df.rename(columns={df.columns[0]: "value"})

    def log_progress(self, message: str) -> None:
        msg = f"[PROGRESS] {message}"
        logging.info(msg)
        self.progress_log.append(msg)

    def _cache_dir(self, subdir: str) -> str:
        path = os.path.join(os.getcwd(), subdir)
        os.makedirs(path, exist_ok=True)
        return path

    def _load_fred_cache(self, sid: str) -> Optional[pd.DataFrame]:
        path = os.path.join(self._cache_dir("fred_cache"), f"{sid}.csv")
        if os.path.exists(path):
            return pd.read_csv(path, index_col=0, parse_dates=True)
        return None

    def _load_local_override(self, sid: str) -> Optional[pd.DataFrame]:
        for candidate in [f"{sid}.csv", f"{sid.upper()}.csv"]:
            path = os.path.join(os.getcwd(), candidate)
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, index_col=0, parse_dates=True)
                    if df.shape[1] == 1:
                        df.columns = ["value"]
                    self.log_progress(f"Loaded local override: {candidate}")
                    return df
                except Exception as e:
                    logging.warning("Failed local override %s: %s", candidate, e)
        return None

    def _fetch_url(self, url: str, timeout: int = 10) -> Optional[pd.DataFrame]:
        headers = {"User-Agent": "Mozilla/5.0 (MacroAllocator/2.0)"}
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        try:
            r = session.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
            if df.shape[1] >= 1:
                df = df.iloc[:, [0]].rename(columns={df.columns[0]: "value"})
            return df
        except Exception as e:
            logging.warning("URL fetch failed %s: %s", url, e)
            return None

    def fetch_fred_data(self, series_ids: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        cache_dir = self._cache_dir("fred_cache")
        api_key = os.environ.get("FRED_API_KEY")
        data = {}
        missing = []

        for name, sid in series_ids.items():
            self.log_progress(f"Fetching FRED: {sid} ({name})")
            timeout = 5 if self.fast else 15

            # Offline fallback chain
            if self.offline:
                cached = self._load_fred_cache(sid)
                if cached is not None:
                    data[name] = cached
                    continue
                local = self._load_local_override(sid)
                if local is not None:
                    data[name] = local
                    continue
                missing.append(sid)
                continue

            # Primary: FRED Web CSV
            graph_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            df = self._fetch_url(graph_url, timeout)
            if df is not None:
                data[name] = df
                df.to_csv(os.path.join(cache_dir, f"{sid}.csv"))
                continue

            # Secondary: Official API
            if api_key:
                api_url = f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={api_key}&file_type=csv"
                df = self._fetch_url(api_url, timeout)
                if df is not None and "value" in df.columns:
                    data[name] = df[["value"]]
                    df[["value"]].to_csv(os.path.join(cache_dir, f"{sid}.csv"))
                    continue

            # Tertiary: Cache/Local
            cached = self._load_fred_cache(sid)
            if cached is not None:
                data[name] = cached
                continue
            local = self._load_local_override(sid)
            if local is not None:
                data[name] = local
                continue

            missing.append(sid)

        if missing:
            path = os.path.join(os.getcwd(), "missing_fred_series.txt")
            with open(path, "w") as f:
                for sid in missing:
                    f.write(f"{sid}\n")
            self.log_progress(f"Missing series saved to {path}")

        return data

    def fetch_market_data(self, symbols: List[str], period: str = "15y") -> pd.DataFrame:
        cache_path = os.path.join(self._cache_dir("market_cache"), "market_data.csv")
        if self.offline:
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                return df
            raise RuntimeError("Offline mode: market cache missing")

        fetch_period = "5y" if self.fast else period
        self.log_progress(f"Downloading {len(symbols)} symbols ({fetch_period})")
        try:
            raw = yf.download(symbols, period=fetch_period, progress=False, threads=False)
            if raw.empty:
                raise RuntimeError("Yahoo Finance returned empty data")
            close = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
            close = close.ffill().dropna()
            close.to_csv(cache_path)
            return close
        except Exception as e:
            logging.error("Market fetch failed: %s", e)
            if os.path.exists(cache_path):
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)
            raise

    def compute_metrics(self, prices: pd.DataFrame, fred_data: dict) -> dict:
        today = prices.index.max()

        def latest_price(sym: str) -> float:
            if sym in prices.columns:
                series = prices[sym].ffill()
                return float(series.loc[today]) if today in series.index else float(series.iloc[-1])
            raise RuntimeError(f"Market data missing symbol {sym}")

        spx = latest_price("^GSPC")
        gold = latest_price("GLD")
        sp_gold_ratio = spx / gold if gold != 0 else float("nan")

        # === ROBUST GDP/Z-SCORE CALCULATION ===
        gdp_df = fred_data.get("GDP")
        breakpoint()
        if gdp_df is None or gdp_df.empty:
            self.log_progress("WARNING: GDP data missing; z_score set to NaN")
            z_score = float("nan")
        else:
            # Normalize column to 'value'
            if "value" not in gdp_df.columns:
                numeric_cols = gdp_df.select_dtypes(include=[np.number]).columns.tolist()
                col_to_use = numeric_cols[0] if numeric_cols else gdp_df.columns[0]
                gdp_df = gdp_df[[col_to_use]].rename(columns={col_to_use: "value"})
            
            # Resample GDP to MONTHLY (not daily!) to match market data frequency
            gdp_monthly = gdp_df["value"].resample("ME").last().ffill()
            
            # Use VOO or ^GSPC as market cap proxy
            market_proxy = prices["VOO"] if "VOO" in prices.columns else prices["^GSPC"]
            market_monthly = market_proxy.resample("ME").last().ffill()
            
            # Align indices and compute Buffett ratio
            common_idx = market_monthly.index.intersection(gdp_monthly.index)
            if len(common_idx) < 60:  # Need at least 5 years of monthly data
                self.log_progress(f"WARNING: Only {len(common_idx)} aligned GDP/market points; z_score unreliable")
                z_score = float("nan")
            else:
                buffett = market_monthly.loc[common_idx] / gdp_monthly.loc[common_idx]
                
                # Rolling stats with adaptive window
                window = min(120, len(buffett) // 2)  # Use up to 10 years, or half available data
                roll_mean = buffett.rolling(window=window, min_periods=window//2).mean()
                roll_std = buffett.rolling(window=window, min_periods=window//2).std()
                
                if roll_std.iloc[-1] > 1e-8 and not pd.isna(roll_mean.iloc[-1]):
                    z_score = float((buffett.iloc[-1] - roll_mean.iloc[-1]) / roll_std.iloc[-1])
                else:
                    self.log_progress("WARNING: Buffett rolling std too low; z_score set to 0.0")
                    z_score = 0.0

        # === YIELD & MACRO METRICS ===
        def fred_last(key: str) -> float:
            df = fred_data.get(key)
            if df is None or df.empty:
                self.log_progress(f"WARNING: FRED series {key} missing")
                return float("nan")
            if "value" in df.columns:
                return float(df["value"].iloc[-1])
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            return float(df[numeric_cols[0]].iloc[-1]) if len(numeric_cols) > 0 else float("nan")

        t10y2y = fred_last("T10Y2Y")
        dgs10 = fred_last("DGS10")
        dgs2 = fred_last("DGS2")
        
        # Use market-implied inflation if available, else fallback
        t10yie = fred_last("T10YIE") if "T10YIE" in fred_data else 2.3
        real_yield = dgs10 - t10yie

        # Volatility proxy
        sp_ret = prices["^GSPC"].pct_change().dropna()
        vix_proxy = float(sp_ret.rolling(window=30).std().iloc[-1] * np.sqrt(252) * 100) if len(sp_ret) >= 30 else 16.0

        # Trend ratios (12-month lookback on monthly data)
        def trend_12m(sym: str) -> float:
            if sym not in prices.columns:
                return 1.0
            monthly = prices[sym].resample("ME").last().ffill()
            if len(monthly) < 12:
                return 1.0
            ma12 = monthly.rolling(window=12, min_periods=6).mean()
            return float(monthly.iloc[-1] / ma12.iloc[-1]) if ma12.iloc[-1] != 0 else 1.0

        # ERP proxy: 10Y yield - long-term earnings yield (~3.5%)
        earnings_yield = 0.035
        erp = dgs10 - earnings_yield

        return {
            "as_of": today,
            "z_score": z_score,
            "sp_gold_ratio": sp_gold_ratio,
            "yield_curve": t10y2y,
            "dgs10": dgs10,
            "real_yield": real_yield,
            "vix": vix_proxy,
            "trend_equity": trend_12m("^GSPC"),
            "trend_gold": trend_12m("GLD"),
            "erp": erp,
        }
        
    def _regime_scores(self, m: Dict[str, float]) -> Dict[str, float]:
        """Map macro indicators to Growth/Inflation/Risk scores (0-1)"""
        growth = np.clip((m["yield_curve"] + 50) / 150, 0, 1)  # Steepening = growth
        growth += np.clip((2.0 - m["z_score"]) / 3.0, 0, 1) * 0.3  # Undervalued = higher growth potential
        growth = np.clip(growth, 0, 1)

        inflation = np.clip((m["real_yield"] + 1.5) / 3.0, 0, 1)  # Low real yield = high inflation pressure
        inflation = np.clip(inflation, 0, 1)

        risk = np.clip((m["vix"] - 12) / 24, 0, 1)
        risk += np.clip((-m["yield_curve"]) / 50, 0, 0.5) * 0.4
        risk = np.clip(risk, 0, 1)

        return {"growth": growth, "inflation": inflation, "risk": risk}

    def allocate_portfolio(self, metrics: Dict[str, float]) -> Dict[str, float]:
        scores = self._regime_scores(metrics)
        g, i, r = scores["growth"], scores["inflation"], scores["risk"]

        # Strategic base weights
        w_eq = 50 + 20 * g - 15 * r - 10 * i
        w_bnd = 20 + 25 * (1 - g) + 15 * (1 - r) - 10 * i
        w_gld = 10 + 20 * i + 15 * r
        w_cash = 5 + 10 * r + 10 * i
        w_crp = 5 + 10 * g - 15 * r - 10 * i

        # Smooth regime transitions using sigmoid-like scaling
        def smooth(x, center=0.5, width=0.2):
            return 1 / (1 + np.exp(-(x - center) / width))

        # Apply economic dynamics:
        # 1. Bond yield attractiveness vs stocks (Fed Model logic)
        if metrics["dgs10"] > metrics["erp"] + 1.0:  # Bonds offer better risk-adjusted return
            w_bnd += 8
            w_eq -= 5
        # 2. Negative real yields boost gold/crypto
        if metrics["real_yield"] < 0:
            w_gld += 5
            w_crp += 2
        # 3. High inflation hurts long bonds
        if metrics["real_yield"] < -1.0:
            w_bnd -= 5

        # Constraints
        weights = {
            "equities": np.clip(w_eq, 10, 75),
            "bonds": np.clip(w_bnd, 5, 60),
            "gold": np.clip(w_gld, 5, 35),
            "cash": np.clip(w_cash, 5, 30),
            "crypto": np.clip(w_crp, 0, 15),
        }

        total = sum(weights.values())
        return {k: round(v / total * 100, 2) for k, v in weights.items()}

    def allocate_subcategories(self, weights: Dict[str, float], metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        sub = {}
        t10 = metrics["dgs10"]
        curve = metrics["yield_curve"]

        # Equities: quality vs growth vs broad
        if weights["equities"] > 55 and metrics["trend_equity"] > 1.0:
            sub["equities"] = {"VOO": 0.40, "MSFT": 0.20, "GOOGL": 0.15, "QQQ": 0.15, "SPY": 0.10}
        else:
            sub["equities"] = {"VOO": 0.55, "MSFT": 0.15, "GOOGL": 0.12, "QQQ": 0.10, "SPY": 0.08}

        # Bonds: duration positioning
        if curve > 0 and t10 < 4.0:
            sub["bonds"] = {"TLT": 0.45, "IEF": 0.30, "BND": 0.25}
        elif curve < -15 or t10 > 5.0:
            sub["bonds"] = {"BIL": 0.50, "SHY": 0.30, "BND": 0.20}
        else:
            sub["bonds"] = {"IEF": 0.40, "BND": 0.35, "TLT": 0.15, "TIP": 0.10}

        # Gold: physical vs ETF
        sub["gold"] = {"GLD": 0.60, "Physical Gold": 0.40} if metrics["real_yield"] < 0 else {"GLD": 0.80, "Physical Gold": 0.20}

        # Crypto: risk-on allocation
        sub["crypto"] = {"BTC": 0.75, "ETH": 0.25} if metrics["vix"] < 25 else {}

        # Cash: ultra-short treasuries
        sub["cash"] = {"BIL": 1.0}
        return sub

    def find_similar_periods(self, prices: pd.DataFrame, fred_data: Dict[str, pd.DataFrame], current: Dict[str, float], top_n: int = 3) -> List[Dict]:
        monthly = prices.resample("ME").last().ffill().dropna()
        breakpoint()
        gdp_m = fred_data["GDP"].resample("ME").ffill()
        common = monthly.index.intersection(gdp_m.index)
        if len(common) < 120:
            return []

        buffett = monthly["^GSPC"].loc[common] / gdp_m.loc[common]
        z_hist = (buffett - buffett.rolling(120).mean()) / buffett.rolling(120).std()
        curve_hist = fred_data["T10Y2Y"].reindex(common).ffill()
        t10_hist = fred_data["DGS10"].reindex(common).ffill()
        vix_hist = monthly["^GSPC"].pct_change().rolling(30).std().dropna() * np.sqrt(12) * 100
        vix_hist = vix_hist.reindex(common).ffill()

        hist = pd.DataFrame({
            "z_score": z_hist,
            "yield_curve": curve_hist,
            "ten_year_yield": t10_hist,
            "vix": vix_hist,
            "sp_gold": monthly["^GSPC"].loc[common] / monthly["GLD"].loc[common]
        }).dropna()

        if hist.empty:
            return []

        features = ["z_score", "yield_curve", "ten_year_yield", "vix", "sp_gold"]
        hist_s = (hist[features] - hist[features].mean()) / hist[features].std()
        cur_vals = np.array([current[f] for f in features])
        cur_s = (cur_vals - hist[features].mean().values) / hist[features].std().values

        dists = np.linalg.norm(hist_s.values - cur_s, axis=1)
        top = hist.assign(distance=dists).nsmallest(top_n, "distance")

        return [
            {
                "date": idx.strftime("%Y-%m"),
                **{k: round(float(row[k]), 2) for k in features if k != "distance"}
            }
            for idx, row in top.iterrows()
        ]

    def build_message(self, metrics: Dict, weights: Dict, suballoc: Dict, similar: List[Dict]) -> str:
        def bar(v):
            return "█" * int(v/4) + "░" * (25 - int(v/4))

        lines = [
            f"📊 **Macro Portfolio Signal — {metrics['as_of'].strftime('%Y-%m-%d')}**",
            "",
            "**🌐 Market Regime**",
            f"• Buffett Z-Score: `{metrics['z_score']:.2f}`",
            f"• S&P/Gold Ratio: `{metrics['sp_gold_ratio']:.2f}`",
            f"• 10Y Yield: `{metrics['dgs10']:.2f}%` | Curve: `{metrics['yield_curve']:.1f}bps`",
            f"• Real Yield: `{metrics['real_yield']:.2f}%` | ERP Proxy: `{metrics['erp']:.2f}%`",
            f"• Volatility (VIX Proxy): `{metrics['vix']:.1f}` | Equity Trend: `{metrics['trend_equity']:.2f}`",
            "",
            "**🎯 Strategic Allocation**",
            f"▸ Equities: `{weights['equities']:.1f}%` {bar(weights['equities'])}",
            f"▸ Bonds: `{weights['bonds']:.1f}%` {bar(weights['bonds'])}",
            f"▸ Gold: `{weights['gold']:.1f}%` {bar(weights['gold'])}",
            f"▸ Crypto: `{weights['crypto']:.1f}%` {bar(weights['crypto'])}",
            f"▸ Cash: `{weights['cash']:.1f}%` {bar(weights['cash'])}",
            "",
            "**📦 Tactical Implementation**",
        ]
        for cat, plan in suballoc.items():
            if not plan:
                continue
            lines.append(f"• {cat.title()}:")
            for t, pct in plan.items():
                lines.append(f"    – `{t}`: `{pct*100:.0f}%`")

        lines += ["", "**🕰️ Historical Analogues**"]
        if similar:
            for s in similar:
                lines.append(f"• `{s['date']}` | Z:`{s['z_score']:.1f}` | Curve:`{s['yield_curve']:.0f}` | 10Y:`{s['ten_year_yield']:.1f}%` | VIX:`{s['vix']:.1f}`")
        else:
            lines.append("• No close analogue found.")

        lines += ["", "_Note: Allocation is regime-aware, yield-sensitive, and dynamically balances growth/inflation/risk exposures._"]
        return "\n".join(lines)

    def send_telegram(self, message: str):
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()

    def run_backtest(self, symbols=None, years=15, plot_path="run9_backtest_v2.png"):
        symbols = symbols or MARKET_SYMBOLS
        prices = self.fetch_market_data(symbols, period=f"{years+2}y")
        monthly = prices.resample("ME").last().ffill()
        fred = self.fetch_fred_data(FRED_SERIES)

        weights_hist, returns_hist, values = [], [], [1.0]
        monthly_ret = monthly.pct_change().dropna()
        tx_cost = 0.0015  # 0.15% per rebalance

        for i in range(1, len(monthly)-1):
            t = monthly.index[i]
            # Look-back data (avoid look-ahead bias)
            slice_p = prices.loc[:t].tail(2500)
            m = self.compute_metrics(slice_p, fred)
            w = self.allocate_portfolio(m)
            weights_hist.append(w)

            # Forward 1-month returns
            fwd = monthly_ret.loc[t]
            port_ret = sum(w[k]/100 * fwd.get(k, 0) for k in w) - tx_cost
            values.append(values[-1] * (1 + port_ret))
            returns_hist.append(port_ret)

        dates = monthly.index[1:len(values)+1]
        perf = pd.Series(values, index=dates)
        ret_series = pd.Series(returns_hist, index=dates[1:])

        # Performance Metrics
        cagr = (perf.iloc[-1] ** (12/len(perf)) - 1) * 100
        vol = ret_series.std() * np.sqrt(12) * 100
        sharpe = (ret_series.mean() * 12) / (ret_series.std() * np.sqrt(12))
        max_dd = (perf / perf.cummax() - 1).min() * 100

        self.log_progress(f"Backtest CAGR: {cagr:.2f}% | Vol: {vol:.2f}% | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.2f}%")

        # Plot
        fig, ax1 = plt.subplots(figsize=(14, 7))
        pd.DataFrame(weights_hist).plot.area(ax=ax1, cmap="Set2", alpha=0.85, stacked=True)
        ax1.set_ylabel("Allocation %"); ax1.set_ylim(0, 100); ax1.set_title("Regime-Aware Allocation Backtest")
        ax2 = ax1.twinx()
        ax2.plot(dates, perf.values, "k--", lw=2)
        ax2.set_ylabel("Portfolio Value"); ax2.legend(["Value"])
        plt.tight_layout(); fig.savefig(plot_path, dpi=150)

        return pd.DataFrame(weights_hist), perf

def main(send_telegram=False, backtest=False, fast=False, offline=False):
    alloc = MacroPortfolioAllocator(offline=offline, fast=fast)
    alloc.log_progress("Starting macro allocation pipeline...")

    fred = alloc.fetch_fred_data(FRED_SERIES)
    prices = alloc.fetch_market_data(MARKET_SYMBOLS)
    metrics = alloc.compute_metrics(prices, fred)
    weights = alloc.allocate_portfolio(metrics)
    sub = alloc.allocate_subcategories(weights, metrics)
    similar = alloc.find_similar_periods(prices, fred, metrics, top_n=3)
    breakpoint()
    msg = alloc.build_message(metrics, weights, sub, similar)
    print(msg)

    with open("run9_report_v2.md", "w", encoding="utf-8") as f:
        f.write(msg.replace("*", "").replace("`", ""))

    if backtest:
        alloc.run_backtest(years=15)

    if send_telegram:
        alloc.send_telegram(msg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Macro Portfolio Allocator v2")
    parser.add_argument("--send", action="store_true", help="Send to Telegram")
    parser.add_argument("--backtest", action="store_true", help="Run 15Y backtest")
    parser.add_argument("--fast", action="store_true", help="Quick run for testing")
    parser.add_argument("--offline", action="store_true", help="Use cached data only")
    args = parser.parse_args()
    main(send_telegram=args.send, backtest=args.backtest, fast=args.fast, offline=args.offline)