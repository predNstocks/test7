import argparse
import io
import os
import warnings
from datetime import datetime

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

FRED_SERIES = {
    "GDP": "GDP",
    "T10Y2Y": "T10Y2Y",
    "DGS10": "DGS10",
    "DGS2": "DGS2",
}

MARKET_SYMBOLS = ["^GSPC", "GLD", "BTC-USD", "ETH-USD", "MSFT", "GOOGL", "VOO", "TLT", "IEF", "BND"]


class MacroPortfolioAllocator:
    def __init__(self, climate_risk_premium: float = 0.3, offline: bool = False, fast: bool = False):
        self.climate_risk_premium = climate_risk_premium
        self.offline = offline
        self.fast = fast
        self.progress_log = []

    def log_progress(self, message: str):
        msg = f"[PROGRESS] {message}"
        print(msg)
        self.progress_log.append(msg)

    def _market_cache_path(self) -> str:
        cache_dir = os.path.join(os.getcwd(), "market_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, "market_data.csv")

    def _fred_cache_path(self, sid: str) -> str:
        cache_dir = os.path.join(os.getcwd(), "fred_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{sid}.csv")

    def _load_fred_cache(self, sid: str):
        path = self._fred_cache_path(sid)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            return df
        return None

    def _load_local_override(self, sid: str):
        # Look for a workspace CSV that the user may have provided (e.g. GDP.csv)
        candidates = [f"{sid}.csv", f"{sid.upper()}.csv", os.path.join(os.getcwd(), f"{sid}.csv")]
        for c in candidates:
            if os.path.exists(c):
                try:
                    df = pd.read_csv(c, index_col=0, parse_dates=True)
                    if df.shape[1] == 1:
                        df = df.rename(columns={df.columns[0]: "value"})
                    self.log_progress(f"Loaded local override file {c} for {sid}")
                    return df
                except Exception as e:
                    self.log_progress(f"Failed reading local override {c}: {e}")
        return None

    def _fred_api_csv_url(self, sid: str, api_key: str | None = None) -> str:
        # Official FRED API endpoint for CSV observations (requires API key)
        if api_key:
            return f"https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={api_key}&file_type=csv"
        # Fallback web-UI CSV download
        return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

    def fetch_fred_data(self, series_ids):
        headers = {"User-Agent": "Mozilla/5.0"}
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        api_key = os.environ.get("FRED_API_KEY")
        series_data = {}
        missing = []

        for name, sid in series_ids.items():
            graph_url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            api_url = self._fred_api_csv_url(sid, api_key)
            self.log_progress(f"Fetching FRED series {sid} ({name})")
            timeout_sec = 1 if self.fast else 30

            # Offline-only: prefer cached or local override
            if self.offline:
                self.log_progress(f"Offline mode: trying cache/local for {sid}")
                cached = self._load_fred_cache(sid)
                if cached is not None:
                    series_data[name] = cached
                    self.log_progress(f"Loaded cached FRED series {sid} in offline mode")
                    continue
                local = self._load_local_override(sid)
                if local is not None:
                    series_data[name] = local
                    continue
                self.log_progress(f"Offline and no data for {sid}")
                missing.append((sid, graph_url, api_url))
                continue

            # Try web-UI CSV first (graph_url)
            try:
                self.log_progress(f"Attempting graph CSV URL: {graph_url} (timeout={timeout_sec}s)")
                response = session.get(graph_url, headers=headers, timeout=timeout_sec)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text), index_col=0, parse_dates=True)
                # normalize column name
                if df.shape[1] >= 1:
                    df = df.iloc[:, [0]].rename(columns={df.columns[0]: "value"})
                series_data[name] = df
                df.to_csv(self._fred_cache_path(sid))
                self.log_progress(f"Fetched and cached FRED series {sid} via graph URL")
                continue
            except Exception as exc_graph:
                self.log_progress(f"Graph CSV fetch failed for {sid}: {exc_graph}")

            # Try official API if API key present
            if api_key:
                try:
                    self.log_progress(f"Attempting official FRED API URL: {api_url}")
                    response = session.get(api_url, headers=headers, timeout=timeout_sec)
                    response.raise_for_status()
                    df = pd.read_csv(io.StringIO(response.text), index_col=0, parse_dates=True)
                    if df.shape[1] >= 1:
                        # Official API returns columns including 'date' and 'value'
                        if "value" in df.columns:
                            df = df[["value"]].rename(columns={"value": "value"})
                        else:
                            df = df.iloc[:, [1]].rename(columns={df.columns[1]: "value"})
                    series_data[name] = df
                    df.to_csv(self._fred_cache_path(sid))
                    self.log_progress(f"Fetched and cached FRED series {sid} via official API")
                    continue
                except Exception as exc_api:
                    self.log_progress(f"Official API fetch failed for {sid}: {exc_api}")

            # Try cache then local workspace override
            cached = self._load_fred_cache(sid)
            if cached is not None:
                series_data[name] = cached
                self.log_progress(f"Using cached FRED series {sid} after network failures")
                continue

            local = self._load_local_override(sid)
            if local is not None:
                series_data[name] = local
                self.log_progress(f"Using local file override for {sid}")
                continue

            # If we reach here, we couldn't retrieve the series
            self.log_progress(f"Unable to retrieve FRED series {sid}; adding to missing list")
            missing.append((sid, graph_url, api_url))

        # Write missing links for manual download if any
        if missing:
            out_path = os.path.join(os.getcwd(), "missing_fred_links.md")
            with open(out_path, "w") as f:
                f.write("# Missing FRED Series Download Links\n\n")
                f.write("If a series failed to download automatically, use one of the links below to download the CSV manually.\n\n")
                for sid, graph_url, api_url in missing:
                    f.write(f"- **{sid}**: Graph CSV: {graph_url}\n")
                    if api_key:
                        f.write(f"  - Official API CSV (with your `FRED_API_KEY`): {api_url}\n")
                    else:
                        # show official API URL template
                        f.write(f"  - Official API CSV (requires API key): https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key=YOUR_KEY&file_type=csv\n")
                f.write("\n")
            self.log_progress(f"Wrote missing FRED download links to {out_path}")

        return series_data

    def fetch_market_data(self, symbols, period="30y"):
        cache_path = self._market_cache_path()
        if self.offline:
            self.log_progress("Offline mode: attempting to load market data from cache...")
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                self.log_progress("Loaded market data from cache.")
                return df
            raise RuntimeError("Offline mode and no market cache available")

        fetch_period = period
        if self.fast:
            fetch_period = "5y"
            self.log_progress(f"Fast mode: reducing market data period to {fetch_period} for speed")

        self.log_progress(f"Downloading market data for {len(symbols)} symbols (period={fetch_period})")
        try:
            raw = yf.download(symbols, period=fetch_period, progress=False, threads=False)
            if raw.empty:
                raise RuntimeError("Yahoo Finance returned no market data")

            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].copy()
            else:
                close = raw[["Close"]].copy()
                close.columns = symbols

            close = close.ffill().dropna()
            try:
                close.to_csv(cache_path)
                self.log_progress(f"Market data cached to {cache_path}")
            except Exception:
                self.log_progress("Warning: failed to write market cache")
            return close
        except Exception as exc:
            self.log_progress(f"Error downloading market data: {exc}")
            if os.path.exists(cache_path):
                self.log_progress("Falling back to cached market data")
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                return df
            raise

    def compute_metrics(self, prices: pd.DataFrame, fred_data: dict) -> dict:
        today = prices.index.max()

        def latest_price(sym: str) -> float:
            if sym in prices.columns:
                try:
                    return float(prices[sym].loc[today])
                except Exception:
                    return float(prices[sym].ffill().iloc[-1])
            raise RuntimeError(f"Market data missing symbol {sym}")

        spx = latest_price("^GSPC")
        gold = latest_price("GLD")
        sp_gold_ratio = spx / gold if gold != 0 else float("nan")

        # Normalize GDP frame to have a 'value' column
        gdp_df = fred_data.get("GDP")
        if gdp_df is None:
            raise RuntimeError("GDP series missing from FRED data")
        if isinstance(gdp_df, pd.Series):
            gdp_df = gdp_df.to_frame(name="value")
        elif "value" not in gdp_df.columns:
            numeric_cols = gdp_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                gdp_df = gdp_df[[numeric_cols[0]]].rename(columns={numeric_cols[0]: "value"})
            else:
                gdp_df = gdp_df.iloc[:, [0]].rename(columns={gdp_df.columns[0]: "value"})

        gdp_daily = gdp_df.resample("D").ffill()

        # Align market cap proxy and GDP
        common_idx = prices.index.intersection(gdp_daily.index)
        if len(common_idx) == 0:
            # fall back to reindexing GDP to prices
            gdp_for_prices = gdp_daily.reindex(prices.index, method="ffill")["value"]
            market_cap_proxy = prices["VOO"] if "VOO" in prices.columns else prices["^GSPC"]
            buffett_series = market_cap_proxy / gdp_for_prices
        else:
            market_cap_proxy = prices["VOO"].loc[common_idx] if "VOO" in prices.columns else prices["^GSPC"].loc[common_idx]
            buffett_series = market_cap_proxy / gdp_daily.loc[common_idx, "value"]

        roll = buffett_series.rolling(window=2520, min_periods=500)
        try:
            mean_last = roll.mean().iloc[-1]
            std_last = roll.std().iloc[-1]
            z_score = float((buffett_series.iloc[-1] - mean_last) / std_last) if std_last != 0 else 0.0
        except Exception:
            z_score = 0.0

        def fred_last_value(key: str) -> float:
            df = fred_data.get(key)
            if df is None:
                raise RuntimeError(f"FRED series {key} missing")
            if isinstance(df, pd.Series):
                return float(df.iloc[-1])
            if "value" in df.columns:
                return float(df["value"].iloc[-1])
            return float(df.iloc[-1, 0])

        t10y2y = fred_last_value("T10Y2Y")
        dgs10 = fred_last_value("DGS10")
        dgs2 = fred_last_value("DGS2")
        real_yield = dgs10 - self.climate_risk_premium
        yield_curve = t10y2y

        returns = prices["^GSPC"].pct_change().dropna()
        if "^VIX" in prices.columns:
            vix = float(prices["^VIX"].iloc[-1])
        else:
            window = min(30, len(returns))
            vix = float(returns.rolling(window=window).std().iloc[-1] * np.sqrt(252) * 100) if len(returns) > 0 else 0.0

        def trend(sym: str) -> float:
            if sym in prices.columns and len(prices[sym]) >= 200:
                return float(prices[sym].iloc[-1] / prices[sym].rolling(window=200, min_periods=1).mean().iloc[-1])
            if sym in prices.columns:
                return float(prices[sym].iloc[-1] / prices[sym].expanding().mean().iloc[-1])
            return 1.0

        trend_equity = trend("^GSPC")
        trend_gold = trend("GLD")
        trend_btc = trend("BTC-USD")

        metrics = {
            "as_of": today,
            "spx": float(spx),
            "gold": float(gold),
            "sp_gold_ratio": float(sp_gold_ratio),
            "z_score": float(z_score),
            "yield_curve": float(yield_curve),
            "ten_year_yield": float(dgs10),
            "two_year_yield": float(dgs2),
            "real_yield": float(real_yield),
            "vix": float(vix),
            "trend_equity": float(trend_equity),
            "trend_gold": float(trend_gold),
            "trend_btc": float(trend_btc),
        }
        return metrics
        returns = prices["^GSPC"].pct_change().dropna()
        vix = prices["^VIX"].iloc[-1] if "^VIX" in prices.columns else float(returns.rolling(window=30).std().iloc[-1] * np.sqrt(252) * 100)

        trend_equity = float(prices["^GSPC"].iloc[-1] / prices["^GSPC"].rolling(window=200, min_periods=200).mean().iloc[-1])
        trend_gold = float(prices["GLD"].iloc[-1] / prices["GLD"].rolling(window=200, min_periods=200).mean().iloc[-1])
        trend_btc = float(prices["BTC-USD"].iloc[-1] / prices["BTC-USD"].rolling(window=200, min_periods=200).mean().iloc[-1])

        metrics = {
            "as_of": today,
            "spx": float(spx),
            "gold": float(gold),
            "sp_gold_ratio": float(sp_gold_ratio),
            "z_score": float(z_score),
            "yield_curve": float(yield_curve),
            "ten_year_yield": float(dgs10),
            "two_year_yield": float(dgs2),
            "real_yield": float(real_yield),
            "vix": float(vix),
            "trend_equity": float(trend_equity),
            "trend_gold": float(trend_gold),
            "trend_btc": float(trend_btc),
        }
        return metrics

    def allocate_portfolio(self, metrics: dict) -> dict:
        z = metrics["z_score"]
        curve = metrics["yield_curve"]
        t10 = metrics["ten_year_yield"]
        spg = metrics["sp_gold_ratio"]
        vix = metrics["vix"]
        real_yield = metrics["real_yield"]

        valuation_risk = np.clip((z - 1.0) / 2.5, 0.0, 1.0)
        inverted_curve = max(0.0, -curve / 50.0)
        bond_yield_signal = np.clip((t10 - 2.5) / 4.5, -0.3, 0.5)
        gold_pref = np.clip((1.75 - spg) / 1.75, 0.0, 1.0)
        volatility_risk = np.clip((vix - 16.0) / 24.0, 0.0, 1.0)
        trend_risk = np.clip((metrics["trend_equity"] - 1.0) * 2.0, -0.4, 0.5)

        equities = 52.0 - 24.0 * valuation_risk - 16.0 * inverted_curve + 12.0 * bond_yield_signal + 10.0 * trend_risk - 8.0 * volatility_risk
        bonds = 20.0 + 16.0 * inverted_curve + 14.0 * max(0.0, bond_yield_signal) + 8.0 * volatility_risk
        gold = 10.0 + 14.0 * gold_pref + 8.0 * max(0.0, -real_yield / 3.0) + 6.0 * volatility_risk
        cash = 5.0 + 5.0 * inverted_curve + 4.0 * volatility_risk
        crypto = 0.0

        if z < 1.5 and curve > -10 and vix < 30:
            crypto = np.clip(7.0 + 6.0 * (1.5 - z) / 1.5 + 3.0 * max(0.0, (30.0 - vix) / 30.0), 0.0, 15.0)

        if t10 > 3.0 and curve > 15:
            bonds += 3.0
            equities += 2.0

        if spg < 1.4 or real_yield < 0.0:
            gold += 4.0
            equities -= 3.0

        if equities < 12.0:
            deficit = 12.0 - equities
            equities += deficit
            cash -= deficit * 0.5
            bonds -= deficit * 0.3
            gold -= deficit * 0.2

        weights = {
            "equities": np.clip(equities, 10.0, 70.0),
            "bonds": np.clip(bonds, 8.0, 55.0),
            "gold": np.clip(gold, 5.0, 35.0),
            "cash": np.clip(cash, 5.0, 30.0),
            "crypto": np.clip(crypto, 0.0, 15.0),
        }

        total = sum(weights.values())
        weights = {k: v / total * 100.0 for k, v in weights.items()}
        return weights

    def allocate_subcategories(self, weights: dict, metrics: dict) -> dict:
        suballoc = {}

        if weights["equities"] > 55.0 and metrics["trend_equity"] > 1.0:
            suballoc["equities"] = {"VOO": 0.45, "MSFT": 0.20, "GOOGL": 0.15, "QQQ": 0.12, "SPY": 0.08}
        elif weights["equities"] > 45.0:
            suballoc["equities"] = {"VOO": 0.50, "MSFT": 0.18, "GOOGL": 0.14, "QQQ": 0.10, "SPY": 0.08}
        else:
            suballoc["equities"] = {"VOO": 0.55, "MSFT": 0.18, "GOOGL": 0.12, "QQQ": 0.10, "SPY": 0.05}

        if metrics["ten_year_yield"] > 3.0:
            suballoc["bonds"] = {"TLT": 0.40, "IEF": 0.30, "BND": 0.20, "TIP": 0.10}
        elif metrics["ten_year_yield"] < 1.8:
            suballoc["bonds"] = {"SHY": 0.40, "BND": 0.35, "IEF": 0.15, "TIP": 0.10}
        else:
            suballoc["bonds"] = {"TLT": 0.35, "IEF": 0.30, "BND": 0.25, "TIP": 0.10}

        if metrics["sp_gold_ratio"] < 1.4 or metrics["real_yield"] < 0.0:
            suballoc["gold"] = {"Physical Gold": 0.50, "GLD": 0.50}
        else:
            suballoc["gold"] = {"GLD": 0.70, "Physical Gold": 0.30}

        if weights["crypto"] > 0.0:
            if metrics["vix"] < 25 and metrics["z_score"] < 1.6:
                suballoc["crypto"] = {"BTC": 0.80, "ETH": 0.20}
            else:
                suballoc["crypto"] = {"BTC": 0.70, "ETH": 0.30}
        else:
            suballoc["crypto"] = {}

        suballoc["cash"] = {"BIL": 1.0}
        return suballoc

    def find_similar_periods(self, prices: pd.DataFrame, fred_data: dict, current: dict, top_n=3) -> list:
        monthly = prices.resample("ME").last().dropna()
        gdp_daily = fred_data["GDP"].resample("D").ffill()
        common_idx = monthly.index.intersection(gdp_daily.index)
        market_cap_proxy = monthly.loc[common_idx, "VOO"] if "VOO" in monthly.columns else monthly.loc[common_idx, "^GSPC"]

        buffett = market_cap_proxy / gdp_daily.loc[common_idx, "GDP"]

        roll = buffett.rolling(window=2520//30, min_periods=500//30)        
        z_hist = (buffett - roll.mean()) / roll.std()

        hist = pd.DataFrame(index=common_idx)
        breakpoint()
        hist["z_score"] = z_hist
        hist["sp_gold_ratio"] = monthly.loc[common_idx, "^GSPC"] / monthly.loc[common_idx, "GLD"]
        hist["yield_curve"] = fred_data["T10Y2Y"].reindex(common_idx, method="ffill")["T10Y2Y"]
        hist["ten_year_yield"] = fred_data["DGS10"].reindex(common_idx, method="ffill")["DGS10"]
        if "^VIX" in monthly.columns:
            hist["vix"] = monthly.loc[common_idx, "^VIX"].fillna(method="ffill")
        else:
            sp_returns = monthly["^GSPC"].pct_change()
            hist["vix"] = sp_returns.rolling(window=30).std() * np.sqrt(252) * 100

        hist = hist.dropna()
        if hist.empty:
            return []

        features = ["z_score", "sp_gold_ratio", "yield_curve", "ten_year_yield", "vix"]
        hist_scaled = (hist - hist.mean()) / hist.std(ddof=0)
        current_values = np.array([current[k] for k in features])
        current_scaled = (current_values - hist.mean().values) / hist.std(ddof=0).values

        distances = np.linalg.norm(hist_scaled.values - current_scaled, axis=1)
        hist = hist.assign(distance=distances).sort_values("distance").head(top_n)

        return [
            {
                "date": idx.strftime("%Y-%m"),
                "z_score": float(row["z_score"]),
                "yield_curve": float(row["yield_curve"]),
                "ten_year_yield": float(row["ten_year_yield"]),
                "sp_gold_ratio": float(row["sp_gold_ratio"]),
                "vix": float(row["vix"]),
            }
            for idx, row in hist.iterrows()
        ]

    def build_message(self, metrics: dict, weights: dict, suballoc: dict, similar: list) -> str:
        def bar(value):
            blocks = int(round(value / 4.0))
            return "█" * blocks + "░" * (25 - blocks)

        lines = [
            f"*Macro Portfolio Signal — {metrics['as_of'].strftime('%Y-%m-%d')}*",
            "",
            "*Market Pulse*",
            f"• S&P/Gold ratio: *{metrics['sp_gold_ratio']:.2f}*",
            f"• Buffett-style z-score: *{metrics['z_score']:.2f}*",
            f"• 10Y Treasury yield: *{metrics['ten_year_yield']:.2f}%*",
            f"• 10Y-2Y curve: *{metrics['yield_curve']:.1f} bps*",
            f"• Implied real yield: *{metrics['real_yield']:.2f}%*",
            f"• VIX proxy: *{metrics['vix']:.1f}*",
            "",
            "*Recommended Asset Allocation*",
            f"▸ Equities: *{weights['equities']:.1f}%* {bar(weights['equities'])}",
            f"▸ Bonds: *{weights['bonds']:.1f}%* {bar(weights['bonds'])}",
            f"▸ Gold: *{weights['gold']:.1f}%* {bar(weights['gold'])}",
            f"▸ Crypto: *{weights['crypto']:.1f}%* {bar(weights['crypto'])}",
            f"▸ Cash: *{weights['cash']:.1f}%* {bar(weights['cash'])}",
            "",
            "*Tactical Implementation*",
        ]

        for category, plan in suballoc.items():
            if not plan:
                continue
            lines.append(f"• {category.title()}: ")
            for ticker, pct in plan.items():
                lines.append(f"    – {ticker}: {pct * 100:.0f}%")

        lines.append("")
        lines.append("*Closest Historical Analogues*")
        if similar:
            for item in similar:
                lines.append(
                    f"• {item['date']}: z={item['z_score']:.2f}, 10Y={item['ten_year_yield']:.1f}%, curve={item['yield_curve']:.0f}bps, gold ratio={item['sp_gold_ratio']:.2f}, VIX={item['vix']:.1f}"
                )
        else:
            lines.append("• No close historical analogue found")

        lines.append("")
        lines.append("_Note: Allocation engine is bond-yield sensitive and uses regime-aware risk positioning._")
        return "\n".join(lines)

    def send_telegram(self, message: str):
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            raise RuntimeError("Telegram credentials are not configured in environment variables")

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()

    def run_backtest(self, symbols=None, years=20, plot_path="run9_backtest.png"):
        symbols = symbols or MARKET_SYMBOLS
        period = f"{years+5}y"
        if self.fast:
            period = f"{min(years+5,10)}y"
            self.log_progress(f"Fast mode backtest: limiting market history to {period}")
        else:
            self.log_progress(f"Backtest market history period: {period}")
        prices = self.fetch_market_data(symbols, period=period)
        monthly = prices.resample("ME").last().dropna()

        weights_history = []
        portfolio = pd.Series(index=monthly.index, dtype=float)
        value = 1.0
        asset_returns = monthly.pct_change().fillna(0.0)

        fred_data = self.fetch_fred_data(FRED_SERIES)
        total_steps = len(monthly.index)
        for i, date in enumerate(monthly.index):
            if i % 12 == 0:
                self.log_progress(f"Backtest progress: {i+1}/{total_steps} months processed")
            slice_prices = prices.loc[:date].tail(2520)
            metrics = self.compute_metrics(slice_prices, fred_data)
            weights = self.allocate_portfolio(metrics)
            weights_history.append(pd.Series(weights, name=date))
            returns = asset_returns.loc[date, ["^GSPC", "GLD"]].copy()
            returns = returns.rename({"^GSPC": "equities", "GLD": "gold"})
            if "TLT" in monthly.columns:
                returns["bonds"] = monthly.loc[date, "TLT"] / monthly.loc[date - pd.offsets.MonthEnd(1), "TLT"] - 1 if date - pd.offsets.MonthEnd(1) in monthly.index else 0.0
            else:
                returns["bonds"] = 0.0025
            returns["cash"] = 0.001667
            returns["crypto"] = monthly.loc[date, "BTC-USD"] / monthly.loc[date - pd.offsets.MonthEnd(1), "BTC-USD"] - 1 if date - pd.offsets.MonthEnd(1) in monthly.index else 0.0

            period_return = sum(weights[a] / 100.0 * float(returns.get(a, 0.0)) for a in weights)
            value *= 1.0 + period_return
            portfolio.loc[date] = value

        weights_df = pd.DataFrame(weights_history)
        self.plot_backtest(weights_df, portfolio, plot_path)
        return weights_df, portfolio

    def plot_backtest(self, weights: pd.DataFrame, portfolio: pd.Series, plot_path: str):
        fig, ax = plt.subplots(figsize=(12, 7))
        weights.plot.area(ax=ax, cmap="tab20", alpha=0.85)
        ax.set_ylabel("Allocation %")
        ax.set_ylim(0, 100)
        ax.set_title("run9.py Regime-Aware Portfolio Allocation Backtest")
        ax.legend(loc="upper left")

        ax2 = ax.twinx()
        ax2.plot(portfolio.index, portfolio.values, color="black", linestyle="--", linewidth=2, label="Portfolio Value")
        ax2.set_ylabel("Portfolio Value")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(plot_path, dpi=150)


def main(send_telegram=False, backtest=False, fast=False, offline=False):
    allocator = MacroPortfolioAllocator(fast=fast, offline=offline)
    allocator.log_progress(f"Initializing MacroPortfolioAllocator (fast={fast}, offline={offline})")
    fred_data = allocator.fetch_fred_data(FRED_SERIES)
    allocator.log_progress("FRED data fetched successfully.")

    allocator.log_progress("Fetching market data from Yahoo Finance. This may take some time...")
    prices = allocator.fetch_market_data(MARKET_SYMBOLS)
    allocator.log_progress("Market data fetched successfully.")

    print("[INFO] Computing key financial metrics. Please wait...")
    metrics = allocator.compute_metrics(prices, fred_data)
    print("[SUCCESS] Metrics computed successfully.")
    
    print("[INFO] Allocating portfolio based on computed metrics...")
    weights = allocator.allocate_portfolio(metrics)
    print("[SUCCESS] Portfolio allocation completed.")

    print("[INFO] Allocating subcategories within portfolio...")
    suballoc = allocator.allocate_subcategories(weights, metrics)
    print("[SUCCESS] Subcategory allocation completed.")

    print("[INFO] Searching for similar historical periods. This may take a while...")
    similar = allocator.find_similar_periods(prices, fred_data, metrics, top_n=3)
    print("[SUCCESS] Similar historical periods identified.")

    allocator.log_progress("Building the final message for output...")
    message = allocator.build_message(metrics, weights, suballoc, similar)
    allocator.log_progress("Message built successfully.")

    allocator.log_progress("Saving report to file...")
    report_file = "run9_report.md"
    with open(report_file, "w") as f:
        f.write(message.replace("*", ""))
    allocator.log_progress(f"Report saved to {report_file}")

    allocator.log_progress("Displaying message in terminal...")
    print(message)

    if backtest:
        allocator.log_progress("Running backtest...")
        weights_df, portfolio = allocator.run_backtest(years=20, plot_path="run9_backtest.png")
        allocator.log_progress("Backtest completed.")
        allocator.log_progress(f"Final portfolio value: {portfolio.iloc[-1]:.3f}")
        allocator.log_progress("Backtest plot saved to run9_backtest.png")

    if send_telegram:
        allocator.log_progress("Sending Telegram message...")
        allocator.send_telegram("\n".join(allocator.progress_log) + "\n\n" + message)
        allocator.log_progress("Telegram message dispatched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run9 macro portfolio allocator")
    parser.add_argument("--send", action="store_true", help="Send summary to Telegram")
    parser.add_argument("--backtest", action="store_true", help="Run a 20-year backtest")
    parser.add_argument("--fast", action="store_true", help="Use shorter periods and quicker timeouts for testing")
    parser.add_argument("--offline", action="store_true", help="Use cached data only (no network)")
    args = parser.parse_args()
    main(send_telegram=args.send, backtest=args.backtest, fast=args.fast, offline=args.offline)
