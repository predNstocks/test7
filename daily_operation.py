"""
Daily strategy operation system.
Manages a live portfolio, executes daily trades, tracks cash injections,
and generates markdown reports with allocation/performance charts.

Commands:
  python3 daily_operation.py --init [--cash 100000] [--portfolio '{"SPY": 0}']
  python3 daily_operation.py --run [--budget 500]
  python3 daily_operation.py --inject 50000
  python3 daily_operation.py --report-only
"""
import sys, os, json, csv
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio import Portfolio
from strategy_all_8 import (
    compute_weights, continuous_allocation, sigmoid,
    ATH_PARAMS, ATH_GATE, SMA_PARAMS, ALL_ASSETS, EQUITY_ASSETS, INTL_ASSETS,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, 'config.json')
INJECTIONS_FILE = os.path.join(BASE_DIR, 'injections.csv')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Config ─────────────────────────────────────────────

@dataclass
class BudgetConfig:
    max_daily_exchange: int = 500
    currency: str = "USD"

@dataclass
class DataConfig:
    cache_dir: str = "data_cache"

@dataclass
class ReportConfig:
    price_freshness_days: int = 7
    indicator_freshness_days: int = 60
    cape_freshness_days: int = 90

@dataclass
class InitialPortfolioConfig:
    cash: int = 100000

@dataclass
class AppConfig:
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    initial_portfolio: InitialPortfolioConfig = field(default_factory=InitialPortfolioConfig)

    @classmethod
    def load(cls, path=None):
        path = path or CONFIG_FILE
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            raw = json.load(f)
        budget = BudgetConfig(**raw.get('budget', {}))
        data = DataConfig(**raw.get('data', {}))
        report = ReportConfig(**raw.get('report', {}))
        initial = InitialPortfolioConfig(**raw.get('initial_portfolio', {}))
        return cls(budget=budget, data=data, report=report, initial_portfolio=initial)

    def to_dict(self):
        return asdict(self)

    def to_markdown(self):
        lines = []
        lines.append('## ⚙️ Configuration')
        lines.append('')
        lines.append(f'| Setting | Value |')
        lines.append(f'|---|---|')
        lines.append(f'| Max Daily Exchange | ${self.budget.max_daily_exchange:,} {self.budget.currency} |')
        lines.append(f'| Data Cache | {self.data.cache_dir} |')
        lines.append(f'| Price Freshness Threshold | {self.report.price_freshness_days}d |')
        lines.append(f'| Indicator Freshness Threshold | {self.report.indicator_freshness_days}d |')
        lines.append(f'| CAPE Freshness Threshold | {self.report.cape_freshness_days}d |')
        lines.append(f'| Initial Portfolio Cash | ${self.initial_portfolio.cash:,} |')
        lines.append('')
        return '\n'.join(lines)


# ── Globals ────────────────────────────────────────────

CONFIG = AppConfig.load()
DATA_DIR = CONFIG.data.cache_dir
TODAY = date.today()
_price_cache = {}


# ── Data Loading ───────────────────────────────────────

def load_cache_csv(filename):
    p = os.path.join(DATA_DIR, filename)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def get_latest_price(ticker, fallback_date=None):
    if ticker not in _price_cache or _price_cache[ticker] is None:
        df = load_cache_csv(f'YF_{ticker}.csv')
        if df is None:
            return None, None
        _price_cache[ticker] = df.sort_values('Date')
    df = _price_cache[ticker]
    if fallback_date:
        mask = df['Date'] <= pd.Timestamp(fallback_date)
        if mask.any():
            row = df[mask].iloc[-1]
            return row['Close'], row['Date']
    latest = df.iloc[-1]
    return latest['Close'], latest['Date']


def update_prices_from_yfinance():
    """Fetch missing daily prices from yfinance and append to cache CSVs."""
    import yfinance as yf
    summary = []
    for a in ALL_ASSETS:
        df = load_cache_csv(f'YF_{a}.csv')
        if df is None:
            ticker = yf.Ticker(a)
            hist = ticker.history(period='max')
            if hist.empty:
                summary.append(f'{a}: no data')
                continue
            hist = hist.reset_index()
            hist['Date'] = hist['Date'].dt.tz_localize(None)
            out = hist[['Date', 'Close']].copy()
            p = os.path.join(DATA_DIR, f'YF_{a}.csv')
            out.to_csv(p, index=False)
            summary.append(f'{a}: fetched {len(out)} rows')
            if a in _price_cache:
                _price_cache[a] = None
            continue

        last_date = df['Date'].max()
        days_missing = (TODAY - last_date.date()).days
        if days_missing <= 0:
            summary.append(f'{a}: up to date')
            continue

        ticker = yf.Ticker(a)
        start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        hist = ticker.history(start=start)
        if hist.empty:
            summary.append(f'{a}: no new data (last: {last_date.date()})')
            continue

        hist = hist.reset_index()
        hist['Date'] = hist['Date'].dt.tz_localize(None)
        new_rows = hist[['Date', 'Close']].copy()

        p = os.path.join(DATA_DIR, f'YF_{a}.csv')
        new_rows.to_csv(p, mode='a', header=False, index=False)
        summary.append(f'{a}: +{len(new_rows)} rows (now through {new_rows["Date"].max().date()})')

        if a in _price_cache:
            _price_cache[a] = None

    return summary


# ── Freshness Checks ───────────────────────────────────

def check_data_freshness():
    lines = []
    all_ok = True
    for a in ALL_ASSETS:
        df = load_cache_csv(f'YF_{a}.csv')
        if df is None:
            lines.append(f'| {a} | MISSING | | |')
            all_ok = False
        else:
            last = df['Date'].max().date()
            age = (TODAY - last).days
            ok = age <= CONFIG.report.price_freshness_days
            if not ok:
                all_ok = False
            lines.append(f'| {a} | {"OK" if ok else "STALE"} | {last} | {age}d |')
    return all_ok, lines


def load_indicator(name, filename, shift_days=45, cape=False):
    p = os.path.join(DATA_DIR, filename)
    if not os.path.exists(p):
        return None, None
    if cape:
        df = pd.read_csv(p)
        df['Date'] = pd.to_datetime(df['Date'])
        last = df.iloc[-1]
        return last['PE10'], last['Date']
    df = pd.read_csv(p, header=None, names=['D', 'V'], skiprows=1)
    df['D'] = pd.to_datetime(df['D'])
    df['V'] = pd.to_numeric(df['V'], errors='coerce')
    df = df.dropna()
    last = df.iloc[-1]
    return last['V'], last['D']


def check_indicator_freshness():
    lines = []
    all_ok = True
    indicators = [
        ('CPI', 'FRED_CPIAUCSL.csv', 45, False),
        ('CAPE', 'SHILLER_CAPE.csv', 0, True),
        ('UNRATE', 'FRED_UNRATE.csv', 45, False),
        ('T10Y2Y', 'FRED_T10Y2Y.csv', 45, False),
        ('FEDFUNDS', 'FRED_FEDFUNDS.csv', 45, False),
        ('INDPRO', 'FRED_INDPRO.csv', 45, False),
    ]
    for name, fn, shift, is_cape in indicators:
        val, dt = load_indicator(name, fn, shift, cape=is_cape)
        if dt is None:
            lines.append(f'| {name} | MISSING | | |')
            all_ok = False
        else:
            d = dt.date() if hasattr(dt, 'date') else dt
            age = (TODAY - d).days
            threshold = CONFIG.report.cape_freshness_days if name == 'CAPE' else CONFIG.report.indicator_freshness_days
            ok = age <= threshold
            if not ok:
                all_ok = False
            lines.append(f'| {name} | {"OK" if ok else "STALE"} | {d} | {age}d |')
    return all_ok, lines


# ── Strategy Target ────────────────────────────────────

def _get_data_safe():
    import importlib
    try:
        mod = importlib.import_module('strategy_all_8')
        return mod._get_data()
    except Exception:
        return None


def compute_todays_target():
    data = _get_data_safe()
    if data is None:
        return None

    m = data['m']
    prices = data['prices']
    ath_series = data['ath_series']

    latest = m.iloc[-1]
    prev = m.iloc[-2] if len(m) > 1 else latest

    cp = prev['CPI_YoY'] if pd.notna(prev.get('CPI_YoY')) else 2
    ce = prev['CAPE'] if pd.notna(prev.get('CAPE')) else 25
    ur = prev['UNRATE'] if pd.notna(prev.get('UNRATE')) else 5
    yi = prev['yield_inv'] if pd.notna(prev.get('yield_inv')) else 0
    rf = prev['FEDFUNDS_real'] if pd.notna(prev.get('FEDFUNDS_real')) else 0
    ip = prev['INDPRO_growth'] if pd.notna(prev.get('INDPRO_growth')) else 0
    rs, mod = continuous_allocation(cp, ce, ur, yi, rf, ip)

    gate_val = 1.0
    if ATH_GATE['enabled'] and pd.notna(prev.get('dCPI_YoY')):
        gate_val = sigmoid(-prev['dCPI_YoY'] + ATH_GATE['thresh'], 0, ATH_GATE['k'])

    ath_factors = {}
    for a in ALL_ASSETS:
        param = ATH_PARAMS.get(a, 0)
        factor = 1.0
        if param > 0 and prev.name in ath_series[a].index:
            ath_v = ath_series[a].loc[prev.name]
            pr = prices[a].loc[prev.name]
            if pr > 0 and ath_v > pr:
                factor = 1 + (ath_v - pr) / pr * param
        if a in EQUITY_ASSETS:
            factor = 1 + (factor - 1) * gate_val
        ath_factors[a] = factor

    w = compute_weights(rs, mod, ce, ath_factors)

    for a in ALL_ASSETS:
        if a in prices:
            series = prices[a].dropna()
            if len(series) > SMA_PARAMS['period']:
                sma = series.rolling(SMA_PARAMS['period']).mean().iloc[-1]
                px = series.iloc[-1]
                if pd.notna(sma) and sma > 0 and pd.notna(px):
                    w[a] *= sigmoid(px / sma, 1.0, SMA_PARAMS['k'])

    t2 = sum(w.values())
    if t2 > 0:
        w = {a: v / t2 for a, v in w.items()}

    return w, {
        'rs': rs, 'm': mod,
        'raw': {
            'cpi_yoy': cp, 'cape': ce, 'unrate': ur,
            'yield_inv': yi, 'fedfunds_real': rf, 'indpro_growth': ip,
        }
    }


# ── Trade Calculation (diff-based) ─────────────────────

def compute_diff_trades(target, current_shares, prices, budget, portfolio_value):
    """Compute daily trades using percentage-diff approach with per-asset cap.

    diff[a] = target_pct[a] - current_pct[a]
    scale = budget / sum(positive_diffs)
    trade_value[a] = diff[a] * scale
    Each trade limited to budget * |diff[a]| (per-asset cap).

    sum(buys) <= budget (before per-asset capping, equals; after, may be less).
    """
    current_pct = {}
    for a in ALL_ASSETS:
        val = current_shares.get(a, 0) * prices.get(a, 0)
        current_pct[a] = val / portfolio_value if portfolio_value > 0 else 0

    diff = {}
    for a in ALL_ASSETS:
        diff[a] = target.get(a, 0) - current_pct.get(a, 0)

    positive_sum = sum(max(0, d) for d in diff.values())
    if positive_sum <= 1e-9:
        return {}

    scale = budget / positive_sum

    trades = {}
    for a in ALL_ASSETS:
        trade_value = diff[a] * scale
        a_cap = budget * abs(diff[a])
        if trade_value > 0:
            trade_value = min(trade_value, a_cap)
        else:
            trade_value = max(trade_value, -a_cap)
        if abs(trade_value) > 0.01:
            trades[a] = trade_value

    return trades


def execute_diff_trades(trades, prices, current_shares, current_cash):
    """Apply trades to shares/cash. Returns (new_shares, new_cash)."""
    new_shares = dict(current_shares)
    new_cash = current_cash

    for a, value in trades.items():
        if a == 'CASH':
            new_cash -= value
            continue
        px = prices.get(a, 0)
        if px > 0:
            new_shares[a] = new_shares.get(a, 0) + value / px
            new_cash -= value

    return new_shares, new_cash


# ── Portfolio CSV ──────────────────────────────────────

CSV_COLUMNS = ['date', 'cash_injected', 'cash_balance'] + ALL_ASSETS + ['total_value']
CSV_FILE = os.path.join(BASE_DIR, 'daily_portfolio.csv')


def csv_exists():
    return os.path.exists(CSV_FILE)


def load_csv():
    return pd.read_csv(CSV_FILE, parse_dates=['date'])


def get_latest_row():
    df = load_csv()
    return df.iloc[-1] if len(df) > 0 else None


def record_day(date_str, cash_injected, cash_balance, shares, total_value):
    row = {
        'date': date_str,
        'cash_injected': cash_injected,
        'cash_balance': cash_balance,
    }
    for a in ALL_ASSETS:
        row[a] = shares.get(a, 0.0)
    row['total_value'] = total_value
    rdf = pd.DataFrame([row])
    if csv_exists():
        rdf.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        rdf.to_csv(CSV_FILE, index=False)


def reconstruct_portfolio(row):
    shares = {}
    cash = float(row['cash_balance'])
    for a in ALL_ASSETS:
        shares[a] = float(row[a])
    return shares, cash


def calc_portfolio_value(shares, prices):
    total = 0.0
    for a in ALL_ASSETS:
        total += shares.get(a, 0) * prices.get(a, 0)
    return total


# ── Injections ─────────────────────────────────────────

def load_injections():
    """Load planned injections from injections.csv."""
    if not os.path.exists(INJECTIONS_FILE):
        return []
    injections = []
    with open(INJECTIONS_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            injections.append({
                'date': row['date'].strip(),
                'amount': float(row['amount'].strip()),
            })
    return injections


def get_pending_injections(last_date_str, today_str):
    """Get injections that fall between last recorded date and today (inclusive)."""
    all_injections = load_injections()
    if not csv_exists():
        return all_injections

    df = load_csv()
    recorded_dates = set(str(d.date()) if hasattr(d, 'date') else str(d) for d in df['date'])

    pending = []
    for inj in all_injections:
        inj_date = inj['date']
        if inj_date > last_date_str and inj_date <= today_str and inj_date not in recorded_dates:
            pending.append(inj)

    return pending


# ── Trading Day Execution ──────────────────────────────

def run_trading_day(prices, budget, prev_shares, prev_cash, cash_injected=0):
    """Execute one day of trading using diff-based approach."""
    shares = dict(prev_shares)
    cash = prev_cash + cash_injected

    portfolio_value = calc_portfolio_value(shares, prices) + cash

    target, info = compute_todays_target()
    if target is None:
        return shares, cash, None, [], None

    trades = compute_diff_trades(target, shares, prices, budget, portfolio_value)
    new_shares, new_cash = execute_diff_trades(trades, prices, shares, cash)

    total = calc_portfolio_value(new_shares, prices) + new_cash

    trades_list = []
    for a, value in sorted(trades.items(), key=lambda x: abs(x[1]), reverse=True):
        if a != 'CASH' and abs(value) > 0.01:
            px = prices.get(a, 1)
            shares_change = value / px if px > 0 else 0
            trades_list.append((a, shares_change, value))

    return new_shares, new_cash, target, trades_list, total, info


# ── Plotting ───────────────────────────────────────────

def plot_portfolio_value(df, initial_cash):
    fig, ax = plt.subplots(figsize=(10, 5))
    dates = pd.to_datetime(df['date'])
    tv = df['total_value']
    ci = df['cash_injected'].cumsum()
    adj = tv - ci + initial_cash

    ax.plot(dates, tv, linewidth=1.5, label='Actual Value', color='#2563EB')
    ax.plot(dates, adj, linewidth=1.5, linestyle='--', label=f'Injection-Adjusted (base=${initial_cash:,.0f})', color='#DC2626')

    inj = df[df['cash_injected'] > 0]
    if len(inj) > 0:
        ax.scatter(inj['date'], inj['total_value'], color='#16A34A', s=30, zorder=5, label='Cash Injection')

    ax.axhline(y=initial_cash, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value')
    ax.set_title('Portfolio Value')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'portfolio_value.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_monthly_composition(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month').last().reset_index()
    monthly['month_str'] = monthly['month'].astype(str)

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(monthly))
    colors = ['#2563EB', '#7C3AED', '#D97706', '#059669', '#DC2626', '#0891B2', '#9333EA']
    for i, a in enumerate(ALL_ASSETS):
        vals = monthly[a].values * monthly['total_value'].values
        v = np.where(np.isnan(vals), 0, vals)
        ax.bar(monthly['month_str'], v, bottom=bottom, label=a, color=colors[i % len(colors)], width=0.8)
        bottom += v

    ax.set_ylabel('Dollar Value')
    ax.set_title('Monthly Portfolio Composition')
    ax.legend(loc='upper right', fontsize=8, ncol=4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    tick_every = max(1, len(monthly) // 12)
    for idx in range(len(monthly)):
        if idx % tick_every != 0:
            monthly.loc[monthly.index[idx], 'month_str'] = ''
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly['month_str'], rotation=45, ha='right', fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'monthly_composition.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ── Report Generation ──────────────────────────────────

def get_last_known_prices():
    px = {'CASH': 1.0}
    for a in ALL_ASSETS:
        p, _ = get_latest_price(a)
        px[a] = float(p) if p else 0.0
    return px


def get_indicator_values():
    indicators = {}
    for name, fn, shift, is_cape in [
        ('CPI', 'FRED_CPIAUCSL.csv', 45, False),
        ('CAPE', 'SHILLER_CAPE.csv', 0, True),
        ('UNRATE', 'FRED_UNRATE.csv', 45, False),
        ('T10Y2Y', 'FRED_T10Y2Y.csv', 45, False),
        ('FEDFUNDS', 'FRED_FEDFUNDS.csv', 45, False),
        ('INDPRO', 'FRED_INDPRO.csv', 45, False),
    ]:
        val, dt = load_indicator(name, fn, shift, cape=is_cape)
        indicators[name] = (val, dt.date() if hasattr(dt, 'date') else dt)
    return indicators


def generate_regime_explanation(raw_indicators, rs, m):
    """Dynamically generate regime detection explanation based on current indicator values."""
    cpi_yoy = raw_indicators.get('cpi_yoy', 2)
    cape = raw_indicators.get('cape', 25)
    unrate = raw_indicators.get('unrate', 5)
    yield_inv = raw_indicators.get('yield_inv', 0)
    fedfunds_real = raw_indicators.get('fedfunds_real', 0)
    indpro_growth = raw_indicators.get('indpro_growth', 0)

    rc = sigmoid(cpi_yoy, 3.5, 3) * 0.25
    rca = sigmoid(cape, 35, 2) * 0.25

    yf = 0.15 if yield_inv else 0.0
    ru = sigmoid(unrate, 5.5, 3) * yf
    rff = sigmoid(fedfunds_real, 1, 5) * 0.30
    rind = sigmoid(indpro_growth, 2, 5) * 0.00
    total_weight = 0.95

    if rs < 0.33:
        regime_label = 'Risk-On'
        regime_desc = 'Low macro risk — equities and risk assets favored'
    elif rs > 0.66:
        regime_label = 'Risk-Off'
        regime_desc = 'High macro risk — bonds and safe havens favored'
    else:
        regime_label = 'Neutral'
        regime_desc = 'Moderate macro risk — balanced allocation'

    lines = []
    lines.append('## Regime Detection')
    lines.append('')
    lines.append('### Risk Score Formula')
    lines.append('')
    lines.append('$$')
    lines.append(r'rs = \frac{rc \cdot w_{cpi} + rca \cdot w_{cape} + ru \cdot w_{unrate} + rff \cdot w_{fedfunds} + rind \cdot w_{indpro}}{\sum w_i}')
    lines.append('$$')
    lines.append('')
    lines.append(r'Each component: \(r_i = \sigma(value_i, c_i, k_i) \cdot w_i\) where \(\sigma(x, c, k) = \frac{1}{1 + e^{-k(x-c)}}\)')
    lines.append('')
    lines.append('| Component | Raw Value | Sigmoid(r) | Weight | Contribution |')
    lines.append('|---|---|---|---|---|')

    contribs = [
        ('CPI YoY', f'{cpi_yoy:.2f}%', f'{rc:.3f}', '0.25', rc),
        ('CAPE', f'{cape:.1f}', f'{rca:.3f}', '0.25', rca),
        ('UNRATE', f'{unrate:.2f}%', f'{ru:.3f}', f'0.15{"×inv" if yield_inv else "×0(y)"}', ru),
        ('FEDFUNDS real', f'{fedfunds_real:.2f}%', f'{rff:.3f}', '0.30', rff),
        ('INDPRO growth', f'{indpro_growth:.2f}%', f'{rind:.3f}', '0.00', rind),
    ]
    for name, raw_val, sig_val, weight, contrib in contribs:
        lines.append(f'| {name} | {raw_val} | {sig_val} | {weight} | {contrib:.3f} |')

    lines.append('')
    lines.append(f'**Risk Score (rs):** {rs:.3f} — **{regime_label}**')
    lines.append('')
    lines.append(f'**Modulation (m):** {m:.3f} _(m = \\(1 - (2rs - 1)^2\\), peaks at rs=0.5)_')
    lines.append('')
    lines.append('| Regime | rs Range | Strategy Response |')
    lines.append('|---|---|---|')
    lines.append('| Risk-On | rs < 0.33 | Equities overweight, bonds underweight |')
    lines.append('| Neutral | 0.33 ≤ rs ≤ 0.66 | Balanced equity/bond split, gold enhanced |')
    lines.append('| Risk-Off | rs > 0.66 | Bonds overweight, equities underweight, international cut |')
    lines.append('')
    lines.append(f'**Interpretation:** {regime_desc}.')
    lines.append('')
    return '\n'.join(lines)


def generate_allocation_explanation(target, rs, m, cape_val, prices):
    """Dynamically generate allocation equation explanation based on current rs/m values."""
    total_equity = max(0, 0.65 - 0.38 * rs)
    tlt_raw = 0.20 + 0.42 * rs
    shy_raw = 0.05 + 0.12 * rs
    gld_raw = 0.10 + 0.18 * m
    dbc_raw = 0.00 + 0.12 * m

    intl_factor = sigmoid(cape_val, 28, 3)
    intl_share = 0.35 * intl_factor * max(0, 1 - rs) ** 3
    domestic_share = max(0, 1 - intl_share)
    spy_raw = total_equity * domestic_share
    vxus_raw = total_equity * intl_share * 0.5
    ewy_raw = total_equity * intl_share * 0.5

    raw_pre_normalize = {
        'SPY': spy_raw, 'TLT': tlt_raw, 'GLD': gld_raw,
        'SHY': shy_raw, 'DBC': dbc_raw, 'VXUS': vxus_raw, 'EWY': ewy_raw,
    }
    tot = sum(raw_pre_normalize.values())
    if tot > 0:
        pre_ath = {a: v / tot for a, v in raw_pre_normalize.items()}
    else:
        pre_ath = dict(raw_pre_normalize)

    lines = []
    lines.append('## Allocation Equations')
    lines.append('')
    lines.append('### Base Allocation (before ATH/SMA adjustments)')
    lines.append('')
    lines.append(f'• total_equity = max(0, 0.65 − 0.38 × {rs:.3f}) = **{total_equity:.1%}**')
    lines.append(f'• TLT = 0.20 + 0.42 × {rs:.3f} = **{tlt_raw:.1%}**')
    lines.append(f'• SHY = 0.05 + 0.12 × {rs:.3f} = **{shy_raw:.1%}**')
    lines.append(f'• GLD = 0.10 + 0.18 × {m:.3f} = **{gld_raw:.1%}**')
    lines.append(f'• DBC = 0.00 + 0.12 × {m:.3f} = **{dbc_raw:.1%}**')
    lines.append('')
    lines.append(f'• intl_factor = σ(CAPE={cape_val:.1f}, center=28, k=3) = **{intl_factor:.3f}**')
    lines.append(f'• intl_share = 0.35 × {intl_factor:.3f} × max(0, 1 − {rs:.3f})³ = **{intl_share:.1%}**')
    lines.append(f'• domestic_share = 1 − {intl_share:.1%} = **{domestic_share:.1%}**')
    lines.append('')
    lines.append(f'• SPY = {total_equity:.1%} × {domestic_share:.1%} = **{spy_raw:.1%}**')
    lines.append(f'• VXUS = {total_equity:.1%} × {intl_share:.1%} × 0.5 = **{vxus_raw:.1%}**')
    lines.append(f'• EWY = {total_equity:.1%} × {intl_share:.1%} × 0.5 = **{ewy_raw:.1%}**')
    lines.append('')
    lines.append('### Normalized st8 Target')
    lines.append('')
    lines.append('| Asset | Raw | Normalized | After ATH | After SMA | Final |')
    lines.append('|---|---|---|---|---|---|')
    for a in ALL_ASSETS:
        raw_pct = raw_pre_normalize.get(a, 0) * 100
        norm_pct = pre_ath.get(a, 0) * 100
        final = target.get(a, 0) * 100
        lines.append(f'| {a} | {raw_pct:.1f}% | {norm_pct:.1f}% | — | — | {final:.1f}% |')
    lines.append('')
    lines.append('*ATH dip-buying and SMA trend filters are applied after normalization to produce the final weights.*')
    lines.append('')
    return '\n'.join(lines)


def generate_report(budget, prev_row, target, trades_list, prices, is_trading_day=True, regime_info=None):
    df = load_csv()
    prices = prices or get_last_known_prices()
    lines = []
    lines.append(f'# Daily Strategy Report -- {TODAY}')
    lines.append('')

    # Config
    lines.append(CONFIG.to_markdown())

    # Market Data
    lines.append('## Market Data')
    lines.append('')
    lines.append('### Asset Prices')
    lines.append('| Asset | Price | Date |')
    lines.append('|---|---|---|')
    for a in ALL_ASSETS:
        p, d = get_latest_price(a)
        if p:
            lines.append(f'| {a} | ${float(p):,.2f} | {d.date() if hasattr(d, "date") else d} |')
        else:
            lines.append(f'| {a} | -- | -- |')
    lines.append('')

    indicators = get_indicator_values()
    lines.append('### Indicators')
    lines.append('| Indicator | Value | Date | Role |')
    lines.append('|---|---|---|---|')
    role_map = {
        'CPI': 'CPI YoY -> allocation',
        'CAPE': 'Valuation -> equity/bond split',
        'UNRATE': 'Labor -> allocation',
        'T10Y2Y': 'Yield curve -> allocation',
        'FEDFUNDS': 'Real rate -> allocation',
        'INDPRO': 'Industrial prod -> allocation',
    }
    for name in ['CPI', 'CAPE', 'UNRATE', 'T10Y2Y', 'FEDFUNDS', 'INDPRO']:
        val, dt = indicators.get(name, (None, None))
        if val is not None:
            if name == 'CPI':
                data = _get_data_safe()
                cpi_yoy = data['m'].iloc[-1]['CPI_YoY'] if data else None
                display = f'{cpi_yoy:.2f}% YoY' if cpi_yoy else f'{val:.1f}'
            elif name == 'CAPE':
                display = f'{val:.2f}'
            elif name == 'FEDFUNDS':
                data = _get_data_safe()
                real = data['m'].iloc[-1]['FEDFUNDS_real'] if data else None
                display = f'{val:.2f}% (real: {real:.2f}%)' if real else f'{val:.2f}%'
            else:
                display = f'{val:.2f}'
            lines.append(f'| {name} | {display} | {dt} | {role_map[name]} |')
        else:
            lines.append(f'| {name} | -- | -- | {role_map[name]} |')
    lines.append('')

    # Ideal Portfolio
    if target is not None:
        lines.append('## Ideal Portfolio (st8 Target)')
        lines.append('')
        if prev_row is not None:
            pv = prev_row['total_value']
            lines.append(f'Budget: **${budget:,}/day** | Portfolio: **${pv:,.2f}** | Max trade: **${budget:,.0f}**')
        else:
            lines.append(f'Budget: **${budget:,}/day**')
        lines.append('')
        lines.append('| Asset | Target % | Target $ | Max Buy $ | Current % | Gap |')
        lines.append('|---|---|---|---|---|---|')
        for a in ALL_ASSETS:
            tgt_pct = target.get(a, 0) * 100
            if prev_row is not None:
                pv = prev_row['total_value']
                tgt_dollar = tgt_pct / 100 * pv
                curr_val = prev_row[a] * prices.get(a, 1) if a in prices else 0
                curr_pct = curr_val / pv * 100 if pv > 0 else 0
                gap = tgt_pct - curr_pct
                sign = '+' if gap >= 0 else ''
                max_gap = budget * abs(gap) / 100
                max_buy = min(max_gap, max(0, tgt_dollar - curr_val))
                lines.append(f'| {a} | {tgt_pct:.1f}% | ${tgt_dollar:>8,.0f} | ${max_buy:,.0f} | {curr_pct:.1f}% | {sign}{gap:.1f}% |')
            else:
                lines.append(f'| {a} | {tgt_pct:.1f}% | -- | -- | -- | -- |')
        if prev_row is not None:
            cash_balance = float(prev_row.get('cash_balance', 0))
            cash_pct = cash_balance / pv * 100 if pv > 0 else 0
            lines.append(f'| CASH | 0.0% | $0 | $0 | {cash_pct:.1f}% | −{cash_pct:.1f}% |')
        lines.append('')

    # Data Status
    lines.append('## Data Status')
    ok_p, price_lines = check_data_freshness()
    ok_i, ind_lines = check_indicator_freshness()
    lines.append('### Prices')
    lines.append('| Asset | Status | Latest | Age |')
    lines.append('|---|---|---|---|')
    lines.extend(price_lines)
    lines.append('')
    lines.append('### Indicators')
    lines.append('| Indicator | Status | Latest | Age |')
    lines.append('|---|---|---|---|')
    lines.extend(ind_lines)
    lines.append('')

    # Cash Injection
    ci = df['cash_injected'].sum()
    if prev_row is not None and prev_row['cash_injected'] > 0:
        if len(df) == 1:
            lines.append('## Initial Portfolio')
            lines.append(f'**${prev_row["cash_injected"]:,.0f}** initial cash. Ready for daily trading.')
        else:
            lines.append('## Cash Injection')
            lines.append(f'**${prev_row["cash_injected"]:,.0f}** added today. Cumulative injections: **${ci:,.0f}**.')
        lines.append('')

    # Trades
    if target is not None:
        lines.append('## Trades')
        lines.append(f'Budget: **${budget:,}/day**')
        lines.append('')

        if not is_trading_day or len(trades_list) == 0:
            lines.append('| Action | Asset | Shares | $ Amount | Reason |')
            lines.append('|---|---|---|---|---|')
            for a in ALL_ASSETS:
                pct = target.get(a, 0) * 100
                lines.append(f'| -- | {a} | 0 | $0 (${budget:,.0f}) | Target {pct:.1f}% |')
            lines.append(f'| | | | **$0** | No trades (off-day or converged) |')
        else:
            lines.append('| Action | Asset | Shares | $ Amount | Reason |')
            lines.append('|---|---|---|---|---|')
            total_traded = 0
            for a, sh, amt in trades_list:
                action = 'BUY' if amt > 0 else 'SELL'
                pct = target.get(a, 0) * 100
                if prev_row is not None and a in prev_row.index:
                    prev_val = prev_row[a] * prices.get(a, 1)
                    curr_val = sh * prices.get(a, 1)
                    curr_w = curr_val / prev_row['total_value'] * 100 if prev_row['total_value'] > 0 else 0
                    reason = f'Target {pct:.1f}%, actual {curr_w:.1f}%'
                else:
                    reason = f'Target {pct:.1f}%'
                lines.append(f'| {action} | {a} | {abs(sh):.4f} | ${abs(amt):>7,.2f} | {reason} |')
                total_traded += abs(amt)
            lines.append(f'| | | | **${total_traded:,.2f}** | Budget used |')
        lines.append('')

        # Regime & Allocation Explanation
        if regime_info:
            raw_i = regime_info.get('raw', {})
            rs = regime_info.get('rs', 0.5)
            m = regime_info.get('m', 0.5)
            cape_val = raw_i.get('cape', 25)
            lines.append(generate_regime_explanation(raw_i, rs, m))
            lines.append(generate_allocation_explanation(target, rs, m, cape_val, prices))

        # Current Allocation
        lines.append('## Current Allocation')
        if prev_row is not None:
            pv = prev_row['total_value']
            lines.append(f'Portfolio Value: **${pv:,.2f}**')
            lines.append('')
            lines.append('| Asset | Target | Current $ | Current % | Delta |')
            lines.append('|---|---|---|---|---|')
            for a in ALL_ASSETS:
                tgt = target.get(a, 0) * 100
                curr_val = prev_row[a] * prices.get(a, 1) if a in prices else 0
                curr_pct = curr_val / pv * 100 if pv > 0 else 0
                delta = curr_pct - tgt
                sign = '+' if delta >= 0 else ''
                lines.append(f'| {a} | {tgt:.1f}% | ${curr_val:>8,.2f} | {curr_pct:.1f}% | {sign}{delta:.1f}% |')
            cash_balance = float(prev_row.get('cash_balance', 0))
            cash_pct = cash_balance / pv * 100 if pv > 0 else 0
            lines.append(f'| CASH | 0.0% | ${cash_balance:>8,.2f} | {cash_pct:.1f}% | −{cash_pct:.1f}% |')
            lines.append('')

    # Portfolio Value
    if len(df) > 1:
        initial_cash = df['cash_injected'].iloc[0]
        vpath = plot_portfolio_value(df, initial_cash)
        lines.append('## Portfolio Value')
        lines.append(f'![Portfolio Value]({vpath})')
        lines.append('')

        mpath = plot_monthly_composition(df)
        lines.append('## Monthly Composition')
        lines.append(f'![Monthly Composition]({mpath})')
        lines.append('')

        df_dates = df['date']
        tv = df['total_value']
        tv_s = pd.Series(tv.values, index=df_dates)
        monthly = tv_s.resample('ME').last()
        monthly_rets = monthly.pct_change().dropna()
        cum = (1 + monthly_rets).cumprod()

        lines.append('## Monthly Returns')
        lines.append('| Month | Return | Cumulative |')
        lines.append('|---|---|---|')
        for idx in monthly_rets.index:
            r = monthly_rets[idx] * 100
            c = (cum[idx] - 1) * 100
            lines.append(f'| {idx.strftime("%Y-%m")} | {r:+.2f}% | {c:+.2f}% |')
        lines.append('')

    # Injection History
    injections = df[df['cash_injected'] > 0]
    if len(injections) > 0:
        lines.append('## Injection History')
        lines.append('| Date | Amount | Total Portfolio |')
        lines.append('|---|---|---|')
        for _, r in injections.iterrows():
            lines.append(f'| {r["date"].date()} | ${r["cash_injected"]:,.0f} | ${r["total_value"]:,.0f} |')
        lines.append('')

    report = '\n'.join(lines)
    report_path = os.path.join(BASE_DIR, f'report_{TODAY.strftime("%Y%m%d")}.md')
    with open(report_path, 'w') as f:
        f.write(report)
    return report, report_path


# ── CLI Commands ───────────────────────────────────────

def cmd_init(cash_amount=None, portfolio_json=None, start_date=None):
    if csv_exists():
        print('CSV already exists. Use --inject to add cash or start fresh with a new file.')
        return
    if portfolio_json:
        try:
            positions = json.loads(portfolio_json)
        except json.JSONDecodeError:
            print('Invalid JSON for --portfolio')
            return
    else:
        cash_amount = cash_amount or CONFIG.initial_portfolio.cash
        positions = {'CASH': cash_amount}

    cash = positions.get('CASH', 0)
    shares = {}
    for a in ALL_ASSETS:
        shares[a] = positions.get(a, 0.0)

    total_value = cash
    for a in ALL_ASSETS:
        if shares[a] > 0:
            p, _ = get_latest_price(a)
            if p:
                total_value += shares[a] * float(p)

    init_date = start_date if start_date else TODAY
    record_day(init_date.isoformat(), cash_amount if cash_amount else cash, cash, shares, total_value)
    print(f'Initialized: ${total_value:,.2f} portfolio ({cash_amount if cash_amount else cash:,.0f} cash)')
    print(f'   Date: {init_date}')
    print(f'   CSV: {CSV_FILE}')


def cmd_run(budget=None):
    budget = budget or CONFIG.budget.max_daily_exchange

    if not csv_exists():
        print('No portfolio CSV. Run --init first.')
        return

    print('Checking for price updates...')
    update_summary = update_prices_from_yfinance()
    for s in update_summary:
        print(f'   {s}')

    df = load_csv()
    last = df.iloc[-1]
    last_date = last['date']
    shares, cash = reconstruct_portfolio(last)

    if isinstance(last_date, pd.Timestamp):
        ld = last_date.date()
    elif isinstance(last_date, str):
        ld = datetime.fromisoformat(last_date).date()
    else:
        ld = last_date

    last_date_str = ld.isoformat()
    today_str = TODAY.isoformat()

    if ld >= TODAY:
        print(f'Today ({TODAY}) already recorded. Regenerating report...')
        target, info = compute_todays_target()
        prices = get_last_known_prices()
        report, path = generate_report(budget, last, target, [], prices, is_trading_day=False, regime_info=info)
        print(report)
        print(f'\nReport: {path}')
        return

    all_dates = pd.bdate_range(ld + timedelta(days=1), TODAY)
    if len(all_dates) == 0:
        print('No new trading days to process.')
        target, info = compute_todays_target()
        prices = get_last_known_prices()
        report, path = generate_report(budget, last, target, [], prices, is_trading_day=False, regime_info=info)
        print(report)
        print(f'\nReport: {path}')
        return

    price_lookup = {}
    for a in ALL_ASSETS:
        price_lookup[a] = {}
        df_p = load_cache_csv(f'YF_{a}.csv')
        if df_p is not None:
            for _, r in df_p.iterrows():
                price_lookup[a][r['Date'].date()] = r['Close']
        all_dates_with_data = sorted(price_lookup[a].keys())
        price_lookup[a]['_last'] = price_lookup[a].get(all_dates_with_data[-1], 0) if all_dates_with_data else 0

    def get_price(a, d):
        d = d.date() if hasattr(d, 'date') else d
        px = price_lookup.get(a, {}).get(d)
        if px is not None and pd.notna(px) and float(px) > 0:
            return float(px)
        return float(price_lookup.get(a, {}).get('_last', 0))

    current_shares = dict(shares)
    current_cash = cash

    pending_injections = get_pending_injections(last_date_str, today_str)
    inj_by_date = {inj['date']: inj['amount'] for inj in pending_injections}

    print(f'Processing {len(all_dates)} day(s)...')
    last_target = None
    last_trades = []
    last_day_prices = {}
    last_regime_info = None
    for d in all_dates:
        d_date = d.date() if hasattr(d, 'date') else d
        d_str = d_date.isoformat()

        day_prices = {a: get_price(a, d_date) for a in ALL_ASSETS}
        has_price = any(v > 0 for v in day_prices.values())

        if not has_price:
            print(f'  {d_date}: no price data (weekend/holiday), skipping')
            continue

        cash_inj = inj_by_date.get(d_str, 0)
        if cash_inj > 0:
            print(f'  + Injecting ${cash_inj:,.0f} on {d_date}')

        result = run_trading_day(
            day_prices, budget, current_shares, current_cash, cash_injected=cash_inj
        )
        new_shares, new_cash, target, trades_list, total, info = result
        if total is None:
            print(f'  {d_date}: strategy data not available, skipping')
            continue

        record_day(d_str, cash_inj, new_cash, new_shares, total)
        current_shares = new_shares
        current_cash = new_cash
        last_target = target
        last_trades = trades_list
        last_day_prices = day_prices
        last_regime_info = info
        print(f'  {d_date}: ${total:,.2f} (trades: {len(trades_list)})')

    last_row = get_latest_row()
    is_trading = last_target is not None
    report, path = generate_report(budget, last_row, last_target, last_trades, last_day_prices,
                                    is_trading_day=is_trading, regime_info=last_regime_info)
    print(f'\nReport: {path}')
    print(report[:2000] + ('\n... (truncated)' if len(report) > 2000 else ''))


def cmd_inject(amount):
    if not csv_exists():
        print('No portfolio CSV. Run --init first.')
        return
    amount = float(amount)
    if amount <= 0:
        print('Amount must be positive.')
        return
    last = get_latest_row()
    shares, cash = reconstruct_portfolio(last)
    new_cash = cash + amount

    day_prices = {'CASH': 1.0}
    for a in ALL_ASSETS:
        p, _ = get_latest_price(a)
        day_prices[a] = float(p) if p else 0.0

    total = calc_portfolio_value(shares, day_prices) + new_cash

    record_day(TODAY.isoformat(), amount, new_cash, shares, total)
    print(f'Injected ${amount:,.0f}. New cash: ${new_cash:,.0f}. Total: ${total:,.0f}.')


def cmd_report_only():
    if not csv_exists():
        print('No portfolio CSV.')
        return
    last = get_latest_row()
    day_prices = {'CASH': 1.0}
    for a in ALL_ASSETS:
        p, _ = get_latest_price(a)
        day_prices[a] = float(p) if p else 0.0

    result = compute_todays_target()
    target = result[0] if result else None
    regime_info = result[1] if result else None
    report, path = generate_report(CONFIG.budget.max_daily_exchange, last, target, [], day_prices,
                                    is_trading_day=True, regime_info=regime_info)
    print(report)
    print(f'\nReport: {path}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == '--init':
        cash = None
        portfolio = None
        start_date = None
        for i in range(2, len(sys.argv)):
            if sys.argv[i] == '--cash' and i + 1 < len(sys.argv):
                cash = float(sys.argv[i + 1])
            elif sys.argv[i] == '--start-date' and i + 1 < len(sys.argv):
                start_date = datetime.strptime(sys.argv[i + 1], '%Y-%m-%d').date()
            elif sys.argv[i] == '--portfolio' and i + 1 < len(sys.argv):
                portfolio = sys.argv[i + 1]
        cmd_init(cash_amount=cash, portfolio_json=portfolio, start_date=start_date)
    elif cmd == '--run':
        budget = CONFIG.budget.max_daily_exchange
        for i in range(2, len(sys.argv)):
            if sys.argv[i] == '--budget' and i + 1 < len(sys.argv):
                budget = float(sys.argv[i + 1])
        cmd_run(budget=budget)
    elif cmd == '--inject' and len(sys.argv) > 2:
        cmd_inject(sys.argv[2])
    elif cmd == '--report-only':
        cmd_report_only()
    else:
        print(f'Unknown: {cmd}')
        print(__doc__)
