"""
8-asset strategy: st7 + risk-gated international rotation.

Key innovation over st7: international share is gated by risk score (rs).
  intl_share *= max(0, 1 - rs) ** INTL_RISK_POWER

When macro risk is high (rs high), international allocation is aggressively cut.
When macro risk is low (rs low), full CAPE-based international rotation is allowed.

This prevents allocating to high-beta international equities during risk-off periods
while preserving the CAPE-based valuation rotation during risk-on periods.

Usage:
    python3 strategy_all_8.py history      # backtest
    python3 strategy_all_8.py current      # current allocation
    python3 strategy_all_8.py optimize     # grid search params
"""
import pandas as pd
import numpy as np
import os, sys, itertools
from typing import Dict, Optional

DATA_DIR = os.environ.get('DATA_CACHE_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache'))
ALL_ASSETS = ['SPY', 'TLT', 'GLD', 'SHY', 'DBC', 'VXUS', 'EWY']
INTL_ASSETS = ['VXUS', 'EWY']
EQUITY_ASSETS = ['SPY', 'VXUS', 'EWY']

# st7 parameters (unchanged from final optimized)
INDICATORS = {
    'CPI':       {'c': 3.5, 'k': 3, 'w': 0.25},
    'CAPE':      {'c': 35,  'k': 2, 'w': 0.25},
    'UNRATE':    {'c': 5.5, 'k': 3, 'w': 0.15},
    'FEDFUNDS_real': {'c': 1, 'k': 5, 'w': 0.30},
    'INDPRO':    {'c': 2,  'k': 5, 'w': 0.00},
}

ATH_PARAMS = {
    'SPY': 14, 'TLT': 0, 'GLD': 0, 'SHY': 0, 'DBC': 3.0,
    'VXUS': 0, 'EWY': 0,
}

ATH_GATE = {'enabled': True, 'metric': 'dCPI_YoY', 'thresh': 0.0, 'k': 30}
SMA_PARAMS = {'period': 220, 'k': 70}

# CAPE-based international split
INTL_ROTATION = {'center': 28, 'k': 3.0, 'max_share': 0.35}

# st8: risk-gated international rotation
# intl_share *= max(0, 1 - risk_score) ** INTL_RISK_POWER
# power=0: no gate (st7 behavior)
# power=3: aggressive cut during risk-off
INTL_RISK_POWER = 3.0

# EWY share of international allocation (rest goes to VXUS)
# 0.5 = balanced split, 1.0 = all to EWY
EWY_SHARE = 0.5


def sigmoid(x, c=1.0, k=10.0):
    return 1 / (1 + np.exp(-k * (x - c)))


# ── Data loading ──────────────────────────────────────────

def load_fred(name, filename, shift_days=45):
    p = f'{DATA_DIR}/{filename}'
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, header=None, names=['D', 'V'], skiprows=1)
    df['D'] = pd.to_datetime(df['D']) + pd.DateOffset(days=shift_days)
    df['V'] = pd.to_numeric(df['V'], errors='coerce')
    return df.dropna().set_index('D')['V']


def load_price(ticker):
    p = f'{DATA_DIR}/YF_{ticker}.csv'
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index('Date')['Close']


def load_cape():
    p = f'{DATA_DIR}/SHILLER_CAPE.csv'
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df['Date'] = pd.to_datetime(df['Date'])
    cape = df.set_index('Date')['PE10']
    cape = pd.to_numeric(cape, errors='coerce').dropna()
    return cape[cape > 0]


# ── Core strategy ─────────────────────────────────────────

def continuous_allocation(cpi_yoy, cape, unrate, yield_inv, fedfunds_real, indpro_growth):
    rc = sigmoid(cpi_yoy, INDICATORS['CPI']['c'], INDICATORS['CPI']['k']) * INDICATORS['CPI']['w']
    rca = sigmoid(cape, INDICATORS['CAPE']['c'], INDICATORS['CAPE']['k']) * INDICATORS['CAPE']['w']
    ru = sigmoid(unrate, INDICATORS['UNRATE']['c'], INDICATORS['UNRATE']['k']) * yield_inv * INDICATORS['UNRATE']['w']
    rff = sigmoid(fedfunds_real, INDICATORS['FEDFUNDS_real']['c'], INDICATORS['FEDFUNDS_real']['k']) * INDICATORS['FEDFUNDS_real']['w'] if pd.notna(fedfunds_real) else 0
    rind = sigmoid(indpro_growth, INDICATORS['INDPRO']['c'], INDICATORS['INDPRO']['k']) * INDICATORS['INDPRO']['w'] if pd.notna(indpro_growth) else 0

    tw = sum(v['w'] for v in INDICATORS.values())
    rs = max(0, min(1, (rc + rca + ru + rff + rind) / tw))
    m = 1 - (2 * rs - 1) ** 2
    return rs, m


def compute_weights(rs, m, cape_val, ath_factors=None,
                    intl_risk_power=None, ewy_share=None,
                    intl_max_share=None):
    """Compute weights with risk-gated international rotation."""
    if intl_risk_power is None:
        intl_risk_power = INTL_RISK_POWER
    if ewy_share is None:
        ewy_share = EWY_SHARE
    if intl_max_share is None:
        intl_max_share = INTL_ROTATION['max_share']

    total_equity = max(0, 0.65 - 0.38 * rs)

    # CAPE-based international split
    intl_factor = sigmoid(cape_val, INTL_ROTATION['center'], INTL_ROTATION['k'])
    intl_share = intl_max_share * intl_factor

    # st8 RISK GATE: aggressively cut international when macro is risky
    intl_share *= max(0, 1 - rs) ** intl_risk_power

    domestic_share = max(0, 1 - intl_share)

    raw = {
        'SPY': total_equity * domestic_share,
        'TLT': 0.20 + 0.42 * rs,
        'SHY': 0.05 + 0.12 * rs,
        'GLD': 0.10 + 0.18 * m,
        'DBC': 0.00 + 0.12 * m,
        'VXUS': total_equity * intl_share * (1 - ewy_share),
        'EWY': total_equity * intl_share * ewy_share,
    }
    tot = sum(raw.values())
    w = {k: max(0, v / tot) for k, v in raw.items()}

    if ath_factors:
        for a, factor in ath_factors.items():
            if factor > 0 and a in w:
                w[a] = max(0, w[a] * factor)

    t2 = sum(w[a] for a in ALL_ASSETS)
    if t2 > 0:
        for a in ALL_ASSETS:
            w[a] /= t2

    return w


# ── Backtest ──────────────────────────────────────────────

def _build_data():
    fred = {}
    for n, fn in [('CPI', 'FRED_CPIAUCSL.csv'), ('UNRATE', 'FRED_UNRATE.csv'),
                  ('T10Y2Y', 'FRED_T10Y2Y.csv'), ('FEDFUNDS', 'FRED_FEDFUNDS.csv'),
                  ('INDPRO', 'FRED_INDPRO.csv')]:
        fred[n] = load_fred(n, fn)
    if any(v is None for v in fred.values()):
        return None

    prices = {}
    for a in ALL_ASSETS:
        p = load_price(a)
        if p is None:
            return None
        prices[a] = p

    cape = load_cape()
    if cape is None:
        return None

    m = pd.DataFrame(index=prices['SPY'].resample('ME').last().index)
    for a in ALL_ASSETS:
        m[a] = prices[a].resample('ME').last()
    for a in ALL_ASSETS:
        m[f'{a}_r'] = m[a].pct_change()
    for n, s in fred.items():
        m[n] = s.reindex(m.index, method='ffill')
    m['CPI_YoY'] = m['CPI'].pct_change(12) * 100
    m['yield_inv'] = (m['T10Y2Y'] < 0).astype(int)
    m['CAPE'] = cape.reindex(m.index, method='ffill')
    m['FEDFUNDS_real'] = m['FEDFUNDS'] - m['CPI_YoY']
    m['INDPRO_growth'] = m['INDPRO'].pct_change(12) * 100
    m['dCPI_YoY'] = m['CPI_YoY'].diff()

    ath_series = {a: prices[a].expanding().max() for a in ALL_ASSETS}
    monthly_prices = {a: prices[a].resample('ME').last() for a in ALL_ASSETS}
    sma_cache = {}

    return {'m': m, 'prices': prices, 'monthly_prices': monthly_prices,
            'ath_series': ath_series, 'sma_cache': sma_cache}


_DATA_CACHE = None

def _get_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = _build_data()
    return _DATA_CACHE


def _get_sma_series(data, period):
    sma_cache = data['sma_cache']
    if period not in sma_cache:
        prices = data['prices']
        raw = {a: prices[a].rolling(period).mean() for a in ALL_ASSETS}
        monthly = {a: raw[a].resample('ME').last() for a in ALL_ASSETS}
        sma_cache[period] = (raw, monthly)
    return sma_cache[period]


def backtest(start_year=2000, end_year=2024,
             ath_params=None, gate=None, sma_params=None,
             intl_risk_power=None, ewy_share=None,
             intl_max_share=None,
             verbose=True):
    if ath_params is None:
        ath_params = ATH_PARAMS
    if gate is None:
        gate = ATH_GATE
    if sma_params is None:
        sma_params = SMA_PARAMS
    if intl_risk_power is None:
        intl_risk_power = INTL_RISK_POWER
    if ewy_share is None:
        ewy_share = EWY_SHARE
    if intl_max_share is None:
        intl_max_share = INTL_ROTATION['max_share']

    data = _get_data()
    if data is None:
        if verbose: print('ERROR: Missing data')
        return None

    m = data['m']
    prices = data['prices']
    ath_series = data['ath_series']
    sma_period = sma_params['period']
    sma_raw, sma_monthly = _get_sma_series(data, sma_period)
    monthly_prices = data.get('monthly_prices', {})

    sr = []
    dates = []
    violations = []

    for i in range(1, len(m)):
        prev, cur = m.iloc[i - 1], m.iloc[i]
        if prev.name.year < start_year:
            continue
        if prev.name.year > end_year:
            break

        cp = prev['CPI_YoY'] if pd.notna(prev['CPI_YoY']) else 2
        ce = prev['CAPE'] if pd.notna(prev.get('CAPE')) else 25
        ur = prev['UNRATE'] if pd.notna(prev['UNRATE']) else 5
        yi = prev['yield_inv'] if pd.notna(prev['yield_inv']) else 0
        rf = prev['FEDFUNDS_real'] if pd.notna(prev.get('FEDFUNDS_real')) else 0
        ip = prev['INDPRO_growth'] if pd.notna(prev.get('INDPRO_growth')) else 0

        rs, mod = continuous_allocation(cp, ce, ur, yi, rf, ip)

        def inflation_gate(dcpi_yoy):
            if not gate['enabled'] or pd.isna(dcpi_yoy):
                return 1.0
            return sigmoid(-dcpi_yoy + gate['thresh'], 0, gate['k'])

        gate_val = inflation_gate(prev.get('dCPI_YoY', 0))

        def get_ath_factor(a, date, series, param):
            if param <= 0 or date not in series.index:
                return 1.0
            ath_v = series.loc[date]
            pr = prices[a].loc[date]
            if pr <= 0 or ath_v <= pr:
                return 1.0
            return 1 + (ath_v - pr) / pr * param

        ath_factors = {}
        for a in ALL_ASSETS:
            factor = get_ath_factor(a, prev.name, ath_series[a], ath_params.get(a, 0))
            if a in EQUITY_ASSETS:
                factor = 1 + (factor - 1) * gate_val
            ath_factors[a] = factor

        w = compute_weights(rs, mod, ce, ath_factors,
                             intl_risk_power=intl_risk_power,
                             ewy_share=ewy_share,
                             intl_max_share=intl_max_share)

        for a in list(w.keys()):
            rn = f'{a}_r'
            if rn not in cur or pd.isna(cur[rn]):
                continue
            if a in sma_monthly and prev.name in sma_monthly[a].shift(1).index and \
               a in monthly_prices and prev.name in monthly_prices[a].shift(1).index:
                sma_val = sma_monthly[a].shift(1).loc[prev.name]
                price_val = monthly_prices[a].shift(1).loc[prev.name]
                if pd.notna(sma_val) and sma_val > 0 and pd.notna(price_val):
                    w[a] *= sigmoid(price_val / sma_val, 1.0, sma_params['k'])

        valid_w = {a: w[a] for a in list(w.keys())
                   if f'{a}_r' in cur and pd.notna(cur[f'{a}_r'])}
        tw = sum(valid_w.values())
        if tw > 0:
            valid_w = {a: v / tw for a, v in valid_w.items()}

        ret = sum(valid_w.get(a, 0.0) * cur[f'{a}_r']
                  for a in valid_w if f'{a}_r' in cur and pd.notna(cur[f'{a}_r']))

        wsum = sum(valid_w.values())
        if abs(wsum - 1.0) > 0.01:
            violations.append(f"{prev.name.date()}: weights sum to {wsum:.4f}")

        if not np.isnan(ret):
            sr.append(ret)
            dates.append(cur.name)

    if violations and verbose:
        print(f"⚠ {len(violations)} weight-sum violations (first 3):")
        for v in violations[:3]:
            print(f"  {v}")

    if not sr or np.std(sr) == 0:
        if verbose: print('ERROR: No valid returns')
        return None

    s = pd.Series(sr, index=dates)
    tr = float((1 + s).prod() - 1)
    sh = float(np.mean(s) / np.std(s) * np.sqrt(12))
    cum = (1 + s).cumprod()
    dd = float((cum / cum.cummax() - 1).min())
    return sh, tr, dd


# ── Optimizer ─────────────────────────────────────────────

def optimize_risk_power():
    """Grid search over INTL_RISK_POWER, EWY_SHARE, and INTL_MAX_SHARE."""
    print(f"\n{'=' * 120}")
    print("ST8 3D OPTIMIZATION: INTL_RISK_POWER × EWY_SHARE × INTL_MAX_SHARE")
    print(f"{'=' * 120}")

    powers = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    ewys = [0.0, 0.25, 0.5, 0.75, 1.0]
    max_shares = [0.25, 0.35, 0.50]

    best_score = 0
    best_config = None
    all_results = []

    for ms in max_shares:
        print(f"\n--- INTL_MAX_SHARE = {ms:.2f} ---")
        print(f"{'Power':>6s} {'EWY_Sh':>7s} {'Sh14':>7s} {'Ret14':>8s} {'DD14':>6s} {'Sh25':>7s} {'Ret25':>8s} {'DD25':>6s} {'Sh60':>7s} {'Ret60':>8s} {'DD60':>6s} {'Score':>7s}")
        print('-' * 100)
        for power in powers:
            for ew in ewys:
                r14 = backtest(2011, 2024, intl_risk_power=power, ewy_share=ew,
                               intl_max_share=ms, verbose=False)
                r25 = backtest(2000, 2024, intl_risk_power=power, ewy_share=ew,
                               intl_max_share=ms, verbose=False)
                r60 = backtest(1965, 2024, intl_risk_power=power, ewy_share=ew,
                               intl_max_share=ms, verbose=False)
                if r14 is None or r25 is None or r60 is None:
                    continue

                sh14, tr14, dd14 = r14
                sh25, tr25, dd25 = r25
                sh60, tr60, dd60 = r60

                score = (sh14 + sh25 + sh60) / 3
                if dd14 < -0.20 or dd25 < -0.25 or dd60 < -0.25:
                    score *= 0.5

                all_results.append((power, ew, ms, sh14, tr14, dd14, sh25, tr25, dd25, sh60, tr60, dd60, score))
                print(f"{power:6.1f} {ew:7.2f} {sh14:7.4f} {tr14*100:+8.1f}% {dd14*100:6.0f}% {sh25:7.4f} {tr25*100:+8.1f}% {dd25*100:6.0f}% {sh60:7.4f} {tr60*100:+8.1f}% {dd60*100:6.0f}% {score:7.4f}")

                if score > best_score:
                    best_score = score
                    best_config = (power, ew, ms, sh14, tr14, dd14, sh25, tr25, dd25, sh60, tr60, dd60)

    if best_config:
        print(f"\n{'=' * 120}")
        print(f"Best: power={best_config[0]}, EWY_share={best_config[1]}, max_share={best_config[2]}")
        print(f"  14yr: Sharpe={best_config[3]:.4f}, Ret={best_config[4]*100:+.1f}%, DD={best_config[5]*100:.0f}%")
        print(f"  25yr: Sharpe={best_config[6]:.4f}, Ret={best_config[7]*100:+.1f}%, DD={best_config[8]*100:.0f}%")
        print(f"  60yr: Sharpe={best_config[9]:.4f}, Ret={best_config[10]*100:+.1f}%, DD={best_config[11]*100:.0f}%")

    return all_results


# ── Command-line interface ────────────────────────────────

def show_results(start, label):
    r = backtest(start, 2024)
    if r is None:
        return
    sh, tr, dd = r
    print(f"\n{'=' * 60}")
    print(f"{label} ({start}-2024, monthly rebalance)")
    print(f"{'=' * 60}")
    print(f"Sharpe: {sh:.4f}")
    print(f"Return:{tr * 100:+8.1f}%")
    print(f"MaxDD:{dd * 100:.0f}%")

    spy_p = load_price('SPY')
    if spy_p is not None:
        spy_m = spy_p.resample('ME').last()
        spy_r = spy_m.pct_change()
        spy_r = spy_r[spy_r.index.year >= start]
        if len(spy_r) > 0:
            spy_sh = np.mean(spy_r) / np.std(spy_r) * np.sqrt(12)
            spy_tr = (1 + spy_r).prod() - 1
            spy_cum = (1 + spy_r).cumprod()
            spy_dd = (spy_cum / spy_cum.cummax() - 1).min()
            print(f"\n  vs SPY: Sharpe={spy_sh:.4f}, Ret={spy_tr*100:+8.1f}%, DD={spy_dd*100:.0f}%")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 strategy_all_8.py [history|current|critique|optimize]')
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == 'history':
        show_results(2000, '25yr')
        show_results(1965, '60yr')
        show_results(2011, '14yr')
    elif cmd == 'current':
        print("Current allocation not yet implemented")
    elif cmd == 'critique':
        print("Critique not yet implemented")
    elif cmd == 'optimize':
        optimize_risk_power()

    elif cmd == 'quick_test':
        # Quick comparative test
        print("\nst7 vs st8 comparison (2011-2024):")
        for p, e, lab in [(0, 0.5, "st7 (no risk gate)"),
                          (3.0, 0.5, "st8 (risk_pow=3, EWY=0.5)"),
                          (3.0, 1.0, "st8 (risk_pow=3, EWY=1.0)")]:
            r = backtest(2011, 2024, intl_risk_power=p, ewy_share=e, verbose=False)
            if r:
                print(f"  {lab:35s}: Sharpe {r[0]:.4f} | Ret {r[1]*100:+8.1f}% | DD {r[2]*100:.0f}%")
