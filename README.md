# Daily Portfolio Management System

Automated daily portfolio management using the **st8 strategy** — an 8-asset macro-driven allocation system with risk-gated international rotation.

## Quick Start

```bash
# Run daily update (fetches fresh prices, computes trades, generates report)
python3 daily_operation.py --run

# Regenerate report without trading
python3 daily_operation.py --report-only

# Add cash injection
python3 daily_operation.py --inject 50000
```

## Strategy (st8)

8-asset allocation: **SPY, TLT, GLD, SHY, DBC, VXUS, EWY**

- **Continuous allocation** based on 6 macro indicators (CPI YoY, CAPE, UNRATE, yield inversion, real FEDFUNDS, INDPRO growth)
- **Risk-gated international rotation** — VXUS/EWY allocation scaled by macro risk score
- **All-time-high dip buying** — boosts equity weights when below ATH
- **SMA filter** — reduces allocation when price drops below 200-day SMA
- **Daily budget constraint** — max $500/day in trades (configurable via `config.json`)

## Configuration

All settings in `config.json`:

```json
{
  "budget": {"max_daily_exchange": 500, "currency": "USD"},
  "report": {"price_freshness_days": 7, "indicator_freshness_days": 60},
  "initial_portfolio": {"cash": 100000}
}
```

Planned cash injections in `injections.csv`:
```
date,amount
2026-04-01,100000
```

## GitHub Actions

Runs Mon-Fri at 16:00 UTC. See `github.md` for setup instructions.

## Data

Price data and macro indicators are in `data_cache/` (committed to repo for reproducibility). Daily prices are auto-updated from yfinance on each run.

## Files

| File | Purpose |
|------|---------|
| `strategy_all_8.py` | st8 strategy with risk-gated international rotation |
| `portfolio.py` | Portfolio rebalancing engine with daily exchange limits |
| `daily_operation.py` | CLI: `--init`, `--run`, `--inject`, `--report-only` |
| `config.json` | All tunable parameters |
| `injections.csv` | Planned cash injections |
| `daily_portfolio.csv` | Portfolio state (updated each run) |
| `md_to_pdf.py` | Report to PDF converter |
| `send_telegram.py` | Telegram report sender |
