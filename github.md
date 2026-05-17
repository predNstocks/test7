# GitHub Actions Daily Portfolio System

## Architecture

This repo runs a daily portfolio management system via GitHub Actions. The system:
1. Downloads fresh market data (yfinance) and macro indicators (FRED)
2. Computes st8 strategy target weights
3. Executes diff-based trades within the daily budget
4. Commits the updated portfolio state back to the repo
5. Generates a markdown report, converts to PDF, and sends via Telegram

## Stateless Design

GitHub Actions runners are stateless. Portfolio state persistence is achieved by:
- **daily_portfolio.csv** is committed to the repo and updated each run
- **data_cache/** is committed (7 asset prices + 5 FRED indicators + Shiller CAPE)
- Daily prices are auto-updated from yfinance on each run; CSVs are appended in-place
- **config.json** and **injections.csv** are committed for configuration

### State Flow
```
[Checkout repo] → [daily_portfolio.csv has yesterday's state]
                → [data_cache/ has baseline prices + indicators]
       ↓
[Run daily_operation.py --run] → [yfinance updates prices in data_cache/]
                               → [Computes trades, appends CSV]
       ↓
[Git diff check] → [If CSV changed, commit + push]
       ↓
[Generate report → PDF → Telegram]
```

## Files Committed to Repo

### Core Strategy
- `strategy_all_8.py` — st8 strategy with risk-gated international rotation
- `portfolio.py` — Portfolio rebalancing engine with daily exchange limits

### Daily Operations
- `daily_operation.py` — CLI with `--init`, `--run`, `--inject`, `--report-only`
- `config.json` — All tunable parameters (budget, freshness thresholds, etc.)
- `injections.csv` — Planned cash injections (date, amount)
- `daily_portfolio.csv` — **State file** — updated every run

### Data Cache (committed)
- `data_cache/YF_*.csv` — Daily prices for 7 assets (SPY, TLT, GLD, SHY, DBC, VXUS, EWY)
- `data_cache/FRED_*.csv` — Monthly macro indicators (CPI, UNRATE, T10Y2Y, FEDFUNDS, INDPRO)
- `data_cache/SHILLER_CAPE.csv` — Shiller PE10 from 1871
- All data is public and auto-updated from yfinance on each CI run

### GitHub Actions
- `.github/workflows/daily_portfolio.yml` — Scheduled workflow (Mon-Fri 16:00 UTC)
- `md_to_pdf.py` — Converts markdown report to PDF
- `send_telegram.py` — Sends report + PDF to Telegram
- `requirements.txt` — Python dependencies

## Files NOT Committed (Local Only)

Research scripts, tests, old strategies, reports, plots, data cache, and any files with local paths. See `.gitignore` for full list.

## GitHub Secrets Required

Set these in **Settings → Secrets and variables → Actions**:

| Secret | Description | How to Get |
|--------|-------------|------------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot API token | @BotFather on Telegram |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID | Use @userinfobot or check bot updates |

## GitHub Variables (Optional)

Set in **Settings → Secrets and variables → Variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_DAILY_EXCHANGE` | 500 | Daily trading budget in USD |

## Schedule

Runs **Monday-Friday at 16:00 UTC** (9am PT / 12pm ET).
- Weekends are skipped by the cron schedule
- Can be triggered manually via **Actions → Run workflow**

## Idempotency & Safety

- **Weekend/holiday runs**: Trades table shows `$0` — no CSV changes
- **Multiple runs per day**: CSV only written once per date (checked before append)
- **Missing data**: Falls back to last known prices; warns in report
- **No secrets in code**: All credentials via GitHub Secrets / environment variables

## Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| yfinance | SPY, TLT, GLD, SHY, DBC, VXUS, EWY daily prices | Daily |
| FRED | CPI, UNRATE, T10Y2Y, FEDFUNDS, INDPRO | Monthly |
| multpl.com | Shiller CAPE (PE10) | Monthly |

## Strategy (st8)

8-asset allocation with:
- **Continuous allocation** based on 6 macro indicators (CPI YoY, CAPE, UNRATE, yield inversion, real FEDFUNDS, INDPRO growth)
- **Risk-gated international rotation** — VXUS/EWY allocation scaled by macro risk score
- **All-time-high dip buying** — boosts equity weights when below ATH
- **SMA filter** — reduces allocation when price drops below 200-day SMA
- **Daily budget constraint** — max $500/day in trades (configurable)

## Local Development

```bash
# Run with local data
python3 daily_operation.py --run --budget 500

# Regenerate report only
python3 daily_operation.py --report-only

# Add cash injection
python3 daily_operation.py --inject 50000

# Initialize new portfolio
python3 daily_operation.py --init --cash 100000
```

## Troubleshooting

### "No portfolio CSV" error
Run `python3 daily_operation.py --init --cash 100000` first.

### Data freshness warnings
Run `python3 daily_operation.py --run` to auto-update prices from yfinance.

### Telegram not sending
Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set in GitHub Secrets.

### PDF generation fails
WeasyPrint requires system dependencies. In GitHub Actions, these are pre-installed.
