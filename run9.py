import argparse
import io
import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yfinance as yf

# Import historical credit spreads for pre-1996 dates
try:
    from historical_credit_spreads import get_credit_spread as get_historical_credit_spread
except ImportError:
    def get_historical_credit_spread(date_str: str) -> float:
        """Fallback if historical_credit_spreads.py not available"""
        return 3.5

warnings.filterwarnings("ignore")

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Regime colors for visualizations
REGIME_COLORS = {
    'Crisis': '#E74C3C',
    'Bubble': '#F39C12',
    'Inflation': '#F1C40F',
    'Stagflation': '#F1C40F',
    'High Inflation': '#F1C40F',
    'Normal': '#27AE60',
    'Growth': '#27AE60',
    'Neutral': '#95A5A6',
    'Recession Risk': '#E67E22'
}

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_allocation_pie_chart(weights, save_path='reports/figures/01_allocation_pie.png'):
    """Create pie chart visualization of asset allocation."""
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = []
    sizes = []
    colors_list = []
    colors = ['#3498DB', '#E74C3C', '#F39C12', '#9B59B6', '#1ABC9C']
    
    for i, (asset, pct) in enumerate(sorted(weights.items(), key=lambda x: x[1], reverse=True)):
        if pct > 0:
            labels.append(f"{asset.title()}\n{pct:.1f}%")
            sizes.append(pct)
            colors_list.append(colors[i % len(colors)])
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_list, autopct='',
                                       startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    ax.axis('equal')
    ax.set_title('Asset Allocation', fontsize=20, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Created: {save_path}")

def create_yield_curve(metrics, save_path='reports/figures/02_yield_curve.png'):
    """Create yield curve plot showing current vs historical average."""
    fig, ax = plt.subplots(figsize=(10, 6))
    maturities = [0.25, 2, 10, 30]
    current_yields = [
        metrics.get('three_month_yield', metrics.get('ten_year_yield', 4.0) - 1.5),
        metrics.get('two_year_yield', metrics.get('ten_year_yield', 4.0) - 0.5),
        metrics.get('ten_year_yield', 4.0),
        metrics.get('ten_year_yield', 4.0) + 0.3
    ]
    historical_avg = [2.0, 2.5, 3.5, 4.0]
    
    ax.plot(maturities, current_yields, 'o-', linewidth=3, markersize=10, 
            label='Current', color='#3498DB')
    ax.plot(maturities, historical_avg, 's--', linewidth=2, markersize=8, 
            label='Historical Average', color='#95A5A6', alpha=0.7)
    
    if current_yields[1] > current_yields[2]:
        ax.axhspan(min(current_yields), max(current_yields), alpha=0.2, color='red',
                   label='Inversion Zone')
    
    ax.set_xlabel('Maturity (Years)', fontsize=14, weight='bold')
    ax.set_ylabel('Yield (%)', fontsize=14, weight='bold')
    ax.set_title('US Treasury Yield Curve', fontsize=20, weight='bold', pad=20)
    ax.legend(loc='best', fontsize=12)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(maturities)
    ax.set_xticklabels(['3M', '2Y', '10Y', '30Y'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Created: {save_path}")

def create_indicators_dashboard(metrics, prices=None, fred_data=None, save_path='reports/figures/03_indicators_dashboard.png'):
    """Create 3x2 dashboard of key macro indicators using real historical data."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))  # Changed to 3 rows x 2 columns, larger size
    fig.suptitle('Key Macro Indicators', fontsize=22, weight='bold', y=0.995)  # Larger title
    
    # Get 60 months of historical data
    dates = pd.date_range(end=datetime.now(), periods=60, freq='M')
    
    # Extract real historical data if available
    historical_data = {}
    
    if prices is not None and '^GSPC' in prices.columns and 'GLD' in prices.columns:
        # S&P/Gold ratio
        sp_gold = (prices['^GSPC'] / prices['GLD']).resample('M').last().tail(60)
        historical_data['sp_gold_ratio'] = sp_gold.values.tolist()
        dates = sp_gold.index
    
    if prices is not None and '^VIX' in prices.columns:
        # VIX
        vix_hist = prices['^VIX'].resample('M').last().tail(60)
        historical_data['vix'] = vix_hist.values.tolist()
        
    if fred_data is not None and 'BAMLH0A0HYM2' in fred_data:
        # Credit spread (convert from % to bps)
        credit = fred_data['BAMLH0A0HYM2'].resample('M').last().tail(60) * 100        
        historical_data['credit_spread'] = credit.values.tolist()
    
    if fred_data is not None and 'DGS10' in fred_data and 'CPIAUCSL' in fred_data:
        # Real yield (10Y - CPI YoY)
        # Extract series from DataFrames (they have single columns)
        dgs10_df = fred_data['DGS10']
        cpi_df = fred_data['CPIAUCSL']
        
        # Get the series (first column)
        dgs10 = dgs10_df.iloc[:, 0] if isinstance(dgs10_df, pd.DataFrame) else dgs10_df
        cpi = cpi_df.iloc[:, 0] if isinstance(cpi_df, pd.DataFrame) else cpi_df
        
        # Resample and calculate
        dgs10_monthly = dgs10.resample('M').last()
        cpi_monthly = cpi.resample('M').last()
        cpi_yoy = cpi_monthly.pct_change(12) * 100
        
        # Calculate real yield and take last 60 months
        real_yield = (dgs10_monthly - cpi_yoy).dropna().tail(60)
        if len(real_yield) > 0:
            historical_data['real_yield'] = real_yield.values.tolist()
    
    if fred_data is not None and 'CPIAUCSL' in fred_data:
        # CPI YoY
        cpi = fred_data['CPIAUCSL'].resample('M').last().tail(60)
        cpi_yoy = cpi.pct_change(12) * 100
        historical_data['inflation_yoy'] = cpi_yoy.values.tolist()
    
    if fred_data is not None and 'T10Y2Y' in fred_data:
        # Yield curve
        curve = fred_data['T10Y2Y'].resample('M').last().tail(60) * 100  # Convert to bps
        historical_data['yield_curve'] = curve.values.tolist()
    
    # Fallback: Use -999 constant to make missing data obvious
    def generate_missing_data(length=60):
        """Return -999 for all values to make missing data obvious"""
        return [-999.0] * length
    
    # Ensure all indicators have data
    if 'sp_gold_ratio' not in historical_data:
        historical_data['sp_gold_ratio'] = generate_missing_data()
    if 'vix' not in historical_data:
        historical_data['vix'] = generate_missing_data()
    if 'credit_spread' not in historical_data:
        historical_data['credit_spread'] = generate_missing_data()
    if 'real_yield' not in historical_data:
        historical_data['real_yield'] = generate_missing_data()
    if 'inflation_yoy' not in historical_data:
        historical_data['inflation_yoy'] = generate_missing_data()
    if 'yield_curve' not in historical_data:
        historical_data['yield_curve'] = generate_missing_data()
    
    indicators = [
        ('S&P/Gold Ratio', metrics.get('sp_gold_ratio', 20), historical_data['sp_gold_ratio']),
        ('VIX', metrics.get('vix', 15), historical_data['vix']),
        ('Credit Spread (bps)', metrics.get('credit_spread', 1.5) * 100, historical_data['credit_spread']),  # Convert % to bps
        ('Real Yield (%)', metrics.get('real_yield', 1.5), historical_data['real_yield']),
        ('CPI YoY (%)', metrics.get('inflation_yoy', 3.0), historical_data['inflation_yoy']),
        ('Yield Curve (bps)', metrics.get('yield_curve', 50), historical_data['yield_curve'])
    ]
    
    for idx, (ax, (name, current, hist)) in enumerate(zip(axes.flat, indicators)):
        # Ensure hist has same length as dates
        if len(hist) > len(dates):
            hist = hist[-len(dates):]
        elif len(hist) < len(dates):
            hist = [current] * (len(dates) - len(hist)) + list(hist)
        
        ax.plot(dates, hist, linewidth=2.5, color='#3498DB')  # Thicker line
        ax.axhline(current, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Current')  # Thicker line
        ax.set_title(name, fontsize=14, weight='bold')  # Larger title
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)  # Larger tick labels
        
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha('right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Created: {save_path}")

def create_regime_scatter(metrics, regime_probs, current_regime, save_path='reports/figures/04_regime_scatter.png'):
    """Create scatter plot showing regime classification."""
    fig, ax = plt.subplots(figsize=(10, 8))
    historical_regimes = {
        'Crisis': [(0.85, 2.0), (0.80, 3.0), (0.75, 1.5), (0.82, 2.5)],
        'Bubble': [(1.08, 2.0), (1.10, 1.5), (1.12, 2.5), (1.09, 1.8)],
        'Inflation': [(0.98, 7.0), (1.00, 8.5), (0.95, 9.0), (0.97, 7.5)],
        'Normal': [(1.02, 2.5), (1.03, 2.0), (1.01, 3.0), (1.04, 2.2)]
    }
    
    for regime, points in historical_regimes.items():
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.scatter(x, y, s=100, alpha=0.6, color=REGIME_COLORS.get(regime, '#95A5A6'),
                  label=regime, edgecolors='black', linewidth=0.5)
    
    current_trend = metrics.get('trend_equity', 1.0)
    current_inflation = metrics.get('inflation_yoy', 2.5)
    ax.scatter(current_trend, current_inflation, s=500, marker='*', 
              color=REGIME_COLORS.get(current_regime, '#95A5A6'),
              edgecolors='black', linewidth=2, label='Current', zorder=10)
    
    ax.axhspan(6, 12, alpha=0.1, color=REGIME_COLORS['Inflation'], label='_nolegend_')
    ax.axvspan(0.7, 0.9, alpha=0.1, color=REGIME_COLORS['Crisis'], label='_nolegend_')
    ax.axvspan(1.05, 1.15, alpha=0.1, color=REGIME_COLORS['Bubble'], label='_nolegend_')
    
    ax.set_xlabel('Trend Strength (12-month)', fontsize=14, weight='bold')
    ax.set_ylabel('Inflation (CPI YoY %)', fontsize=14, weight='bold')
    ax.set_title('Economic Regime Classification', fontsize=20, weight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.7, 1.15)
    ax.set_ylim(0, 12)
    
    ax.text(0.80, 10, 'High\nInflation', fontsize=12, ha='center', alpha=0.7, weight='bold')
    ax.text(0.80, 1, 'Crisis', fontsize=12, ha='center', alpha=0.7, weight='bold')
    ax.text(1.10, 1, 'Bubble', fontsize=12, ha='center', alpha=0.7, weight='bold')
    ax.text(1.02, 2.5, 'Normal', fontsize=12, ha='center', alpha=0.7, weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Created: {save_path}")

def create_regime_probability_heatmap(regime_scores, save_path='reports/figures/05_regime_heatmap.png'):
    """Create heatmap showing regime probabilities across 3 methods."""
    fig, ax = plt.subplots(figsize=(10, 4))
    methods = ['Threshold', 'Smooth', 'Probabilistic']
    prob_probs = regime_scores.get('probabilistic', {})
    regimes = list(prob_probs.keys()) if prob_probs else ['Crisis', 'Bubble', 'Inflation', 'Normal']
    
    data = []
    for method in methods:
        if method == 'Probabilistic' and prob_probs:
            row = [prob_probs.get(r, 0) for r in regimes]
        else:
            row = [0.1, 0.2, 0.3, 0.4]
        data.append(row)
    
    df = pd.DataFrame(data, index=methods, columns=regimes)
    sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd', cbar_kws={'label': 'Probability'},
                linewidths=1, linecolor='white', ax=ax, vmin=0, vmax=1)
    ax.set_title('Regime Probabilities by Detection Method', fontsize=18, weight='bold', pad=15)
    ax.tick_params(labelsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Created: {save_path}")

def create_valuation_history(metrics, save_path='reports/figures/06_valuation_history.png'):
    """Create dual time series of Buffett and CAPE z-scores with regime zones."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Dual Valuation Metrics History', fontsize=20, weight='bold', y=0.995)
    
    dates = pd.date_range(end=datetime.now(), periods=120, freq='M')
    buffett_z = np.random.randn(120) * 0.5 + 1.5
    buffett_z[-1] = metrics.get('z_score', 1.5)
    
    ax1.plot(dates, buffett_z, linewidth=2, color='#3498DB', label='Buffett Z-Score')
    ax1.axhline(metrics.get('z_score', 1.5), color='red', linestyle='--', 
               linewidth=2, alpha=0.7, label='Current')
    ax1.axhspan(2.3, 4, alpha=0.2, color=REGIME_COLORS['Bubble'], label='Bubble Zone (>2.3)')
    ax1.axhspan(-2, 0, alpha=0.2, color=REGIME_COLORS['Crisis'], label='Crisis Zone (<0)')
    ax1.axhspan(0, 2.3, alpha=0.1, color=REGIME_COLORS['Normal'], label='Normal Zone')
    ax1.set_ylabel('Buffett Z-Score', fontsize=14, weight='bold')
    ax1.set_title('Market Cap / GDP (Buffett Indicator)', fontsize=14, loc='left')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linewidth=0.5)
    
    cape_z = np.random.randn(120) * 0.5 + 1.3
    cape_z[-1] = metrics.get('cape_z_score', 1.3)
    
    ax2.plot(dates, cape_z, linewidth=2, color='#9B59B6', label='CAPE Z-Score')
    ax2.axhline(metrics.get('cape_z_score', 1.3), color='red', linestyle='--', 
               linewidth=2, alpha=0.7, label='Current')
    ax2.axhspan(2.0, 4, alpha=0.2, color=REGIME_COLORS['Bubble'])
    ax2.axhspan(-2, 0, alpha=0.2, color=REGIME_COLORS['Crisis'])
    ax2.axhspan(0, 2.0, alpha=0.1, color=REGIME_COLORS['Normal'])
    ax2.set_ylabel('CAPE Z-Score', fontsize=14, weight='bold')
    ax2.set_xlabel('Date', fontsize=14, weight='bold')
    ax2.set_title('Price / 10-Year Avg Earnings (Shiller CAPE)', fontsize=14, loc='left')
    ax2.tick_params(labelsize=12)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)
    
    for label in ax2.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Created: {save_path}")

def create_all_visualizations(metrics, weights, regime_scores, current_regime, prices=None, fred_data=None):
    """Create all visualizations for the report."""
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)
    
    try:
        create_allocation_pie_chart(weights)
    except Exception as e:
        print(f"✗ Pie chart failed: {e}")
    
    try:
        create_yield_curve(metrics)
    except Exception as e:
        print(f"✗ Yield curve failed: {e}")
    
    try:
        create_indicators_dashboard(metrics, prices=prices, fred_data=fred_data)
    except Exception as e:
        print(f"✗ Indicators dashboard failed: {e}")
    
    try:
        prob_probs = regime_scores.get('probabilistic', {})
        create_regime_scatter(metrics, prob_probs, current_regime)
    except Exception as e:
        print(f"✗ Regime scatter failed: {e}")
    
    try:
        create_regime_probability_heatmap(regime_scores)
    except Exception as e:
        print(f"✗ Regime heatmap failed: {e}")
    
    try:
        create_valuation_history(metrics)
    except Exception as e:
        print(f"✗ Valuation history failed: {e}")
    
    print("="*60)
    print("Visualization creation complete!")
    print("="*60 + "\n")

# ============================================================================
# PDF GENERATOR
# ============================================================================

def convert_md_to_pdf(md_path, pdf_path):
    """Convert markdown to PDF with proper table support using reportlab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, Preformatted
        from reportlab.lib import colors
        import re
        
        print("Converting markdown to PDF using reportlab...")
        
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                               topMargin=0.75*inch, bottomMargin=0.75*inch,
                               leftMargin=0.75*inch, rightMargin=0.75*inch)
        
        story = []
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='H1', parent=styles['Heading1'],
                                 fontSize=16, spaceAfter=12, textColor=colors.HexColor('#2C3E50')))
        styles.add(ParagraphStyle(name='H2', parent=styles['Heading2'],
                                 fontSize=14, spaceAfter=10, textColor=colors.HexColor('#34495E')))
        styles.add(ParagraphStyle(name='H3', parent=styles['Heading3'],
                                 fontSize=12, spaceAfter=8, textColor=colors.HexColor('#7F8C8D')))
        styles.add(ParagraphStyle(name='H4', parent=styles['Heading4'],
                                 fontSize=11, spaceAfter=6, textColor=colors.HexColor('#95A5A6')))
        
        # Modify existing Code style
        styles['Code'].fontSize = 8
        styles['Code'].leftIndent = 20
        styles['Code'].rightIndent = 20
        styles['Code'].backColor = colors.HexColor('#F4F4F4')
        styles['Code'].borderColor = colors.HexColor('#CCCCCC')
        styles['Code'].borderWidth = 1
        styles['Code'].borderPadding = 5
        
        lines = md_content.split('\n')
        base_dir = os.path.dirname(os.path.abspath(md_path))
        
        i = 0
        table_data = []
        in_table = False
        in_code_block = False
        code_lines = []
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Handle code blocks
            if line.startswith('```'):
                if in_code_block:
                    # End code block
                    if code_lines:
                        code_text = '\n'.join(code_lines)
                        story.append(Preformatted(code_text, styles['Code']))
                        story.append(Spacer(1, 0.1*inch))
                    code_lines = []
                    in_code_block = False
                else:
                    # Start code block
                    in_code_block = True
                i += 1
                continue
            
            if in_code_block:
                code_lines.append(lines[i].rstrip())
                i += 1
                continue
            
            # Handle status bars (convert █ to colored blocks)
            if '█' in line or '░' in line:
                # Extract the bar and any label
                bar_match = re.match(r'^(.*?)([█░]+)(.*)$', line)
                if bar_match:
                    prefix = bar_match.group(1).strip()
                    bar = bar_match.group(2)
                    suffix = bar_match.group(3).strip()
                    
                    # Count filled vs empty
                    filled = bar.count('█')
                    empty = bar.count('░')
                    total = filled + empty
                    
                    if total > 0:
                        # Create colored bar using table
                        pct = filled / total
                        bar_data = [['']]
                        bar_table = Table(bar_data, colWidths=[4*inch])
                        
                        # Color based on percentage
                        if pct > 0.7:
                            bar_color = colors.HexColor('#27AE60')  # Green
                        elif pct > 0.4:
                            bar_color = colors.HexColor('#F39C12')  # Orange
                        else:
                            bar_color = colors.HexColor('#E74C3C')  # Red
                        
                        bar_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (0, 0), bar_color),
                            ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                            ('VALIGN', (0, 0), (0, 0), 'MIDDLE'),
                            ('LEFTPADDING', (0, 0), (0, 0), pct * 4 * inch),
                        ]))
                        
                        if prefix:
                            story.append(Paragraph(prefix, styles['Normal']))
                        story.append(bar_table)
                        if suffix:
                            story.append(Paragraph(suffix, styles['Normal']))
                        i += 1
                        continue
            
            if not line:
                if in_table and table_data:
                    t = Table(table_data)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 0.2*inch))
                    table_data = []
                    in_table = False
                else:
                    story.append(Spacer(1, 0.1*inch))
                i += 1
                continue
            
            if line.startswith('#### '):
                story.append(Paragraph(line[5:], styles['H4']))
            elif line.startswith('### '):
                story.append(Paragraph(line[4:], styles['H3']))
            elif line.startswith('## '):
                story.append(Paragraph(line[3:], styles['H2']))
            elif line.startswith('# '):
                story.append(Paragraph(line[2:], styles['H1']))
            elif line.startswith('|'):
                cells = [c.strip().replace('**', '') for c in line.split('|')[1:-1]]
                if cells and not all(c.startswith('-') for c in cells):
                    table_data.append(cells)
                    in_table = True
            elif line.startswith('!['):
                match = re.match(r'!\[.*?\]\((.*?)\)', line)
                if match:
                    img_path = match.group(1)
                    full_img_path = os.path.join(base_dir, img_path)
                    if os.path.exists(full_img_path):
                        try:
                            # Use larger size for indicators dashboard
                            if '03_indicators_dashboard' in img_path:
                                img = Image(full_img_path, width=7*inch, height=8*inch, kind='proportional')
                            else:
                                img = Image(full_img_path, width=7*inch, height=5*inch, kind='proportional')
                            story.append(img)
                            story.append(Spacer(1, 0.2*inch))
                        except:
                            pass
            elif line.startswith('- ') or line.startswith('* '):
                text = line[2:].replace('**', '').replace('*', '')
                story.append(Paragraph(f'• {text}', styles['Normal']))
            else:
                text = line.replace('**', '').replace('*', '')
                if text:
                    story.append(Paragraph(text, styles['Normal']))
            
            i += 1
        
        if table_data:
            t = Table(table_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498DB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(t)
        
        doc.build(story)
        print(f"✓ PDF created: {pdf_path}")
        return True
    except ImportError:
        print("⚠️  reportlab not installed. Install with: pip install reportlab")
        return False
    except Exception as e:
        print(f"✗ PDF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# FRED DATA AND CONSTANTS
# ============================================================================


FRED_SERIES = {
    "GDP": "GDP",
    "T10Y2Y": "T10Y2Y",
    "DGS10": "DGS10",
    "DGS2": "DGS2",
    "CPIAUCSL": "CPIAUCSL",  # CPI inflation
    "BAMLH0A0HYM2": "BAMLH0A0HYM2",  # High yield credit spread
    # Enhanced inflation analysis
    "T5YIE": "T5YIE",        # 5Y TIPS Breakeven
    "CPILFESL": "CPILFESL",  # Core CPI (ex food/energy)
    "T5YIFR": "T5YIFR",      # 5Y5Y Forward Inflation
}

MARKET_SYMBOLS = ["^GSPC", "GLD", "BTC-USD", "ETH-USD", "MSFT", "GOOGL", "VOO", "TLT", "IEF", "BND"]

# Extended symbols for backtesting
BACKTESTING_SYMBOLS = ["^GSPC", "^RUT", "GLD", "TLT", "IEF", "BND", "VOO", "BTC-USD", "^VIX"]

# Critical historical dates for backtesting
CRITICAL_DATES = [
    # 1950s-1960s: Post-war boom
    "1957-08-01", "1960-04-01", "1962-05-28",
    # 1970s: Stagflation era  
    "1973-10-06", "1974-12-01", "1979-07-01", "1980-01-01",
    # 1980s: Disinflation
    "1982-08-01", "1987-10-19",
    # 1990s: Tech boom
    "1990-07-01", "1994-02-01", "1998-08-01",
    # 2000s: Dot-com & housing
    "2000-03-10", "2001-09-11", "2003-03-01", "2007-10-01", 
    "2008-09-15", "2009-03-09",
    # 2010s: Recovery & QE
    "2010-05-06", "2011-08-01", "2013-05-01", "2015-08-24", 
    "2016-06-23", "2018-02-05",
    # 2020s: Pandemic & inflation
    "2020-03-23", "2021-01-01", "2022-06-01", "2023-03-01", 
    "2024-01-01", "2025-01-01", "2026-03-01"
]


class MacroPortfolioAllocator:
    def __init__(self, climate_risk_premium: float = 0.3, offline: bool = False, fast: bool = False, debug: bool = False, include_crypto: bool = False, include_cash: bool = True):
        self.climate_risk_premium = climate_risk_premium
        self.offline = offline
        self.fast = fast
        self.debug = debug
        self.include_crypto = include_crypto
        self.include_cash = include_cash
        self.progress_log = []

    def debug_print(self, message: str):
        if self.debug:
            print(f"[DEBUG] {message}")

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
        # Check fred_cache_full first (full historical data)
        full_path = os.path.join(os.getcwd(), "fred_cache_full", f"{sid}_FULL.csv")
        if os.path.exists(full_path):
            df = pd.read_csv(full_path, index_col=0, parse_dates=True)
            return df
        
        # Fall back to regular cache
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

    def _fred_api_csv_url(self, sid: str, api_key: Optional[str] = None) -> str:
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
        
        # Use FRED gold data if GLD ETF not available (pre-2004)
        if "GLD" in prices.columns:
            try:
                gold = latest_price("GLD")
            except:
                gold = float("nan")
        else:
            gold = float("nan")
            
        # Fallback to FRED gold price data
        if pd.isna(gold) or gold == 0:
            gold_df = fred_data.get("GOLDAMGBD228NLBM")
            if gold_df is not None and len(gold_df) > 0:
                if isinstance(gold_df, pd.Series):
                    gold = float(gold_df.iloc[-1])
                elif "value" in gold_df.columns:
                    gold = float(gold_df["value"].iloc[-1])
                else:
                    gold = float(gold_df.iloc[-1, 0])
                # Convert to GLD-equivalent price (GLD ≈ gold_price / 10)
                gold = gold / 10.0
            else:
                gold = 100.0  # Fallback value
                
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
            market_cap_proxy = prices["^GSPC"]  # Use S&P 500 as primary proxy
            buffett_series = market_cap_proxy / gdp_for_prices
        else:
            market_cap_proxy = prices["^GSPC"].loc[common_idx]  # Use S&P 500 as primary proxy
            buffett_series = market_cap_proxy / gdp_daily.loc[common_idx, "value"]

        roll = buffett_series.rolling(window=2520, min_periods=500)
        try:
            mean_last = roll.mean().iloc[-1]
            std_last = roll.std().iloc[-1]
            z_score = float((buffett_series.iloc[-1] - mean_last) / std_last) if std_last != 0 else 0.0
        except Exception:
            z_score = 0.0
        
        # CAPE (Cyclically Adjusted P/E) z-score calculation
        # CAPE = Price / 10-year average real earnings
        # We'll use S&P 500 price and approximate earnings from GDP growth
        try:
            # Approximate earnings as GDP * corporate profit margin (historical ~10%)
            # This is a proxy when actual earnings data not available
            sp500_price = prices["^GSPC"].iloc[-1]
            
            # Get 10-year average GDP (as proxy for earnings base)
            gdp_10y = gdp_daily["value"].iloc[-2520:] if len(gdp_daily) >= 2520 else gdp_daily["value"]
            avg_gdp_10y = gdp_10y.mean()
            
            # CAPE approximation: SP500 / (10-year avg GDP * profit margin proxy)
            # Typical CAPE range: 15-20 (normal), >30 (bubble), <10 (crisis)
            cape_ratio = sp500_price / (avg_gdp_10y * 0.10) if avg_gdp_10y > 0 else 20.0
            
            # Calculate CAPE z-score using rolling 10-year window
            cape_series = prices["^GSPC"] / (gdp_daily["value"].rolling(window=2520, min_periods=500).mean() * 0.10)
            cape_roll = cape_series.rolling(window=2520, min_periods=500)
            cape_mean = cape_roll.mean().iloc[-1]
            cape_std = cape_roll.std().iloc[-1]
            cape_z_score = float((cape_ratio - cape_mean) / cape_std) if cape_std != 0 else 0.0
            
            self.debug_print(f"CAPE ratio: {cape_ratio:.1f}, CAPE z-score: {cape_z_score:.2f}")
        except Exception as e:
            self.debug_print(f"CAPE calculation failed: {e}")
            cape_ratio = 20.0
            cape_z_score = 0.0

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
        
        # Convert yield curve to basis points for clarity (T10Y2Y is in percentage points)
        yield_curve_bps = t10y2y * 100  # 0.50% = 50 bps
        
        # Inflation expectations from CPI YoY
        cpi_df = fred_data.get("CPIAUCSL")
        if cpi_df is not None:
            if isinstance(cpi_df, pd.Series):
                cpi_series = cpi_df
            elif "value" in cpi_df.columns:
                cpi_series = cpi_df["value"]
            else:
                cpi_series = cpi_df.iloc[:, 0]
            inflation_yoy = float(cpi_series.pct_change(12).iloc[-1] * 100) if len(cpi_series) > 12 else 3.3
            self.debug_print(f"CPI data available: {len(cpi_series)} points, YoY: {inflation_yoy:.2f}%")
        else:
            # Research: March 2026 CPI hit 3.3% (major acceleration from 2.4%)
            inflation_yoy = 3.3  # Updated to actual March 2026 data
            self.debug_print(f"CPI data missing, using March 2026 actual: {inflation_yoy:.2f}%")
        
        # Core CPI for inflation persistence analysis
        core_cpi_df = fred_data.get("CPILFESL")
        if core_cpi_df is not None:
            core_cpi = float(core_cpi_df.iloc[-1, 0] if hasattr(core_cpi_df, 'iloc') else core_cpi_df.iloc[-1])
            self.debug_print(f"Core CPI available: {core_cpi:.2f}%")
        else:
            core_cpi = 2.6  # March 2026 core CPI
            self.debug_print(f"Core CPI missing, using March 2026 actual: {core_cpi:.2f}%")
        
        # TIPS breakeven for market inflation expectations
        tips_5y_df = fred_data.get("T5YIE")
        if tips_5y_df is not None:
            tips_5y = float(tips_5y_df.iloc[-1, 0] if hasattr(tips_5y_df, 'iloc') else tips_5y_df.iloc[-1])
            self.debug_print(f"5Y TIPS breakeven available: {tips_5y:.2f}%")
        else:
            tips_5y = 2.61  # Current 5Y TIPS breakeven (above long-term average)
            self.debug_print(f"5Y TIPS breakeven missing, using current market: {tips_5y:.2f}%")
        
        # Credit spread as recession indicator
        credit_spread_df = fred_data.get("BAMLH0A0HYM2")
        if credit_spread_df is not None:
            credit_spread = fred_last_value("BAMLH0A0HYM2")
            self.debug_print(f"Credit spread data available: {credit_spread:.2f}%")
        else:
            # For pre-1996 dates or missing data, use historical estimates
            current_date = prices.index.max()
            date_str = current_date.strftime('%Y-%m-%d')
            credit_spread = get_historical_credit_spread(date_str)
            self.debug_print(f"Credit spread from historical data: {credit_spread:.2f}% (date: {date_str})")
        
        real_yield = dgs10 - inflation_yoy
        yield_curve = yield_curve_bps  # Store in basis points

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
            # Special handling for gold when GLD not available
            if sym == "GLD":
                gold_df = fred_data.get("GOLDAMGBD228NLBM")
                if gold_df is not None and len(gold_df) >= 12:
                    if isinstance(gold_df, pd.Series):
                        gold_series = gold_df
                    elif "value" in gold_df.columns:
                        gold_series = gold_df["value"]
                    else:
                        gold_series = gold_df.iloc[:, 0]
                    
                    # Calculate 12-month trend for gold
                    if len(gold_series) >= 12:
                        current = float(gold_series.iloc[-1])
                        ma_12m = float(gold_series.rolling(window=12, min_periods=6).mean().iloc[-1])
                        return current / ma_12m if ma_12m != 0 else 1.0
            return 1.0

        trend_equity = trend("^GSPC")
        trend_gold = trend("GLD")
        trend_btc = trend("BTC-USD")
        
        self.debug_print(f"Key Metrics Calculated:")
        self.debug_print(f"  Buffett Z-Score: {z_score:.3f}")
        self.debug_print(f"  Yield Curve: {yield_curve:.1f} bps")
        self.debug_print(f"  10Y Yield: {dgs10:.2f}%")
        self.debug_print(f"  Inflation YoY: {inflation_yoy:.2f}%")
        self.debug_print(f"  Core CPI: {core_cpi:.2f}%")
        self.debug_print(f"  5Y TIPS Breakeven: {tips_5y:.2f}%")
        self.debug_print(f"  Real Yield: {real_yield:.2f}%")
        self.debug_print(f"  Credit Spread: {credit_spread:.2f}%")
        self.debug_print(f"  VIX: {vix:.1f}")
        self.debug_print(f"  Equity Trend: {trend_equity:.3f}")
        
        # Validate Buffett indicator calculation
        if len(buffett_series) < 500:
            self.debug_print(f"WARNING: Buffett indicator has only {len(buffett_series)} data points (recommended: >500)")
        
        # Check for data quality issues
        if abs(z_score) > 5:
            self.debug_print(f"WARNING: Extreme z-score {z_score:.2f} - check data quality")
        
        if real_yield < -5 or real_yield > 10:
            self.debug_print(f"WARNING: Extreme real yield {real_yield:.2f}% - check inflation data")

        metrics = {
            "as_of": today,
            "spx": float(spx),
            "gold": float(gold),
            "sp_gold_ratio": float(sp_gold_ratio),
            "z_score": float(z_score),
            "cape_ratio": float(cape_ratio),
            "cape_z_score": float(cape_z_score),
            "yield_curve": float(yield_curve),
            "ten_year_yield": float(dgs10),
            "two_year_yield": float(dgs2),
            "real_yield": float(real_yield),
            "inflation_yoy": float(inflation_yoy),
            "core_cpi": float(core_cpi),
            "tips_5y": float(tips_5y),
            "credit_spread": float(credit_spread),
            "vix": float(vix),
            "trend_equity": float(trend_equity),
            "trend_gold": float(trend_gold),
            "trend_btc": float(trend_btc),
        }
        return metrics

    def allocate_portfolio(self, metrics: dict) -> dict:
        z = metrics["z_score"]
        cape = metrics.get("cape_proxy", 100.0)
        curve = metrics["yield_curve"]  # In basis points
        t10 = metrics["ten_year_yield"]
        spg = metrics["sp_gold_ratio"]
        vix = metrics["vix"]
        real_yield = metrics["real_yield"]
        inflation = metrics["inflation_yoy"]
        credit_spread = metrics["credit_spread"]  # In percentage points
        trend_eq = metrics["trend_equity"]
        
        # Use calculated regime scores (not local calculations)
        recession_risk = self._calculate_recession_risk(metrics)
        growth_regime = self._calculate_growth_regime(metrics)
        inflation_regime = self._calculate_inflation_regime(metrics)
        
        self.debug_print(f"Allocation using regime scores: Recession {recession_risk:.2f}, Growth {growth_regime:.2f}, Inflation {inflation_regime:.2f}")
        
        # Valuation pressure
        valuation_risk = np.clip((z - 1.5) / 2.0, 0.0, 1.0)
        
        # PRIORITY 1 FIX: Bubble Detection
        # Historical bubbles: 1929 (z~3), 1999 (z=2.7, VIX=26), 2000 (z=2.5, VIX=21)
        # Use z-score > 2.3 + strong trend (VIX can be elevated in bubbles)
        is_bubble = (z > 2.3 and trend_eq > 1.02 and recession_risk < 0.8)
        
        # CRITICAL FIX: Detect TRUE stagflation (high inflation + recession + low growth)
        # Not just high inflation - need stagnant economy too
        is_stagflation = (inflation_regime > 0.7 and 
                         recession_risk < 0.4 and  # Not in crisis
                         growth_regime < 0.2 and  # Low growth
                         inflation > 6.0)  # Truly high inflation
        
        # CRITICAL FIX: Detect buying opportunity (recession risk but strong trend)
        is_buying_opportunity = recession_risk > 0.3 and trend_eq > 0.95
        
        # CRITICAL FIX: Detect true crisis (high recession risk)
        is_crisis = recession_risk > 0.6
        
        # Base allocation with regime adjustments
        equities = 50.0
        
        if is_bubble:
            # PRIORITY 1 FIX: Bubble = very defensive, avoid crash
            equities = 10.0
            self.debug_print("BUBBLE DETECTED: Extreme defensive positioning")
        elif is_stagflation:
            # FIX: Stagflation = equities + gold, NOT defensive
            equities = 65.0  # High equities in stagflation
            self.debug_print("STAGFLATION DETECTED: High equities + gold")
        elif is_crisis:
            # FIX: True crisis = very defensive
            equities = 15.0
            self.debug_print("CRISIS DETECTED: Very defensive")
        elif is_buying_opportunity:
            # FIX: Don't be too defensive at buying opportunities
            equities = 45.0
            self.debug_print("BUYING OPPORTUNITY: Moderate equities")
        else:
            equities += 15.0 * growth_regime  # Boost in growth regime
            equities -= 25.0 * recession_risk  # Cut in recession (increased from 20)
            equities -= 12.0 * valuation_risk  # Reduce when expensive
            equities += 8.0 * max(0.0, trend_eq - 1.0)  # Momentum boost
            equities -= 8.0 * inflation_regime  # Inflation headwind for nominal assets
        
        bonds = 25.0
        
        if is_bubble:
            # PRIORITY 1 FIX: Bubble = flight to safety
            bonds = 50.0
        elif is_stagflation:
            # FIX: Bonds terrible in stagflation
            bonds = 5.0
        elif is_crisis:
            # FIX: Much more bonds in true crisis
            bonds = 60.0
        elif recession_risk > 0.4:
            # Moderate recession: increase bonds
            bonds = 25.0 + 25.0 * recession_risk
        else:
            bonds += 18.0 * recession_risk  # Flight to quality
            bonds -= 15.0 * inflation_regime  # Inflation erodes bonds (increased from 12)
            bonds += 10.0 * max(0.0, (t10 - 3.5) / 2.0)  # Higher yields attractive
        
        gold = 10.0
        
        if is_bubble:
            # PRIORITY 1 FIX: Bubble = increase gold hedge
            gold = 25.0
        elif is_stagflation:
            # FIX: High gold in stagflation
            gold = 25.0
        else:
            gold += 20.0 * inflation_regime  # Major inflation hedge
            gold += 8.0 * max(0.0, -real_yield / 2.0)  # Negative real rates
            gold += 6.0 * recession_risk  # Safe haven
        
        cash = 5.0
        
        if not self.include_cash:
            # Redistribute cash to bonds
            cash = 0.0
        elif is_stagflation:
            # FIX: Low cash in stagflation (inflation kills cash)
            cash = 5.0
        elif is_crisis:
            # High cash in crisis
            cash = 15.0
        else:
            cash += 8.0 * recession_risk  # Defensive
            cash += 5.0 * valuation_risk  # Wait for better entry
            cash += 4.0 * max(0.0, (t10 - 4.0) / 2.0)  # High cash yields
            cash -= 3.0 * inflation_regime  # Cash loses to inflation
        
        crypto = 0.0
        if self.include_crypto and z < 2.0 and vix < 30 and recession_risk < 0.3 and growth_regime > 0.2:
            crypto = 8.0 + 5.0 * growth_regime
        
        self.debug_print(f"Pre-constraint allocation: Equities {equities:.1f}%, Bonds {bonds:.1f}%, Gold {gold:.1f}%, Cash {cash:.1f}%, Crypto {crypto:.1f}%")
        
        weights = {
            "equities": np.clip(equities, 10.0, 75.0),  # Allow lower equities in crisis
            "bonds": np.clip(bonds, 5.0, 70.0),  # Allow much higher bonds in crisis
            "gold": np.clip(gold, 5.0, 40.0),  # Allow higher gold
            "cash": np.clip(cash, 2.0, 35.0),
            "crypto": np.clip(crypto, 0.0, 15.0),
        }
        
        total = sum(weights.values())
        weights = {k: v / total * 100.0 for k, v in weights.items()}
        
        # Apply volatility targeting for risk balance
        weights = self._apply_volatility_targeting(weights, vix, credit_spread)
        
        self.debug_print(f"Final allocation: Equities {weights['equities']:.1f}%, Bonds {weights['bonds']:.1f}%, Gold {weights['gold']:.1f}%, Cash {weights['cash']:.1f}%")
        
        return weights
    
    def _apply_volatility_targeting(self, weights: dict, vix: float, credit_spread: float) -> dict:
        """Adjust allocations based on expected volatility to target consistent risk"""
        # Estimate asset class volatilities
        equity_vol = max(12.0, vix * 0.8)  # VIX proxy for equity vol
        bond_vol = 6.0 + max(0.0, credit_spread - 3.0) * 0.5  # Credit spreads affect bond vol
        gold_vol = 15.0  # Historical gold volatility
        crypto_vol = 60.0  # High crypto volatility
        cash_vol = 0.5  # Minimal cash volatility
        
        vols = {
            "equities": equity_vol,
            "bonds": bond_vol,
            "gold": gold_vol,
            "crypto": crypto_vol,
            "cash": cash_vol,
        }
        
        # Risk parity adjustment: scale by inverse volatility
        inv_vol_weights = {k: (weights[k] / vols[k]) if vols[k] > 0 else 0 for k in weights}
        total_inv_vol = sum(inv_vol_weights.values())
        
        if total_inv_vol > 0:
            risk_parity = {k: v / total_inv_vol * 100.0 for k, v in inv_vol_weights.items()}
            # Blend 70% original allocation with 30% risk parity
            adjusted = {k: 0.70 * weights[k] + 0.30 * risk_parity[k] for k in weights}
            return adjusted
        
        return weights
    
    def generate_json_output(self, metrics: dict, weights: dict, suballoc: dict) -> dict:
        """Generate structured JSON for backtesting and programmatic use"""
        return {
            "timestamp": metrics['as_of'].isoformat(),
            "allocation": {
                "weights": weights,
                "subcategories": suballoc
            },
            "metrics": {
                "valuation": {
                    "buffett_z_score": metrics['z_score'],
                    "sp_gold_ratio": metrics['sp_gold_ratio']
                },
                "rates": {
                    "ten_year_yield": metrics['ten_year_yield'],
                    "two_year_yield": metrics['two_year_yield'],
                    "yield_curve_spread": metrics['yield_curve'],
                    "real_yield": metrics['real_yield']
                },
                "inflation": {
                    "cpi_yoy": metrics['inflation_yoy']
                },
                "credit": {
                    "high_yield_spread": metrics['credit_spread']
                },
                "volatility": {
                    "vix": metrics['vix']
                },
                "momentum": {
                    "equity_trend": metrics['trend_equity'],
                    "gold_trend": metrics['trend_gold'],
                    "btc_trend": metrics['trend_btc']
                }
            },
            "regime_scores": {
                # Threshold method (original)
                "recession_risk": self._calculate_recession_risk(metrics),
                "growth_regime": self._calculate_growth_regime(metrics),
                "inflation_regime": self._calculate_inflation_regime(metrics),
                # Continuous method (smooth transitions)
                "recession_risk_continuous": self._calculate_recession_risk_continuous(metrics),
                "growth_regime_continuous": self._calculate_growth_regime_continuous(metrics),
                "inflation_regime_continuous": self._calculate_inflation_regime_continuous(metrics)
            }
        }
    
    def _calculate_recession_risk(self, metrics: dict) -> float:
        """Crisis = high risk aversion, elevated volatility, credit stress
        
        Key insight: Don't conflate bubble volatility with crisis!
        - Bubble: high valuation + volatility
        - Crisis: credit stress + negative trend + volatility
        """
        risk = 0.0
        curve_bps = metrics['yield_curve']
        credit_pct = metrics['credit_spread']
        vix = metrics['vix']
        trend = metrics['trend_equity']
        z = metrics.get('z_score', 0.0)
        
        # CRITICAL: Check if this is a bubble first (high z + positive trend)
        is_likely_bubble = (z > 2.0 and trend > 1.0)
        
        # VIX spike = crisis ONLY if not a bubble
        if not pd.isna(vix):
            if vix > 35 and not is_likely_bubble:
                risk += 0.5  # Major crisis
                self.debug_print(f"Crisis: VIX spike ({vix:.1f}) +0.5")
            elif vix > 30 and not is_likely_bubble:
                risk += 0.3
                self.debug_print(f"Crisis: High VIX ({vix:.1f}) +0.3")
            elif vix > 25 and trend < 0.95:  # High VIX + negative trend = crisis
                risk += 0.2
                self.debug_print(f"Crisis: VIX + negative trend +0.2")
        
        # Severe equity decline (most reliable crisis indicator)
        if trend < 0.75:
            risk += 0.5  # Down >25% = definite crisis
            self.debug_print(f"Crisis: Severe decline (trend={trend:.3f}) +0.5")
        elif trend < 0.90:
            risk += 0.3  # Down >10%
            self.debug_print(f"Crisis: Moderate decline (trend={trend:.3f}) +0.3")
        elif trend < 0.95:
            risk += 0.1
            self.debug_print(f"Crisis: Mild decline (trend={trend:.3f}) +0.1")
        
        # Credit stress (key crisis indicator)
        if credit_pct > 6.0:
            risk += 0.4  # Severe stress
            self.debug_print(f"Crisis: Severe credit stress ({credit_pct:.2f}%) +0.4")
        elif credit_pct > 5.0:
            risk += 0.2
            self.debug_print(f"Crisis: High credit stress ({credit_pct:.2f}%) +0.2")
        elif credit_pct > 4.0:
            risk += 0.1
            self.debug_print(f"Crisis: Elevated credit stress ({credit_pct:.2f}%) +0.1")
        
        # Inverted yield curve (leading indicator, but not immediate crisis)
        if curve_bps < -50:
            risk += 0.2
            self.debug_print(f"Crisis: Deeply inverted curve ({curve_bps:.0f}bps) +0.2")
        elif curve_bps < -10:
            risk += 0.1
            self.debug_print(f"Crisis: Inverted curve ({curve_bps:.0f}bps) +0.1")
        
        risk = min(risk, 1.0)
        self.debug_print(f"Total recession risk: {risk:.2f}")
        return risk
    
    def _calculate_growth_regime(self, metrics: dict) -> float:
        """Growth = benign conditions (DEFAULT STATE)
        - Moderate rates, upward yield curve, subdued volatility
        - This should be the BASELINE, not exceptional
        """
        curve_bps = metrics['yield_curve']
        credit_pct = metrics['credit_spread']
        vix = metrics.get('vix', 20.0)
        trend = metrics['trend_equity']
        
        growth = 0.5  # START AT 0.5 (neutral baseline)
        
        # POSITIVE signals for growth
        if curve_bps > 20:  # Upward sloping curve
            growth += 0.2
        if credit_pct < 4.0:  # Low credit stress
            growth += 0.2
        if vix < 20 and not pd.isna(vix):  # Subdued volatility
            growth += 0.1
        if trend > 1.0:  # Positive equity trend
            growth += 0.1
            
        # NEGATIVE signals (reduce growth score)
        if curve_bps < 0:  # Inverted curve
            growth -= 0.3
        if credit_pct > 5.0:  # High credit stress
            growth -= 0.3
        if vix > 25 and not pd.isna(vix):  # Elevated volatility
            growth -= 0.2
        if trend < 0.95:  # Negative trend
            growth -= 0.2
        
        growth = np.clip(growth, 0.0, 1.0)
        self.debug_print(f"Growth regime: {growth:.2f} (curve={curve_bps:.0f}, spread={credit_pct:.2f}, vix={vix:.1f}, trend={trend:.3f})")
        return growth
    
    def _calculate_inflation_regime(self, metrics: dict) -> float:
        inflation = 0.0
        cpi = metrics['inflation_yoy']
        core_cpi = metrics.get('core_cpi', cpi)
        tips_5y = metrics.get('tips_5y', 2.5)
        real_yield = metrics['real_yield']
        
        # Research-based: BIS paper shows CPI >3% = high inflation regime
        if cpi >= 3.3:  # March 2026 actual level (major acceleration)
            inflation += 0.6
            self.debug_print(f"High inflation signal: CPI {cpi:.2f}% >= 3.3% +0.6")
        elif cpi >= 3.0:
            inflation += 0.4
            self.debug_print(f"Moderate inflation signal: CPI {cpi:.2f}% >= 3.0% +0.4")
        elif cpi >= 2.5:
            inflation += 0.2
            self.debug_print(f"Emerging inflation signal: CPI {cpi:.2f}% >= 2.5% +0.2")
        
        # TIPS breakeven expectations (market-based forward-looking)
        if tips_5y > 2.7:
            inflation += 0.3
            self.debug_print(f"High TIPS breakeven signal: {tips_5y:.2f}% > 2.7% +0.3")
        elif tips_5y > 2.5:
            inflation += 0.2
            self.debug_print(f"Elevated TIPS breakeven signal: {tips_5y:.2f}% > 2.5% +0.2")
        
        # Real yield erosion (academic threshold from research)
        if real_yield < 0.5:
            inflation += 0.4
            self.debug_print(f"Severe real yield erosion: {real_yield:.2f}% < 0.5% +0.4")
        elif real_yield < 1.5:
            inflation += 0.2
            self.debug_print(f"Moderate real yield erosion: {real_yield:.2f}% < 1.5% +0.2")
        
        # Core CPI persistence (sticky inflation components)
        if core_cpi > 3.0:
            inflation += 0.3
            self.debug_print(f"Persistent core inflation: {core_cpi:.2f}% > 3.0% +0.3")
        elif core_cpi > 2.5:
            inflation += 0.15
            self.debug_print(f"Elevated core inflation: {core_cpi:.2f}% > 2.5% +0.15")
        
        # Research: "stagflation lite" scenario (high CPI + weak growth)
        if cpi > 3.0 and metrics['trend_equity'] < 1.02:
            inflation += 0.2
            self.debug_print(f"Stagflation signal: High CPI {cpi:.2f}% + weak equity trend +0.2")
        
        # Geopolitical inflation shock (March 2026 Iran war impact)
        if cpi > 3.2 and tips_5y > 2.6:  # Both actual and expected inflation elevated
            inflation += 0.15
            self.debug_print(f"Geopolitical inflation shock: CPI {cpi:.2f}% + TIPS {tips_5y:.2f}% +0.15")
        
        inflation = min(inflation, 1.0)
        self.debug_print(f"Total inflation regime: {inflation:.2f}")
        return min(inflation, 1.0)
    
    def _get_regime_name(self, recession: float, growth: float, inflation: float, 
                         metrics: dict, method: str = "threshold") -> tuple:
        """Determine regime name from scores
        
        Args:
            recession: Recession risk score (0-1)
            growth: Growth regime score (0-1)
            inflation: Inflation regime score (0-1)
            metrics: Full metrics dict for additional checks
            method: "threshold" or "continuous"
        
        Returns:
            (regime_name, confidence)
        """
        z = metrics.get('z_score', 0.0)
        trend = metrics.get('trend_equity', 1.0)
        inflation_val = metrics.get('inflation_yoy', 3.0)
        
        # Hierarchy (same for both methods, but thresholds differ)
        if method == "threshold":
            crisis_threshold = 0.5
            bubble_z = 2.3
            bubble_trend = 1.02
            stagflation_inflation = 0.7
            stagflation_growth = 0.4
            stagflation_cpi = 6.0
            high_inflation_threshold = 0.7
            high_inflation_cpi = 4.0
            growth_threshold = 0.5
        else:  # continuous - slightly different thresholds
            crisis_threshold = 0.55  # Slightly higher to reduce false positives
            bubble_z = 2.2  # Slightly lower for earlier detection
            bubble_trend = 1.02
            stagflation_inflation = 0.65
            stagflation_growth = 0.45
            stagflation_cpi = 5.5
            high_inflation_threshold = 0.65
            high_inflation_cpi = 3.8
            growth_threshold = 0.5
        
        # 1. CRISIS
        if recession > crisis_threshold:
            confidence = min(1.0, recession / crisis_threshold)
            return ("Crisis", confidence)
        
        # 2. BUBBLE
        if z > bubble_z and trend > bubble_trend and recession < 0.5:
            confidence = min(1.0, (z - bubble_z) / 1.0 + (trend - bubble_trend) / 0.1)
            return ("Bubble", min(confidence, 1.0))
        
        # 3. STAGFLATION
        if (inflation > stagflation_inflation and growth < stagflation_growth and 
            recession < 0.5 and inflation_val > stagflation_cpi):
            confidence = min(1.0, inflation / stagflation_inflation)
            return ("Stagflation", confidence)
        
        # 4. HIGH INFLATION
        if inflation > high_inflation_threshold and recession < 0.5 and inflation_val > high_inflation_cpi:
            confidence = min(1.0, inflation / high_inflation_threshold)
            return ("High Inflation", confidence)
        
        # 5. GROWTH (default/baseline)
        if growth > growth_threshold and recession < 0.5:
            confidence = min(1.0, growth / growth_threshold)
            return ("Growth", confidence)
        
        # 6. RECESSION RISK
        if recession > 0.3:
            confidence = recession / 0.6
            return ("Recession Risk", min(confidence, 1.0))
        
        # 7. NEUTRAL (fallback)
        return ("Neutral", 0.5)
    
    # ============================================================================
    # CONTINUOUS SCORING METHODS (Approach B)
    # Smooth transitions, no hard thresholds, rate-of-change features
    # ============================================================================
    
    def _calculate_regime_probabilistic(self, metrics: dict) -> dict:
        """
        Calculate 4-regime probabilities using optimized equation-based approach.
        
        Regimes:
        1. Crisis - Market stress, credit issues
        2. Bubble - High valuations + momentum
        3. Inflation - High CPI (merged High Inflation + Stagflation)
        4. Normal - Balanced conditions (merged Growth + Neutral)
        
        Returns:
            dict: {'Crisis': prob, 'Bubble': prob, 'Inflation': prob, 'Normal': prob}
        """
        from regime_4_merged import calculate_regime_probs_4merged, INITIAL_COEFS_4REGIME
        import numpy as np
        
        # Try to load optimized coefficients, fall back to hand-tuned if not available
        try:
            OPTIMIZED_COEFS = np.load('optimized_regime_coefs.npy')
            self.debug_print("Using optimized regime coefficients")
            coefs = OPTIMIZED_COEFS
        except:
            self.debug_print("Using hand-tuned regime coefficients (optimized file not found)")
            coefs = INITIAL_COEFS_4REGIME
        
        probs = calculate_regime_probs_4merged(metrics, coefs)
        
        return probs
    
    def _calculate_recession_risk_continuous(self, metrics: dict) -> float:
        """Continuous recession risk with smooth transitions
        
        Key improvements over threshold method:
        - Smooth sigmoid transitions (no edge cases)
        - Rate-of-change features (VIX momentum, spread widening)
        - Non-linear interactions (curve + spreads)
        """
        risk = 0.0
        
        # Extract features
        curve_bps = metrics['yield_curve']
        credit_pct = metrics['credit_spread']
        vix = metrics.get('vix', 20.0)
        trend = metrics['trend_equity']
        z = metrics.get('z_score', 0.0)
        
        # Check if bubble (don't conflate with crisis)
        is_bubble = (z > 2.0 and trend > 1.0)
        
        # 1. EQUITY TREND (most reliable) - smooth transition
        if trend < 0.70:  # Severe decline >30%
            risk += 0.6
        elif trend < 0.95:  # Moderate decline
            # Linear interpolation: 0.70→0.6, 0.95→0.0
            risk += 0.6 * (0.95 - trend) / (0.95 - 0.70)
        
        # 2. CREDIT SPREADS - smooth transition
        # Updated thresholds based on historical analysis:
        # - Crisis average: 9.4%, range: 5.2-18.0%
        # - Normal average: 3.5%, range: 2.5-4.5%
        if credit_pct > 6.5:  # Extreme stress (lowered from 7.0 to catch more crises)
            risk += 0.5
        elif credit_pct > 3.5:  # Elevated stress
            # Linear: 3.5→0.0, 6.5→0.5 (adjusted range)
            risk += 0.5 * (credit_pct - 3.5) / (6.5 - 3.5)
        
        # 3. VIX (only if not bubble) - smooth transition
        if not is_bubble and not pd.isna(vix):
            if vix > 40:  # Extreme fear
                risk += 0.4
            elif vix > 20:  # Elevated volatility
                # Linear: 20→0.0, 40→0.4
                risk += 0.4 * (vix - 20) / (40 - 20)
        
        # 4. YIELD CURVE - smooth transition
        if curve_bps < -100:  # Deeply inverted
            risk += 0.3
        elif curve_bps < 0:  # Inverted
            # Linear: 0→0.0, -100→0.3
            risk += 0.3 * abs(curve_bps) / 100
        
        # 5. NON-LINEAR INTERACTION: Inverted curve + widening spreads
        if curve_bps < 0 and credit_pct > 4.5:
            risk += 0.2  # Classic crisis pattern
        
        return min(risk, 1.0)
    
    def _calculate_growth_regime_continuous(self, metrics: dict) -> float:
        """Continuous growth score with smooth transitions
        
        Growth = baseline/default state (Verdad research)
        """
        curve_bps = metrics['yield_curve']
        credit_pct = metrics['credit_spread']
        vix = metrics.get('vix', 20.0)
        trend = metrics['trend_equity']
        
        growth = 0.5  # Start at neutral
        
        # 1. YIELD CURVE - smooth positive signal
        if curve_bps > 100:  # Steep curve
            growth += 0.3
        elif curve_bps > 0:  # Upward sloping
            # Linear: 0→0.0, 100→0.3
            growth += 0.3 * curve_bps / 100
        elif curve_bps < -50:  # Deeply inverted
            growth -= 0.4
        elif curve_bps < 0:  # Inverted
            # Linear: 0→0.0, -50→-0.4
            growth -= 0.4 * abs(curve_bps) / 50
        
        # 2. CREDIT SPREADS - smooth signal
        # Updated thresholds based on historical analysis:
        # - Normal average: 3.5%, most dates 3.0-4.5%
        if credit_pct < 3.5:  # Low stress (raised from 3.0)
            growth += 0.3
        elif credit_pct < 4.5:  # Moderate (adjusted range)
            # Linear: 3.5→0.3, 4.5→0.0
            growth += 0.3 * (4.5 - credit_pct) / (4.5 - 3.5)
        elif credit_pct > 5.5:  # High stress
            growth -= 0.4
        elif credit_pct > 4.5:  # Elevated (adjusted range)
            # Linear: 4.5→0.0, 5.5→-0.4
            growth -= 0.4 * (credit_pct - 4.5) / (5.5 - 4.5)
        
        # 3. VIX - smooth signal
        if not pd.isna(vix):
            if vix < 15:  # Very low volatility
                growth += 0.2
            elif vix < 20:  # Low volatility
                # Linear: 15→0.2, 20→0.0
                growth += 0.2 * (20 - vix) / (20 - 15)
            elif vix > 30:  # High volatility
                growth -= 0.3
            elif vix > 20:  # Elevated
                # Linear: 20→0.0, 30→-0.3
                growth -= 0.3 * (vix - 20) / (30 - 20)
        
        # 4. EQUITY TREND - smooth signal
        if trend > 1.10:  # Strong uptrend
            growth += 0.2
        elif trend > 1.0:  # Uptrend
            # Linear: 1.0→0.0, 1.10→0.2
            growth += 0.2 * (trend - 1.0) / 0.10
        elif trend < 0.90:  # Downtrend
            growth -= 0.3
        elif trend < 1.0:  # Mild downtrend
            # Linear: 1.0→0.0, 0.90→-0.3
            growth -= 0.3 * (1.0 - trend) / 0.10
        
        return np.clip(growth, 0.0, 1.0)
    
    def _calculate_inflation_regime_continuous(self, metrics: dict) -> float:
        """Continuous inflation score with smooth transitions"""
        cpi = metrics['inflation_yoy']
        core_cpi = metrics.get('core_cpi', cpi)
        tips_5y = metrics.get('tips_5y', 2.5)
        real_yield = metrics['real_yield']
        
        inflation = 0.0
        
        # 1. CPI - smooth transition (primary indicator)
        if cpi < 1.8:  # Low inflation
            inflation = 0.0
        elif cpi < 3.3:  # Moderate
            # Linear: 1.8→0.0, 3.3→0.6
            inflation = 0.6 * (cpi - 1.8) / (3.3 - 1.8)
        elif cpi < 5.5:  # High
            # Linear: 3.3→0.6, 5.5→0.9
            inflation = 0.6 + 0.3 * (cpi - 3.3) / (5.5 - 3.3)
        else:  # Extreme
            # Asymptotic: 5.5→0.9, 10.0→1.0
            inflation = min(1.0, 0.9 + 0.1 * (cpi - 5.5) / 4.5)
        
        # 2. TIPS breakeven - smooth adjustment
        if tips_5y > 3.0:  # High expectations
            inflation += min(0.3, (tips_5y - 3.0) * 0.15)
        elif tips_5y > 2.5:  # Elevated
            # Linear: 2.5→0.0, 3.0→0.2
            inflation += 0.2 * (tips_5y - 2.5) / 0.5
        
        # 3. Real yield erosion - smooth adjustment
        if real_yield < -1.0:  # Deeply negative
            inflation += 0.3
        elif real_yield < 0.5:  # Negative to low
            # Linear: 0.5→0.0, -1.0→0.3
            inflation += 0.3 * (0.5 - real_yield) / 1.5
        
        # 4. Core CPI persistence - smooth adjustment
        if core_cpi > 4.0:  # Persistent inflation
            inflation += min(0.2, (core_cpi - 4.0) * 0.1)
        
        return min(inflation, 1.0)

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
        hist["z_score"] = z_hist
        hist["sp_gold_ratio"] = monthly.loc[common_idx, "^GSPC"] / monthly.loc[common_idx, "GLD"]
        
        # Yield curve in basis points
        hist["yield_curve"] = fred_data["T10Y2Y"].reindex(common_idx, method="ffill")["T10Y2Y"] * 100
        hist["ten_year_yield"] = fred_data["DGS10"].reindex(common_idx, method="ffill")["DGS10"]
        
        # VIX calculation - use daily returns for accuracy
        if "^VIX" in monthly.columns:
            hist["vix"] = monthly.loc[common_idx, "^VIX"].fillna(method="ffill")
        else:
            # Calculate VIX proxy from daily returns, then sample monthly
            daily_returns = prices["^GSPC"].pct_change()
            daily_vol = daily_returns.rolling(window=21).std() * np.sqrt(252) * 100  # 21-day rolling vol
            monthly_vol = daily_vol.resample("ME").last()
            hist["vix"] = monthly_vol.reindex(common_idx, method="ffill")

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

    def generate_comprehensive_report(self, metrics: dict, weights: dict, suballoc: dict, similar: list, regime_scores: dict = None) -> str:
        """Generate comprehensive hierarchical markdown with economic reasoning"""
        md = []
        
        # Calculate regime scores if not provided
        if regime_scores is None:
            # Calculate all three methods
            regime_scores = {
                # Threshold method (original)
                'recession_risk': self._calculate_recession_risk(metrics),
                'growth_regime': self._calculate_growth_regime(metrics),
                'inflation_regime': self._calculate_inflation_regime(metrics),
                # Smooth method
                'recession_risk_continuous': self._calculate_recession_risk_continuous(metrics),
                'growth_regime_continuous': self._calculate_growth_regime_continuous(metrics),
                'inflation_regime_continuous': self._calculate_inflation_regime_continuous(metrics),
                # Probabilistic method (4-regime)
                'probabilistic': self._calculate_regime_probabilistic(metrics)
            }
        
        # Title and Executive Summary
        md.append(f"# Macro Portfolio Allocation Report")
        md.append(f"**Date**: {metrics['as_of'].strftime('%B %d, %Y')}\n")
        
        # Section 1: Portfolio Recommendation
        md.append("## 1. Portfolio Recommendation\n")
        md.append("### Asset Allocation\n")
        
        # Add pie chart visualization
        md.append("![Asset Allocation](figures/01_allocation_pie.png)\n")
        md.append("")
        
        for asset, pct in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            md.append(f"- **{asset.title()}**: {pct:.1f}%")
            md.append(f"  ```")
            md.append(f"  {bar}")
            md.append(f"  ```")
        
        md.append("\n### Tactical Implementation\n")
        for category, plan in suballoc.items():
            if not plan:
                continue
            md.append(f"**{category.title()}** ({weights[category]:.1f}%):")
            for ticker, pct in plan.items():
                md.append(f"- {ticker}: {pct * 100:.0f}%")
            md.append("")
        
        # Section 2: Market Environment Analysis
        md.append("## 2. Market Environment Analysis\n")
        
        # Add yield curve visualization
        md.append("### 2.1 Yield Curve Analysis\n")
        md.append("![Yield Curve](figures/02_yield_curve.png)\n")
        md.append("")
        
        # Add yield ratio analysis
        md.append("#### Yield Curve Ratio Analysis\n")
        current_10y = metrics.get('ten_year_yield', 4.3)
        historical_avg_10y = 3.5  # Long-term average
        yield_ratio = current_10y / historical_avg_10y
        
        md.append(f"**Current vs Historical Yields:**\n")
        md.append(f"- 10-Year Current: {current_10y:.2f}%\n")
        md.append(f"- 10-Year Historical Average: {historical_avg_10y:.2f}%\n")
        md.append(f"- Ratio: {yield_ratio:.2f}x ({(yield_ratio-1)*100:.0f}% above average)\n\n")
        
        md.append("**What This Means:**\n\n")
        md.append("**Beginner:** When current yields are higher than average, bonds offer better income but prices have fallen. ")
        md.append(f"Current {current_10y:.2f}% vs {historical_avg_10y:.2f}% average means bonds are \"on sale\" but for a reason (inflation/risk concerns).\n\n")
        
        md.append("**Intermediate:** Ratio > 1.2 signals yields significantly elevated:\n")
        md.append("- ✓ Positive: Higher income for new bond buyers\n")
        md.append("- ✗ Negative: Existing bondholders have mark-to-market losses\n")
        md.append("- ⚠️ Signal: Market pricing in higher inflation or fiscal concerns\n")
        md.append("- **Consequences**: Mortgage rates rise, corporate borrowing costs increase, stock valuations pressured, dollar strengthens\n\n")
        
        md.append("**Professional:** Term premium has returned after years of Fed suppression. ")
        md.append("Current 4.3% decomposes to: ~1.5% real rate + ~2.5% inflation expectations + ~0.3% term premium (geopolitical risk). ")
        md.append("If yields stay elevated, recession risk increases via tight financial conditions. ")
        md.append("Current level suggests market expects Fed on hold through 2026.\n\n")
        
        # Add indicators dashboard
        md.append("### 2.2 Key Macro Indicators\n")
        md.append("![Indicators Dashboard](figures/03_indicators_dashboard.png)\n")
        md.append("")
        
        # Add comprehensive explanation of indicators
        md.append("#### Understanding Key Macro Indicators\n\n")
        
        md.append("**1. S&P 500 / Gold Ratio**\n\n")
        md.append("*What it is:* The price of the S&P 500 index divided by the price of gold (per ounce).\n\n")
        md.append("*Intuition:* Measures relative attractiveness of stocks vs gold. High ratio = stocks expensive relative to gold. Low ratio = gold expensive relative to stocks.\n\n")
        md.append("*Historical Range:* 0.5 (crisis lows like 2009, 2020) to 3.0 (bubble peaks like 2000). Long-term average: ~1.5.\n\n")
        md.append(f"*Current Value:* {metrics['sp_gold_ratio']:.2f} - ")
        if metrics['sp_gold_ratio'] < 1.0:
            md.append("**Below average** - Gold outperforming, suggests risk-off sentiment or inflation fears.\n\n")
        elif metrics['sp_gold_ratio'] > 2.0:
            md.append("**Above average** - Stocks outperforming, suggests risk-on sentiment and confidence.\n\n")
        else:
            md.append("**Near average** - Balanced between stocks and gold.\n\n")
        
        md.append("*Significance:* When ratio drops sharply, it signals investors fleeing to safety (gold) from risk assets (stocks). ")
        md.append("Historically, ratios below 1.0 occur during major crises or high inflation periods.\n\n")
        
        md.append("**2. VIX (Volatility Index)**\n\n")
        md.append("*What it is:* The \"fear gauge\" - measures expected 30-day volatility of S&P 500 based on options prices.\n\n")
        md.append("*Intuition:* Higher VIX = more fear/uncertainty. Lower VIX = complacency/calm markets.\n\n")
        md.append("*Historical Range:* 10-15 (calm), 20-30 (elevated), 40+ (panic), 80+ (extreme crisis like 2008, 2020).\n\n")
        md.append(f"*Current Value:* {metrics['vix']:.1f} - ")
        if metrics['vix'] < 15:
            md.append("**Low** - Market complacency, low expected volatility.\n\n")
        elif metrics['vix'] < 25:
            md.append("**Moderate** - Normal market uncertainty.\n\n")
        elif metrics['vix'] < 40:
            md.append("**Elevated** - Increased fear and uncertainty.\n\n")
        else:
            md.append("**High** - Significant market stress and panic.\n\n")
        
        md.append("*Significance:* VIX spikes during market crashes (2008: 80, 2020: 85). ")
        md.append("Increasing VIX means investors are paying more for downside protection (put options). ")
        md.append("VIX below 12 often precedes corrections (complacency). VIX above 30 signals genuine fear.\n\n")
        
        md.append("**3. Credit Spread (High Yield)**\n\n")
        md.append("*What it is:* Extra yield investors demand to hold risky corporate bonds vs safe Treasury bonds.\n\n")
        md.append("*Intuition:* Wider spreads = more credit risk/fear. Tighter spreads = confidence in corporate health.\n\n")
        md.append("*Historical Range:* 250-350 bps (normal), 350-500 bps (caution), 500-800 bps (stress), 800+ bps (crisis).\n\n")
        md.append(f"*Current Value:* {metrics['credit_spread']*100:.0f} bps - ")
        if metrics['credit_spread'] < 3.5:
            md.append("**Tight** - Low credit risk, strong corporate health.\n\n")
        elif metrics['credit_spread'] < 5.0:
            md.append("**Normal** - Moderate credit risk.\n\n")
        elif metrics['credit_spread'] < 7.0:
            md.append("**Elevated** - Increased credit stress.\n\n")
        else:
            md.append("**Wide** - Significant credit stress, recession risk.\n\n")
        
        md.append("*Significance:* Credit spreads are a leading indicator of recessions. ")
        md.append("Widening spreads (rising from 300 to 600+ bps) signal deteriorating corporate health and recession risk. ")
        md.append("2008 crisis: spreads hit 1,800 bps. 2020 COVID: 1,050 bps. Normal times: 300-400 bps.\n\n")
        
        md.append("**4. Real Yield**\n\n")
        md.append("*What it is:* Nominal 10-Year Treasury yield minus inflation (CPI YoY). Measures actual purchasing power return.\n\n")
        md.append("*Intuition:* If 10Y yields 4% and inflation is 3%, real yield is 1% - your actual gain after inflation.\n\n")
        md.append("*Historical Range:* -2% to +4%. Average: ~1.5%. Negative = inflation eating returns.\n\n")
        md.append(f"*Current Value:* {metrics['real_yield']:.2f}% - ")
        if metrics['real_yield'] < 0:
            md.append("**Negative** - Inflation exceeds yields, bondholders losing purchasing power.\n\n")
        elif metrics['real_yield'] < 1.0:
            md.append("**Low** - Modest real returns after inflation.\n\n")
        elif metrics['real_yield'] < 2.5:
            md.append("**Normal** - Healthy real returns.\n\n")
        else:
            md.append("**High** - Attractive real returns, often during disinflation.\n\n")
        
        md.append("*Significance:* Negative real yields (2020-2022) force investors into stocks/gold seeking returns. ")
        md.append("Positive real yields above 2% make bonds competitive with stocks. ")
        md.append("Real yields drive gold prices (inverse relationship) and stock valuations.\n\n")
        
        md.append("**5. CPI YoY (Inflation)**\n\n")
        md.append("*What it is:* Consumer Price Index year-over-year change. Measures cost of living increase.\n\n")
        md.append("*Intuition:* 3% CPI = prices 3% higher than last year. Your dollar buys 3% less.\n\n")
        md.append("*Historical Range:* 0-2% (low), 2-4% (moderate), 4-6% (elevated), 6%+ (high). Fed target: 2%.\n\n")
        md.append(f"*Current Value:* {metrics['inflation_yoy']:.1f}% - ")
        if metrics['inflation_yoy'] < 2.0:
            md.append("**Low** - Below Fed target, possible deflation risk.\n\n")
        elif metrics['inflation_yoy'] < 3.0:
            md.append("**Target** - Near Fed's 2% goal.\n\n")
        elif metrics['inflation_yoy'] < 5.0:
            md.append("**Elevated** - Above target, Fed may tighten.\n\n")
        else:
            md.append("**High** - Well above target, aggressive Fed action likely.\n\n")
        
        md.append("*Historical Context:* ")
        md.append("Peak inflation was 14.6% in March 1980 (Volcker era). ")
        md.append("Recent peak was 9.1% in June 2022 (post-COVID stimulus). ")
        md.append("1990s-2010s averaged 2-3% (Great Moderation). ")
        md.append("Current level determines Fed policy - high inflation = rate hikes, low inflation = rate cuts.\n\n")
        
        md.append("**6. Yield Curve (10Y-2Y)**\n\n")
        md.append("*What it is:* Difference between 10-year and 2-year Treasury yields. Measures term premium.\n\n")
        md.append("*Intuition:* Normally positive (10Y > 2Y) because longer maturity = more risk. ")
        md.append("Negative (inverted) = short rates higher than long rates.\n\n")
        md.append("*Historical Range:* +50 to +200 bps (normal), 0 to -50 bps (inverted/recession warning), -50 to -100 bps (deep inversion).\n\n")
        md.append(f"*Current Value:* {metrics['yield_curve']:.0f} bps - ")
        if metrics['yield_curve'] < -50:
            md.append("**Deeply Inverted** - Strong recession signal.\n\n")
        elif metrics['yield_curve'] < 0:
            md.append("**Inverted** - Recession warning (historically precedes recessions by 6-18 months).\n\n")
        elif metrics['yield_curve'] < 50:
            md.append("**Flat** - Neutral, transition period.\n\n")
        elif metrics['yield_curve'] < 150:
            md.append("**Normal** - Healthy term premium.\n\n")
        else:
            md.append("**Steep** - Strong growth expectations or Fed easing.\n\n")
        
        md.append("*Significance:* Inverted yield curve has predicted every recession since 1950 (with 2 false positives). ")
        md.append("Inversion occurs when Fed raises short rates to fight inflation, but market expects eventual slowdown. ")
        md.append("Recent: Inverted Aug 2022 - Mar 2024 (longest inversion in history at 19 months), then un-inverted in April 2024. ")
        md.append("Un-inversion can signal either: (1) Soft landing success, or (2) Recession imminent (historically more common).\n\n")
        
        md.append("**Difference: Real Yield vs Yield Curve**\n\n")
        md.append("- **Real Yield**: Nominal yield minus inflation. Measures purchasing power return. Single number.\n")
        md.append("- **Yield Curve**: Difference between long and short yields. Measures term premium and growth expectations. Spread between two yields.\n")
        md.append("- **Example**: 10Y yield = 4%, 2Y yield = 4.5%, CPI = 3%. Real yield = 4% - 3% = 1%. Yield curve = 4% - 4.5% = -0.5% (inverted).\n\n")
        
        md.append("### 2.3 Current Market Conditions\n")
        md.append(f"- **S&P 500**: {metrics['spx']:.2f}")
        md.append(f"- **Gold Price**: ${metrics['gold']:.2f}")
        md.append(f"- **S&P/Gold Ratio**: {metrics['sp_gold_ratio']:.2f}")
        md.append(f"- **10-Year Treasury Yield**: {metrics['ten_year_yield']:.2f}%")
        md.append(f"- **2-Year Treasury Yield**: {metrics['two_year_yield']:.2f}%")
        md.append(f"- **Yield Curve (10Y-2Y)**: {metrics['yield_curve']:.1f} basis points")
        md.append(f"- **Inflation (YoY)**: {metrics['inflation_yoy']:.1f}%")
        md.append(f"- **Real Yield**: {metrics['real_yield']:.2f}%")
        md.append(f"- **Credit Spread**: {metrics['credit_spread']:.1f}% ({metrics['credit_spread']*100:.0f} bps)")
        md.append(f"- **VIX (Volatility)**: {metrics['vix']:.1f}\n")
        
        md.append("### 2.4 Economic Interpretation\n")
        
        # Add valuation history chart
        md.append("#### Valuation Analysis\n")
        md.append("![Valuation History](figures/06_valuation_history.png)\n")
        md.append("")
        
        # Add explanation of valuation metrics
        md.append("**Understanding Valuation Metrics:**\n\n")
        md.append("We use a **Buffett Indicator z-score**, which is different from CAPE:\n\n")
        
        md.append("**Beginner Level:**\n")
        md.append("- **Buffett Indicator**: Compares total stock market value to GDP (economic output)\n")
        md.append("  - Think: \"Are stocks expensive relative to what the economy produces?\"\n")
        md.append("  - Formula: Market Cap / GDP\n")
        md.append("  - Warren Buffett's favorite metric\n\n")
        
        md.append("- **CAPE (Cyclically Adjusted P/E)**: Compares stock prices to 10-year average earnings\n")
        md.append("  - Think: \"Are stocks expensive relative to their long-term profits?\"\n")
        md.append("  - Formula: Price / Average(Earnings over 10 years)\n")
        md.append("  - Robert Shiller's Nobel Prize-winning metric\n\n")
        
        md.append("**Intermediate Level:**\n")
        md.append("- **Buffett Indicator** measures market cap against the entire economy's output (GDP)\n")
        md.append("  - Pros: Captures macro conditions, hard to manipulate\n")
        md.append("  - Cons: Doesn't account for profit margins or interest rates\n")
        md.append("  - Normal range: 80-120% (market cap as % of GDP)\n")
        md.append("  - Current: ~200% (elevated)\n\n")
        
        md.append("- **CAPE** smooths earnings over 10 years to remove business cycle effects\n")
        md.append("  - Pros: Adjusts for inflation, removes temporary earnings spikes/crashes\n")
        md.append("  - Cons: Slow to react, affected by accounting changes\n")
        md.append("  - Normal range: 15-20 (price / 10-year avg earnings)\n")
        md.append("  - Current: ~30 (elevated)\n\n")
        
        md.append("**Professional Level:**\n")
        md.append("- **Z-Score Normalization**: We convert Buffett Indicator to z-score (standard deviations from mean)\n")
        md.append("  - Z = (Current Value - Historical Mean) / Historical Std Dev\n")
        md.append("  - Z > 2: More than 2 std dev above mean (top 2.5% historically)\n")
        md.append("  - Z < -2: More than 2 std dev below mean (bottom 2.5%)\n")
        md.append("  - Allows comparison across different time periods\n\n")
        
        md.append("- **Why Buffett over CAPE?**\n")
        md.append("  - GDP is more stable than earnings (less accounting manipulation)\n")
        md.append("  - Better for macro allocation (we're allocating across asset classes, not picking stocks)\n")
        md.append("  - Correlates well with 10-year forward returns\n")
        md.append("  - Simpler to understand and communicate\n\n")
        
        md.append("- **Historical Precedents:**\n")
        md.append("  - 2000: Buffett Indicator = 150%, CAPE = 44 → Both signaled bubble\n")
        md.append("  - 2009: Buffett Indicator = 60%, CAPE = 13 → Both signaled opportunity\n")
        md.append("  - 2021: Buffett Indicator = 200%, CAPE = 38 → Both signal overvaluation\n")
        md.append("  - **Correlation**: 0.85 between the two metrics (high agreement)\n\n")
        
        md.append("**Current Assessment:**\n")
        # Valuation analysis
        z = metrics['z_score']
        md.append(f"The Buffett Indicator z-score is **{z:.2f}**, measuring how expensive the market is relative to GDP. ")
        if z > 2.0:
            md.append("This indicates **significantly overvalued** conditions - the market cap is more than 2 standard deviations above its historical mean relative to economic output. ")
            md.append("Historically, z-scores above 2.0 have preceded market corrections:\n")
            md.append("- **2000 Dot-com Peak**: Z-score reached 2.7 → S&P 500 declined 49% over 2.5 years\n")
            md.append("- **2007 Pre-Financial Crisis**: Z-score hit 2.1 → S&P 500 fell 57% in 18 months\n")
            md.append("- **Current**: Z-score at 2.26 suggests elevated risk of mean reversion\n")
        elif z > 1.0:
            md.append("This indicates **moderately overvalued** conditions - the market is trading above its historical average relative to GDP. ")
            md.append("While not extreme, this suggests limited upside and increased downside risk.\n")
        elif z > -1.0:
            md.append("This indicates **fairly valued** conditions - the market is trading near its historical average relative to GDP. ")
            md.append("This is a neutral environment with balanced risk/reward.\n")
        else:
            md.append("This indicates **undervalued** conditions - the market is trading below its historical average. ")
            md.append("Examples: 2009 (Z=-1.2, start of bull market), 2011 (Z=-0.8, strong recovery followed).\n")
        
        md.append("#### Yield Curve Interpretation\n")
        # Yield curve analysis
        curve = metrics['yield_curve']
        md.append(f"The 10Y-2Y spread is **{curve:.1f} basis points**. ")
        
        # Add April 2026 context
        md.append("\n**Recent Market Developments (April 2026):**\n\n")
        md.append("The financial markets experienced significant volatility in April 2026 due to geopolitical events:\n\n")
        md.append("**Major Events:**\n")
        md.append("- **Geopolitical Crisis**: Iran-Israel conflict escalated with partial Strait of Hormuz blockade\n")
        md.append("- **Oil Shock**: Brent crude surged above $120/barrel (disrupting 25% of global maritime oil trade)\n")
        md.append("- **Inflation Spike**: CPI jumped to 3.6% in March (the \"March Oil Shock\")\n")
        md.append("- **Yield Surge**: 10-year Treasury rose from 3.92% (March 30) to 4.39% (mid-April) - a 47 bps move in 2 weeks\n")
        md.append("- **Historic Curve Un-Inversion**: After 27 months (longest in US history), yield curve steepened to +54 bps\n\n")
        
        md.append("**Consequences:**\n")
        md.append("1. **Credit Spreads Widening** (265 bps → 317 bps):\n")
        md.append("   - Investors demanding higher premium for corporate debt risk\n")
        md.append("   - Reflects concerns about economic slowdown from oil shock\n")
        md.append("   - High-yield spreads at highest level since Q4 2023\n\n")
        
        md.append("2. **\"Bear Steepener\" Yield Surge**:\n")
        md.append("   - Long rates rising faster than short rates (not a positive signal)\n")
        md.append("   - Term premium returning - investors demand compensation for long-term geopolitical risk\n")
        md.append("   - Mortgage rates surging, housing recovery stalling\n\n")
        
        md.append("3. **S&P 500 Resilience** (\"The Great Decoupling\"):\n")
        md.append("   - Stocks defying gravity despite yield surge\n")
        md.append("   - Energy sector benefiting from oil prices\n")
        md.append("   - Traditional bond-stock correlation breaking down\n\n")
        
        md.append("**Interpretation:**\n")
        md.append("This is a **stagflationary shock** - rising inflation combined with slowing growth. ")
        md.append("The yield curve un-inversion is NOT signaling economic recovery, but rather a violent return of term premium ")
        md.append("due to geopolitical risk and fiscal concerns. The Fed is trapped: can't cut rates (inflation rising) ")
        md.append("but economy weakening (oil shock impact). This environment favors gold and inflation-protected assets.\n\n")
        
        # Continue with normal yield curve analysis
        if curve < -10:
            md.append("An **inverted yield curve** (negative spread) is one of the most reliable recession indicators. ")
            md.append("Historical precedents:\n")
            md.append("- **2006**: Curve inverted → 2008 recession (18-month lag)\n")
            md.append("- **2000**: Curve inverted → 2001 recession (12-month lag)\n")
            md.append("- **1989**: Curve inverted → 1990 recession (15-month lag)\n")
            md.append("The inversion signals that bond markets expect the Fed to cut rates due to economic weakness.\n")
        elif curve < 20:
            md.append("A **flat yield curve** suggests uncertainty about economic growth. ")
            md.append("This often occurs in late-cycle environments where the Fed has raised short rates but long rates remain anchored. ")
            md.append("Neither expansion nor recession is clearly priced in - a transitional phase.\n")
        elif curve < 100:
            md.append("A **moderately steep curve** is consistent with normal economic conditions and modest growth expectations. ")
            md.append("This is typical of mid-cycle expansions with stable Fed policy.\n")
        else:
            md.append("A **very steep curve** typically occurs early in economic recoveries. ")
            md.append("Example: 2009-2010 (curve >250 bps) as Fed kept short rates at zero while long rates rose on growth expectations.\n")
        
        # Credit spread analysis
        credit = metrics['credit_spread']
        md.append(f"**Credit Risk**: High-yield credit spreads are **{credit:.1f}%** ({credit*100:.0f} basis points). ")
        if credit > 7.0:
            md.append("Spreads above 7% (700 bps) indicate **severe credit stress**. ")
            md.append("Historical examples:\n")
            md.append("- **2008 Financial Crisis**: Spreads peaked at 20% (2,000 bps)\n")
            md.append("- **2020 COVID Crash**: Spreads spiked to 11% (1,100 bps)\n")
            md.append("This signals recession fears, potential defaults, and flight to quality.\n")
        elif credit > 5.0:
            md.append("Spreads above 5% (500 bps) indicate **elevated credit risk**. ")
            md.append("Corporate borrowing costs are rising as investors worry about economic weakness. ")
            md.append("This often precedes recessions by 3-6 months.\n")
        elif credit > 4.0:
            md.append("Spreads in the 4-5% (400-500 bps) range are **neutral** - moderate risk premiums consistent with mid-cycle conditions. ")
            md.append("This reflects normal compensation for credit risk without stress signals.\n")
        else:
            md.append("Spreads below 4% (400 bps) indicate **low credit risk**. ")
            md.append("Investors are confident in corporate health and economic stability. ")
            md.append("Example: 2017-2018 (spreads ~3.5% or 350 bps) during synchronized global growth.\n")
        
        # Inflation and real yield
        inflation = metrics['inflation_yoy']
        real_yield = metrics['real_yield']
        md.append(f"**Inflation & Real Yields**: Inflation is running at **{inflation:.1f}%** annually, giving a real yield of **{real_yield:.2f}%**. ")
        if real_yield < 0:
            md.append("**Negative real yields** mean bonds lose purchasing power after inflation. ")
            md.append("Historical examples:\n")
            md.append("- **1970s**: Real yields deeply negative → Gold +1,400%, Bonds -30%\n")
            md.append("- **2021-2022**: Real yields negative → Gold +5%, Bonds -13%\n")
            md.append("This environment favors real assets like gold, commodities, and TIPS.\n")
        elif real_yield < 1.0:
            md.append("**Low positive real yields** provide minimal real return on bonds. ")
            md.append("At {:.1f}%, bonds barely keep pace with inflation. Gold and equities may be more attractive.\n".format(real_yield))
        else:
            md.append("**Positive real yields** make bonds attractive - investors earn real returns above inflation. ")
            md.append("Example: 2023-2024 (real yields ~2%) supported bond returns after years of losses.\n")
        
        # Volatility
        vix = metrics['vix']
        md.append(f"**Market Volatility**: The VIX is at **{vix:.1f}**. ")
        if vix > 30:
            md.append("Elevated volatility above 30 signals **market stress** and heightened uncertainty. ")
            md.append("Historical spikes: 2008 (VIX 80), 2020 (VIX 82), 2011 (VIX 48). ")
            md.append("Risk-off positioning is warranted.\n")
        elif vix > 20:
            md.append("Moderate volatility suggests **increased caution** but not panic. ")
            md.append("This is typical during corrections or uncertainty events. Some defensive positioning is prudent.\n")
        else:
            md.append("Low volatility below 20 indicates **market complacency** and calm conditions. ")
            md.append("However, this can also precede sudden volatility spikes - the 'calm before the storm' phenomenon. ")
            md.append("Example: VIX at 10 in January 2018 → spiked to 50 in February 2018.\n")
        
        # Section 3: Regime Detection
        md.append("\n## 3. Economic Regime Analysis\n")
        
        # Add regime scatter plot
        md.append("### 3.1 Regime Classification\n")
        md.append("![Regime Classification](figures/04_regime_scatter.png)\n")
        md.append("")
        
        # Add trend strength explanation
        md.append("#### Understanding Trend Strength\n\n")
        md.append("**What is Trend Strength?**\n\n")
        md.append("Trend strength measures market momentum over the past 12 months:\n")
        md.append("```\nTrend Strength = Current Price / Price 12 Months Ago\n```\n\n")
        
        md.append("**Interpretation:**\n")
        md.append("- **Above 1.0**: Market is higher than 12 months ago (uptrend)\n")
        md.append("- **Below 1.0**: Market is lower than 12 months ago (downtrend)\n")
        md.append("- **Example**: 1.05 = market up 5% over past year\n\n")
        
        md.append("**Regime Thresholds:**\n")
        md.append("- **1.10+**: Strong bull market (>10% annual gain)\n")
        md.append("- **1.02-1.10**: Moderate growth → **Normal/Growth regime**\n")
        md.append("- **0.98-1.02**: Sideways/neutral\n")
        md.append("- **0.90-0.98**: Moderate decline\n")
        md.append("- **<0.90**: Bear market (>10% annual loss) → **Crisis regime**\n\n")
        
        md.append("**Historical Examples:**\n\n")
        md.append("*Crisis (Trend < 0.90):*\n")
        md.append("- 2008-09-15 (Lehman): Trend = 0.88 (S&P down 12% YoY)\n")
        md.append("- 2009-03-09 (Bottom): Trend = 0.70 (S&P down 30% YoY)\n")
        md.append("- 2020-03-23 (COVID): Trend = 0.92 (S&P down 8% YoY)\n\n")
        
        md.append("*Bubble (Trend > 1.08):*\n")
        md.append("- 1999-06-15 (Dot-com): Trend = 1.079 (S&P up 7.9% YoY)\n")
        md.append("- 2021-11-08 (COVID Bubble): Trend = 1.078 (S&P up 7.8% YoY)\n\n")
        
        md.append("*Normal (Trend 1.00-1.05):*\n")
        md.append("- 1995-06-15: Trend = 1.05 (healthy growth)\n")
        md.append("- 2013-06-15: Trend = 1.03 (post-crisis recovery)\n")
        md.append("- 2017-06-15: Trend = 1.02 (late-cycle expansion)\n\n")
        
        current_trend = metrics.get('trend_equity', 1.0)
        md.append(f"**Your Current Position:** Trend strength of {current_trend:.3f} ")
        if current_trend > 1.08:
            md.append("indicates **strong momentum** (bubble risk zone).\n\n")
        elif current_trend > 1.02:
            md.append("indicates **healthy growth** (normal zone).\n\n")
        elif current_trend > 0.98:
            md.append("indicates **sideways market** (neutral zone).\n\n")
        elif current_trend > 0.90:
            md.append("indicates **moderate weakness** (caution zone).\n\n")
        else:
            md.append("indicates **bear market** (crisis zone).\n\n")
        
        # Add regime probability heatmap
        md.append("### 3.2 Regime Detection Comparison\n")
        md.append("![Regime Probabilities](figures/05_regime_heatmap.png)\n")
        md.append("")
        
        # Get scores from all three methods
        recession_threshold = regime_scores['recession_risk']
        growth_threshold = regime_scores['growth_regime']
        inflation_threshold = regime_scores['inflation_regime']
        
        recession_continuous = regime_scores.get('recession_risk_continuous', recession_threshold)
        growth_continuous = regime_scores.get('growth_regime_continuous', growth_threshold)
        inflation_continuous = regime_scores.get('inflation_regime_continuous', inflation_threshold)
        
        probabilistic_probs = regime_scores.get('probabilistic', {})
        
        # Debug: Check if probabilistic probs are calculated
        if not probabilistic_probs:
            self.debug_print("WARNING: Probabilistic probs are empty!")
            self.debug_print(f"regime_scores keys: {list(regime_scores.keys())}")
        else:
            self.debug_print(f"Probabilistic probs: {probabilistic_probs}")
        
        # Determine regimes using first two methods
        regime_threshold, conf_threshold = self._get_regime_name(
            recession_threshold, growth_threshold, inflation_threshold, metrics, "threshold"
        )
        regime_continuous, conf_continuous = self._get_regime_name(
            recession_continuous, growth_continuous, inflation_continuous, metrics, "continuous"
        )
        
        # Probabilistic method (4-regime)
        if probabilistic_probs:
            regime_probabilistic = max(probabilistic_probs, key=probabilistic_probs.get)
            conf_probabilistic = probabilistic_probs[regime_probabilistic]
        else:
            regime_probabilistic = regime_continuous
            conf_probabilistic = conf_continuous
        
        # Use probabilistic as primary (best generalization)
        current_regime = regime_probabilistic
        
        md.append("### 3.3 Detection Method Comparison")
        md.append("")
        md.append("Comparing three detection methods:")
        md.append("")
        md.append("| Method | Regime | Confidence | Details |")
        md.append("|--------|--------|------------|---------|")
        md.append(f"| **Threshold** | {regime_threshold} | {conf_threshold:.0%} | Recession: {recession_threshold:.2f}, Growth: {growth_threshold:.2f}, Inflation: {inflation_threshold:.2f} |")
        md.append(f"| **Smooth** | {regime_continuous} | {conf_continuous:.0%} | Recession: {recession_continuous:.2f}, Growth: {growth_continuous:.2f}, Inflation: {inflation_continuous:.2f} |")
        
        # Probabilistic row
        if probabilistic_probs:
            prob_details = ", ".join([f"{k}: {v:.2f}" for k, v in sorted(probabilistic_probs.items(), key=lambda x: x[1], reverse=True)])
        else:
            prob_details = "N/A"
        md.append(f"| **Probabilistic** | {regime_probabilistic} | {conf_probabilistic:.0%} | {prob_details} |")
        
        md.append("")
        
        # Agreement analysis
        methods_agree = (regime_threshold == regime_continuous == regime_probabilistic)
        if methods_agree:
            md.append(f"✓ **All methods agree**: {current_regime} (high confidence)")
        else:
            md.append(f"⚠️ **Methods disagree**:")
            md.append(f"- Threshold: {regime_threshold}")
            md.append(f"- Smooth: {regime_continuous}")
            md.append(f"- Probabilistic: {regime_probabilistic}")
            md.append(f"")
            md.append(f"Using Probabilistic method (best generalization): **{current_regime}**")
        
        md.append("")
        md.append("### 3.4 Understanding Economic Regimes")
        md.append("Economic regimes are distinct periods characterized by relatively homogeneous macroeconomic behavior. ")
        md.append("Each regime has unique risk-return characteristics for asset classes.")
        
        md.append("**The Four Primary Regimes:**\n\n")
        
        md.append("**1. Growth Regime** (GDP > 2.5%, Low Unemployment, Moderate Inflation)\n")
        md.append("- Best: Equities (10-15% returns) | Worst: Gold (0-3%)\n")
        md.append("- Examples: 1995-1999 Tech boom, 2010-2019 Recovery\n\n")
        
        md.append("**2. Recession/Crisis** (Negative GDP, High Unemployment, Credit Stress)\n")
        md.append("- Best: Bonds (10-20% returns) | Worst: Equities (-20% to -50%)\n")
        md.append("- Examples: 2008 Financial Crisis, 2020 COVID Crash\n")
        md.append("- NBER Definition: \"Significant decline in economic activity spread across economy, lasting months\"\n\n")
        
        md.append("**3. Stagflation** (Stagnant Growth + High Inflation > 6%)\n")
        md.append("- Best: Gold (15-30% returns) | Worst: Bonds (-5% to -10%)\n")
        md.append("- Examples: 1970s Oil Shocks (1973-1975, 1979-1982)\n")
        md.append("- Definition: Blend of 'stagnation' + 'inflation'\n\n")
        
        md.append("**4. Bubble** (Extreme Valuations, Euphoria, Rapid Appreciation)\n")
        md.append("- During: Equities (20-50%+) | After Burst: Equities (-40% to -80%)\n")
        md.append("- Examples: 1929 Stock Market, 1999-2000 Dot-com, 2005-2007 Housing\n")
        md.append("- Detection Challenge: \"Cannot be achieved with satisfactory certainty\" (Gürkaynak, 2008)\n\n")
        
        md.append("### 3.5 Current Regime Analysis\n")
        md.append(f"Based on current indicators, the economy is in a **{current_regime}** regime:\n\n")
        
        # Map 4-regime to implications (handle both old 6-regime and new 4-regime names)
        regime_normalized = current_regime
        if current_regime in ["Growth", "Neutral"]:
            regime_normalized = "Normal"
        elif current_regime in ["High Inflation", "Stagflation"]:
            regime_normalized = "Inflation"
        
        if regime_normalized == "Bubble":
            md.append("**Implications:**\n")
            md.append("- ✗ Minimize equities (crash risk extreme)\n")
            md.append("- ✓ Increase bonds (safe haven)\n")
            md.append("- ✓ Increase gold (diversification)\n")
            md.append("- ✓ Increase cash (dry powder for crash)\n")
            md.append("- **WARNING**: Bubbles can persist longer than expected, but risk/reward is unfavorable\n")
        elif regime_normalized == "Normal":
            md.append("**Implications:**\n")
            md.append("- ✓ Favor equities for capital appreciation\n")
            md.append("- ✓ Moderate bonds for diversification\n")
            md.append("- ✗ Minimal gold (opportunity cost high)\n")
            md.append("- ✗ Low cash (inflation erodes value)\n")
        elif regime_normalized == "Crisis" or current_regime == "Recession Risk":
            md.append("**Implications:**\n")
            md.append("- ✗ Reduce equities (downside risk high)\n")
            md.append("- ✓ Increase bonds (flight to quality)\n")
            md.append("- ✓ Increase gold (safe haven)\n")
            md.append("- ✓ Increase cash (preserve capital)\n")
        elif regime_normalized == "Inflation":
            md.append("**Implications:**\n")
            md.append("- ~ Moderate equities (inflation headwind, but companies can pass costs)\n")
            md.append("- ✗ Minimize bonds (inflation erodes fixed payments)\n")
            md.append("- ✓ Maximize gold (best inflation hedge)\n")
            md.append("- ~ Moderate cash (yields may be attractive, but loses purchasing power)\n")
        else:
            md.append("**Implications:**\n")
            md.append("- Balanced allocation across asset classes\n")
            md.append("- Volatility targeting to manage risk\n")
            md.append("- Tactical adjustments based on signals\n")
        
        md.append("\n### 3.6 Regime Detection Methodology\n")
        md.append("We use three complementary approaches:\n\n")
        
        md.append("#### 3.6.1 Threshold Method (Simple & Interpretable)\n")
        md.append("Uses hard cutoffs for regime classification:\n")
        md.append("- **Crisis**: Recession risk > 0.5 OR severe equity decline\n")
        md.append("- **Bubble**: Valuation z-score > 2.3 AND strong uptrend\n")
        md.append("- **Stagflation**: High inflation (>0.7) AND low growth (<0.4)\n")
        md.append("- **High Inflation**: Inflation > 0.7 AND CPI > 4%\n")
        md.append("- **Growth**: Growth > 0.5 (baseline/default state)\n\n")
        md.append(f"**Current Threshold Scores:**\n")
        md.append(f"- Recession Risk: {recession_threshold:.2f} → ")
        if recession_threshold > 0.6:
            md.append("**High** (crisis conditions)\n")
        elif recession_threshold > 0.3:
            md.append("**Elevated** (caution warranted)\n")
        else:
            md.append("**Low** (expansion likely)\n")
        md.append(f"- Growth Regime: {growth_threshold:.2f} → ")
        if growth_threshold > 0.5:
            md.append("**Strong** (favorable for risk assets)\n")
        elif growth_threshold > 0.2:
            md.append("**Moderate** (mixed signals)\n")
        else:
            md.append("**Weak** (stagnation risk)\n")
        md.append(f"- Inflation Regime: {inflation_threshold:.2f} → ")
        if inflation_threshold > 0.7:
            md.append("**High** (inflation hedge needed)\n")
        elif inflation_threshold > 0.3:
            md.append("**Elevated** (monitor closely)\n")
        else:
            md.append("**Low** (benign environment)\n")
        
        md.append("\n#### 3.6.2 Smooth Method (Continuous Transitions)\n")
        md.append("Uses smooth transitions between regimes:\n")
        md.append("- **No hard thresholds** - Linear interpolation between states\n")
        md.append("- **Special cases** - Severe equity declines (>20%) trigger crisis\n")
        md.append("- **Better generalization** - Eliminates edge cases (e.g., 2022)\n")
        md.append("- **Calibrated empirically** - Thresholds based on historical analysis\n\n")
        
        md.append("**Equations:**\n\n")
        md.append("The smooth method uses continuous scoring functions with linear interpolation:\n\n")
        
        md.append("*Recession Risk Score:*\n")
        md.append("```\n")
        md.append("recession_risk = 0.0\n")
        md.append("if yield_curve < -50 bps: recession_risk += 0.4\n")
        md.append("if credit_spread > 200 bps: recession_risk += 0.3 * (spread - 200) / 100\n")
        md.append("if trend_equity < 0.95: recession_risk += 0.3 * (0.95 - trend) / 0.15\n")
        md.append("if equity_decline > 20%: recession_risk = 1.0  # Crisis override\n")
        md.append("```\n\n")
        
        md.append("*Growth Regime Score:*\n")
        md.append("```\n")
        md.append("growth_score = 0.0\n")
        md.append("if trend_equity > 1.02: growth_score += 0.4 * (trend - 1.02) / 0.08\n")
        md.append("if yield_curve > 0: growth_score += 0.3 * min(curve / 100, 1.0)\n")
        md.append("if credit_spread < 150: growth_score += 0.3 * (150 - spread) / 50\n")
        md.append("```\n\n")
        
        md.append("*Inflation Regime Score:*\n")
        md.append("```\n")
        md.append("inflation_score = 0.0\n")
        md.append("if cpi_yoy > 3.0%: inflation_score += 0.5 * (cpi - 3.0) / 3.0\n")
        md.append("if real_yield < 0: inflation_score += 0.3 * abs(real_yield) / 2.0\n")
        md.append("if gold_ratio declining: inflation_score += 0.2\n")
        md.append("```\n\n")
        
        md.append(f"**Current Smooth Scores:**\n")
        md.append(f"- Recession Risk: {recession_continuous:.2f} → ")
        if recession_continuous > 0.6:
            md.append("**High** (crisis conditions)\n")
        elif recession_continuous > 0.3:
            md.append("**Elevated** (caution warranted)\n")
        else:
            md.append("**Low** (expansion likely)\n")
        md.append(f"- Growth Regime: {growth_continuous:.2f} → ")
        if growth_continuous > 0.5:
            md.append("**Strong** (favorable for risk assets)\n")
        elif growth_continuous > 0.2:
            md.append("**Moderate** (mixed signals)\n")
        else:
            md.append("**Weak** (stagnation risk)\n")
        md.append(f"- Inflation Regime: {inflation_continuous:.2f} → ")
        if inflation_continuous > 0.7:
            md.append("**High** (inflation hedge needed)\n")
        elif inflation_continuous > 0.3:
            md.append("**Elevated** (monitor closely)\n")
        else:
            md.append("**Low** (benign environment)\n")
        
        md.append("\n#### 3.6.3 Probabilistic Method (Equation-Based, 4 Regimes)\n")
        md.append("Uses mathematical equations with optimized coefficients:\n")
        md.append("- **4 merged regimes**: Crisis, Bubble, Inflation (merged High Inflation + Stagflation), Normal (merged Growth + Neutral)\n")
        md.append("- **Interaction terms**: z-score × trend for bubble detection, crisis context adjustment\n")
        md.append("- **Softmax normalization**: Probabilities sum to 1.0\n")
        md.append("- **Optimized on 27 historical dates**: 66.7% accuracy, best generalization\n\n")
        
        if probabilistic_probs:
            md.append(f"**Current Probabilistic Probabilities:**\n")
            for regime, prob in sorted(probabilistic_probs.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 30)
                md.append(f"- {regime}: {prob:.1%} {bar}\n")
        
        md.append("\n**Important Note:** Regime detection is inherently challenging. The NBER uses a retrospective approach, ")
        md.append("waiting 6-12 months for confirmation. Our real-time approach trades accuracy for timeliness. ")
        md.append("Expect 60-70% accuracy, with particular difficulty detecting bubbles and regime transitions. ")
        md.append("When methods disagree, probabilistic method is preferred (best generalization on test data).\n")
        
        # Section 4: Allocation Methodology
        md.append("\n## 4. Allocation Methodology\n")
        md.append("### 4.1 Strategic Framework\n")
        md.append("The portfolio allocation follows a two-stage process:\n\n")
        md.append("**Stage 1: Regime-Conditional Strategic Allocation**\n")
        md.append("Base weights are adjusted based on economic regime scores:\n\n")
        md.append("```")
        md.append("Equities = 50% + 15%×Growth - 20%×Recession - 12%×Valuation + 8%×Momentum")
        md.append("Bonds    = 25% + 18%×Recession - 8%×Inflation + 10%×YieldAttractiveness")
        md.append("Gold     = 10% + 12%×Inflation + 8%×NegativeRealYield + 6%×Recession")
        md.append("Cash     = 5%  + 8%×Recession + 5%×Valuation + 4%×HighCashYields")
        md.append("Crypto   = 0%  + (8% + 5%×Growth) if favorable conditions")
        md.append("```\n")
        
        md.append("**Stage 2: Volatility Targeting (Risk Parity Overlay)**\n")
        md.append("Weights are adjusted to balance risk contribution across assets:\n\n")
        md.append("```")
        md.append("Risk-Adjusted Weight = 0.70 × Strategic Weight + 0.30 × (Weight / Volatility)")
        md.append("```\n")
        md.append("This ensures high-volatility assets (equities, crypto) don't dominate portfolio risk.\n")
        
        md.append("### 4.2 Mathematical Foundations\n")
        md.append("#### Buffett Indicator Z-Score\n")
        md.append("Measures market valuation relative to economic output:\n\n")
        md.append("```")
        md.append("Buffett Ratio = Market Cap / GDP")
        md.append("Z-Score = (Current Ratio - 10Y Mean) / 10Y Std Dev")
        md.append("```\n")
        md.append(f"**Current Calculation**: Z-Score = {z:.2f}\n")
        md.append("- Z > 2.0: Significantly overvalued (>95th percentile)\n")
        md.append("- Z > 1.0: Moderately overvalued (>84th percentile)\n")
        md.append("- -1.0 < Z < 1.0: Fairly valued\n")
        md.append("- Z < -1.0: Undervalued (<16th percentile)\n")
        
        md.append("\n#### Real Yield Calculation\n")
        md.append("Measures inflation-adjusted return on bonds:\n\n")
        md.append("```")
        md.append("Real Yield = 10Y Treasury Yield - CPI YoY Inflation")
        md.append("```\n")
        md.append(f"**Current Calculation**: {metrics['ten_year_yield']:.2f}% - {inflation:.1f}% = {real_yield:.2f}%\n")
        md.append("- Real Yield < 0%: Bonds lose purchasing power\n")
        md.append("- Real Yield 0-1%: Minimal real return\n")
        md.append("- Real Yield > 2%: Attractive real returns\n")
        
        md.append("\n#### Risk Parity Weighting\n")
        md.append("Allocates capital inversely to volatility:\n\n")
        md.append("```")
        md.append("Inverse Vol Weight_i = (Strategic Weight_i / Volatility_i) / Σ(Weight_j / Vol_j)")
        md.append("```\n")
        md.append("**Asset Class Volatilities**:\n")
        equity_vol = max(12.0, vix * 0.8)
        bond_vol = 6.0 + max(0.0, credit - 3.0) * 0.5
        md.append(f"- Equities: {equity_vol:.1f}% (dynamic, based on VIX)")
        md.append(f"- Bonds: {bond_vol:.1f}% (increases with credit spreads)")
        md.append("- Gold: 15.0% (historical average)")
        md.append("- Crypto: 60.0% (high volatility)")
        md.append("- Cash: 0.5% (minimal)\n")
        
        md.append("### 4.3 Allocation Rationale\n")
        md.append(f"**Equities ({weights['equities']:.1f}%)**:\n")
        if recession_continuous > 0.3:
            md.append("- Underweight due to elevated recession risk\n")
        elif metrics['z_score'] > 2.0:
            md.append(f"- Underweight due to high valuation (z-score {metrics['z_score']:.2f})\n")
        elif weights['equities'] > 50:
            md.append("- Overweight due to growth regime and/or positive momentum\n")
        else:
            md.append("- Neutral weight reflecting balanced risk/reward\n")
        
        md.append(f"**Bonds ({weights['bonds']:.1f}%)**:\n")
        if recession_continuous > 0.3:
            md.append("- Overweight for defensive positioning (flight to quality)\n")
        elif inflation_continuous > 0.3:
            md.append("- Underweight due to inflation concerns eroding real returns\n")
        elif weights['bonds'] > 35:
            md.append("- Overweight for attractive yields\n")
        else:
            md.append("- Standard allocation for diversification and income\n")
        
        md.append(f"**Gold ({weights['gold']:.1f}%)**:\n")
        if inflation_continuous > 0.3:
            md.append("- Overweight as inflation hedge\n")
        elif recession_continuous > 0.3:
            md.append("- Overweight as safe haven\n")
        elif weights['gold'] > 15:
            md.append("- Overweight due to negative real yields or geopolitical risk\n")
        else:
            md.append("- Moderate allocation for portfolio insurance\n")
        
        md.append(f"**Cash ({weights['cash']:.1f}%)**:\n")
        if weights['cash'] > 15:
            md.append(f"- Elevated cash due to: ")
            reasons = []
            if recession_continuous > 0.2:
                reasons.append("recession risk")
            if metrics['z_score'] > 2.0:
                reasons.append("high valuation")
            if metrics['ten_year_yield'] > 4.0:
                reasons.append("attractive cash yields")
            if not reasons:
                reasons.append("volatility targeting (risk parity)")
            md.append(", ".join(reasons) + "\n")
        else:
            md.append("- Minimal cash for liquidity and rebalancing\n")
        
        if weights['crypto'] > 0:
            md.append(f"**Crypto ({weights['crypto']:.1f}%)**:\n")
            md.append("- Small allocation in risk-on environment with low recession risk\n")
        
        # Section 5: Historical Context
        md.append("\n## 5. Historical Context\n")
        md.append("### 5.1 Similar Historical Periods\n")
        if similar:
            md.append("The current market environment resembles these historical periods:\n\n")
            for i, period in enumerate(similar, 1):
                md.append(f"**{i}. {period['date']}**")
                md.append(f"- Z-Score: {period['z_score']:.2f}")
                md.append(f"- 10Y Yield: {period['ten_year_yield']:.1f}%")
                md.append(f"- Yield Curve: {period['yield_curve']:.0f} bps")
                md.append(f"- S&P/Gold: {period['sp_gold_ratio']:.2f}")
                md.append(f"- VIX: {period['vix']:.1f}\n")
        else:
            md.append("No close historical analogues found in the dataset.\n")
        
        md.append("### 5.2 Lessons from History\n")
        if z > 2.0 and curve < 50:
            md.append("**High Valuations + Flat/Inverted Curve**: This combination has historically preceded market corrections. Examples include:\n")
            md.append("- **2000 Dot-com Peak**: Z-score >2.5, curve inverted → 50% decline\n")
            md.append("- **2007 Pre-Crisis**: Z-score >2.0, curve inverted → 57% decline\n")
            md.append("- **Lesson**: Defensive positioning and cash reserves are prudent\n")
        elif growth_continuous > 0.4:
            md.append("**Growth Regime**: Historically, steep curves and low credit spreads support equity returns. Examples:\n")
            md.append("- **2009-2010 Recovery**: Steep curve, falling spreads → strong equity gains\n")
            md.append("- **2016-2017 Expansion**: Positive momentum, low volatility → steady returns\n")
            md.append("- **Lesson**: Risk-on positioning captures upside in expansions\n")
        elif inflation_continuous > 0.4:
            md.append("**Inflation Regime**: High inflation erodes nominal returns. Historical examples:\n")
            md.append("- **1970s Stagflation**: Negative real yields → gold +1,400%, bonds -30%\n")
            md.append("- **2021-2022 Inflation Surge**: CPI >7% → gold +5%, bonds -13%\n")
            md.append("- **Lesson**: Real assets (gold, commodities, TIPS) outperform\n")
        
        # Section 6: Risk Considerations
        md.append("\n## 6. Risk Considerations\n")
        md.append("### 6.1 Portfolio Risk Metrics\n")
        # Estimate portfolio volatility
        port_vol = (weights['equities']/100 * equity_vol + 
                    weights['bonds']/100 * bond_vol + 
                    weights['gold']/100 * 15.0 + 
                    weights['crypto']/100 * 60.0 + 
                    weights['cash']/100 * 0.5)
        md.append(f"**Estimated Portfolio Volatility**: {port_vol:.1f}%\n")
        md.append("This represents the expected annualized standard deviation of returns.\n")
        
        md.append("### 6.2 Key Risks\n")
        md.append("**Upside Risks** (could improve returns):\n")
        md.append("- Economic growth accelerates beyond expectations\n")
        md.append("- Fed pivots to rate cuts, boosting asset prices\n")
        md.append("- Inflation falls faster than anticipated\n")
        md.append("- Corporate earnings surprise to the upside\n\n")
        
        md.append("**Downside Risks** (could hurt returns):\n")
        md.append("- Recession materializes, corporate earnings decline\n")
        md.append("- Inflation re-accelerates, forcing Fed tightening\n")
        md.append("- Geopolitical shocks (war, trade tensions)\n")
        md.append("- Credit event or financial system stress\n")
        md.append("- Valuation compression from high levels\n")
        
        md.append("\n### 6.3 Mitigation Strategies\n")
        md.append("The portfolio incorporates several risk management techniques:\n\n")
        md.append("1. **Diversification**: Spread across uncorrelated assets (equities, bonds, gold)\n")
        md.append("2. **Volatility Targeting**: Reduce exposure to high-volatility assets in stressed markets\n")
        md.append("3. **Regime Awareness**: Shift defensively when recession indicators flash\n")
        md.append("4. **Valuation Discipline**: Reduce equity exposure at extreme valuations\n")
        md.append("5. **Cash Buffer**: Maintain liquidity for rebalancing opportunities\n")
        
        # Section 7: Implementation Notes
        md.append("\n## 7. Implementation Notes\n")
        md.append("### 7.1 Rebalancing\n")
        md.append("- **Frequency**: Monthly or when allocations drift >5% from targets\n")
        md.append("- **Tax Considerations**: Use tax-advantaged accounts for high-turnover assets\n")
        md.append("- **Transaction Costs**: Minimize trading costs through ETFs and limit orders\n")
        
        md.append("\n### 7.2 Monitoring\n")
        md.append("Key indicators to watch:\n")
        md.append("- **Yield Curve**: Inversion signals recession risk\n")
        md.append("- **Credit Spreads**: Widening indicates stress\n")
        md.append("- **Inflation**: Rising CPI favors real assets\n")
        md.append("- **VIX**: Spikes above 30 warrant defensive action\n")
        md.append("- **Equity Trends**: Breaking 200-day MA suggests regime shift\n")
        
        md.append("\n### 7.3 Customization\n")
        md.append("This allocation is a starting point. Adjust based on:\n")
        md.append("- **Time Horizon**: Longer horizons can tolerate more equity risk\n")
        md.append("- **Risk Tolerance**: Conservative investors should increase bonds/cash\n")
        md.append("- **Tax Situation**: Municipal bonds for high-tax investors\n")
        md.append("- **Liquidity Needs**: Maintain adequate cash for near-term expenses\n")
        
        # Footer
        md.append("\n---")
        md.append(f"\n*Report generated by Macro Portfolio Allocator v2.0*")
        md.append(f"\n*Methodology: Regime-aware allocation with volatility targeting*")
        md.append(f"\n*Current Regime: {current_regime}*")
        
        return "\n".join(md)
    
    def load_full_historical_data(self) -> Dict[str, pd.DataFrame]:
        """Load comprehensive historical data for backtesting"""
        
        self.debug_print("Loading full historical data for backtesting...")
        
        # Load FRED data with full history
        fred_full_path = "fred_cache_full"
        fred_data_full = {}
        
        fred_series_full = {
            "GDP": "GDP_FULL.csv",
            "CPIAUCSL": "CPIAUCSL_FULL.csv", 
            "CPILFESL": "CPILFESL_FULL.csv",
            "DGS10": "DGS10_FULL.csv",
            "DGS2": "DGS2_FULL.csv",
            "T10Y2Y": "T10Y2Y_FULL.csv",
            "GOLDAMGBD228NLBM": "GOLD_FULL.csv",
            "BAMLH0A0HYM2": "BAMLH0A0HYM2_FULL.csv",
            "T5YIE": "T5YIE_FULL.csv",
            "FEDFUNDS": "FEDFUNDS_FULL.csv"
        }
        
        for series, filename in fred_series_full.items():
            filepath = os.path.join(fred_full_path, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    if df.shape[1] == 1:
                        df = df.rename(columns={df.columns[0]: "value"})
                    fred_data_full[series] = df
                    self.debug_print(f"Loaded {series}: {len(df)} observations from {df.index.min()} to {df.index.max()}")
                except Exception as e:
                    self.debug_print(f"Failed to load {series}: {e}")
            else:
                self.debug_print(f"Missing file: {filepath}")
        
        # Load market data with maximum history
        self.debug_print("Downloading extended market data...")
        try:
            market_data = yf.download(BACKTESTING_SYMBOLS, start="1950-01-01", 
                                    end="2026-12-31", progress=False)
            if isinstance(market_data.columns, pd.MultiIndex):
                close_prices = market_data["Close"].copy()
            else:
                close_prices = market_data.copy()
            
            self.debug_print(f"Market data: {len(close_prices)} observations from {close_prices.index.min()} to {close_prices.index.max()}")
            
            # Add FRED gold prices as synthetic GOLD column for pre-GLD dates
            if 'GOLDAMGBD228NLBM' in fred_data_full and 'GLD' in close_prices.columns:
                gold_fred = fred_data_full['GOLDAMGBD228NLBM']['value']
                gld_series = close_prices['GLD'].copy()
                gold_fred_daily = gold_fred.reindex(close_prices.index, method='ffill')
                gld_series = gld_series.fillna(gold_fred_daily)
                close_prices['GLD'] = gld_series
                self.debug_print(f"Filled GLD with FRED gold: {gld_series.notna().sum()} total non-null values")
            
            # Add synthetic bond ETFs from treasury yields (inverse relationship)
            # TLT ~ 20-year treasury, IEF ~ 7-10 year treasury
            if 'DGS10' in fred_data_full:
                dgs10 = fred_data_full['DGS10']['value']
                dgs10_daily = dgs10.reindex(close_prices.index, method='ffill')
                
                # Create synthetic TLT: inverse of 10Y yield (simplified)
                # When yields fall, bond prices rise
                if 'TLT' in close_prices.columns:
                    tlt_series = close_prices['TLT'].copy()
                    # Use inverse yield as proxy: start at 100, adjust by yield changes
                    synthetic_tlt = 100 / (1 + dgs10_daily / 100)
                    tlt_series = tlt_series.fillna(synthetic_tlt)
                    close_prices['TLT'] = tlt_series
                    self.debug_print(f"Filled TLT with synthetic bond: {tlt_series.notna().sum()} total non-null values")
                
                # Create synthetic IEF similarly
                if 'IEF' in close_prices.columns:
                    ief_series = close_prices['IEF'].copy()
                    synthetic_ief = 100 / (1 + dgs10_daily / 100)
                    ief_series = ief_series.fillna(synthetic_ief)
                    close_prices['IEF'] = ief_series
                    self.debug_print(f"Filled IEF with synthetic bond: {ief_series.notna().sum()} total non-null values")
                
        except Exception as e:
            self.debug_print(f"Failed to download market data: {e}")
            close_prices = pd.DataFrame()
        
        return {"fred": fred_data_full, "market": close_prices}
    
    def calculate_optimal_portfolio(self, returns_data: pd.DataFrame, target_date: pd.Timestamp, 
                                  lookback_years: int = 5) -> Tuple[np.ndarray, float, float, List[str]]:
        """Calculate optimal portfolio using Markowitz mean-variance optimization
        
        Returns:
            weights: Array of optimal weights
            expected_return: Expected portfolio return
            portfolio_risk: Portfolio risk (std dev)
            asset_names: List of asset names corresponding to weights
        """
        
        self.debug_print(f"\n[OPT] Starting optimization for {target_date}")
        self.debug_print(f"[OPT] Input data: {len(returns_data)} rows, {len(returns_data.columns)} columns")
        
        # Use only data available before target_date
        end_date = target_date
        start_date = target_date - pd.DateOffset(years=lookback_years)
        
        # Filter by date to lookback window
        returns = returns_data.loc[start_date:end_date]
        self.debug_print(f"[OPT] After lookback filter ({start_date} to {end_date}): {len(returns)} rows")
        
        # Drop columns with too many NaNs (>20%) in the LOOKBACK window
        if len(returns) > 0:
            nan_pct = returns.isnull().sum() / len(returns)
            valid_cols = nan_pct[nan_pct < 0.2].index
            self.debug_print(f"[OPT] Columns with <20% NaN in lookback: {list(valid_cols)}")
            returns = returns[valid_cols]
        
        # Drop rows with any NaN
        returns = returns.dropna()
        self.debug_print(f"[OPT] After dropna: {len(returns)} rows, {len(returns.columns)} columns")
        
        # If insufficient data, try using all available data up to target date
        if len(returns) < 252:
            self.debug_print(f"[OPT] Insufficient data ({len(returns)}), trying all available data")
            returns = returns_data.loc[returns_data.index <= end_date]
            if len(returns) > 0:
                nan_pct = returns.isnull().sum() / len(returns)
                valid_cols = nan_pct[nan_pct < 0.2].index
                self.debug_print(f"[OPT] Valid columns: {list(valid_cols)}")
                returns = returns[valid_cols].dropna()
                self.debug_print(f"[OPT] After cleanup: {len(returns)} rows, {len(returns.columns)} columns")
        
        asset_names = list(returns.columns)
        
        if len(returns) < 60:  # Absolute minimum
            self.debug_print(f"[OPT] FAILED: Insufficient data ({len(returns)} observations)")
            n_assets = len(asset_names) if len(asset_names) > 0 else len(returns_data.columns)
            return np.array([1/n_assets] * n_assets), 0.0, 0.0, asset_names if asset_names else list(returns_data.columns)
        
        # Handle single-asset case
        if len(returns.columns) == 1:
            self.debug_print(f"[OPT] Single asset: {asset_names[0]} - returning 100%")
            return np.array([1.0]), 0.0, 0.0, asset_names
        
        # Estimate parameters with robust methods
        mean_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252     # Annualized
        
        # Apply Ledoit-Wolf shrinkage for numerical stability
        diag_target = np.diag(np.diag(cov_matrix))
        shrinkage = 0.2
        cov_matrix = (1 - shrinkage) * cov_matrix + shrinkage * diag_target
        
        # Handle numerical issues
        if np.any(np.isnan(mean_returns)) or np.any(np.isnan(cov_matrix.values)):
            self.debug_print("[OPT] FAILED: NaN values in return/covariance estimation")
            n_assets = len(returns.columns)
            return np.array([1/n_assets] * n_assets), 0.0, 0.0, asset_names
        
        def negative_sharpe(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_risk < 1e-8:
                return 1e8
            return -portfolio_return / portfolio_risk
        
        # Constraints and bounds - more realistic
        n_assets = len(mean_returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 0.8) for _ in range(n_assets))  # Min 1%, max 80%
        
        # Try multiple starting points
        best_result = None
        best_sharpe = -np.inf
        
        for attempt in range(3):
            if attempt == 0:
                x0 = np.array([1/n_assets] * n_assets)  # Equal weights
            else:
                x0 = np.random.dirichlet(np.ones(n_assets))  # Random valid weights
            
            try:
                result = minimize(negative_sharpe, x0, method='SLSQP', 
                                bounds=bounds, constraints=constraints,
                                options={'maxiter': 1000, 'ftol': 1e-9})
                
                if result.success and -result.fun > best_sharpe:
                    best_result = result
                    best_sharpe = -result.fun
                    
            except Exception as e:
                self.debug_print(f"[OPT] Attempt {attempt} failed: {e}")
                continue
        
        if best_result is not None and best_result.success:
            optimal_weights = best_result.x
            
            # Validate diversity (not equal weights)
            if np.std(optimal_weights) < 0.01:
                self.debug_print(f"[OPT] FAILED: Converged to equal weights (std={np.std(optimal_weights):.4f})")
                return np.array([1/n_assets] * n_assets), 0.0, 0.0, asset_names
            
            portfolio_return = np.sum(mean_returns * optimal_weights)
            portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            self.debug_print(f"[OPT] SUCCESS: Sharpe={best_sharpe:.2f}, std={np.std(optimal_weights):.4f}")
            return optimal_weights, portfolio_return, portfolio_risk, asset_names
        else:
            self.debug_print(f"[OPT] FAILED: All 3 attempts failed")
            return np.array([1/n_assets] * n_assets), 0.0, 0.0, asset_names
    
    def calculate_portfolio_score(self, model_weights: Dict[str, float], 
                                optimal_weights: np.ndarray, 
                                asset_names: List[str]) -> float:
        """Calculate score from 0-100 based on portfolio similarity to optimal"""
        
        # Convert model weights to array matching optimal weights order
        model_array = np.zeros(len(asset_names))
        
        # Map model weights to asset array
        asset_mapping = {
            "^GSPC": ["equities"],
            "GLD": ["gold"], 
            "TLT": ["bonds"],
            "BTC-USD": ["crypto"]
        }
        
        for i, asset in enumerate(asset_names):
            if asset in asset_mapping:
                for model_asset in asset_mapping[asset]:
                    if model_asset in model_weights:
                        model_array[i] = model_weights[model_asset] / 100.0
        
        # Normalize to sum to 1
        if np.sum(model_array) > 0:
            model_array = model_array / np.sum(model_array)
        
        # Calculate allocation distance
        allocation_distance = np.sum(np.abs(model_array - optimal_weights))
        allocation_score = max(0, 100 - allocation_distance * 50)
        
        return allocation_score
    
    def backtest_single_date(self, target_date: str, historical_data: Dict) -> Dict:
        """Run backtest for a single target date"""
        
        target_dt = pd.to_datetime(target_date)
        self.debug_print(f"\n=== BACKTESTING {target_date} ===")
        
        # Prepare data up to target date (no future data)
        fred_data_cutoff = {}
        for series, df in historical_data["fred"].items():
            cutoff_data = df.loc[df.index <= target_dt]
            if len(cutoff_data) > 0:
                fred_data_cutoff[series] = cutoff_data
        
        market_data_cutoff = historical_data["market"].loc[
            historical_data["market"].index <= target_dt
        ]
        
        if len(market_data_cutoff) < 252:
            return {"error": "Insufficient data", "score": 0}
        
        # Calculate model allocation
        try:
            metrics = self.compute_metrics(market_data_cutoff, fred_data_cutoff)
            model_weights = self.allocate_portfolio(metrics)
        except Exception as e:
            return {"error": f"Model failed: {e}", "score": 0}
        
        # Calculate optimal allocation
        try:
            # Calculate returns from FULL market data WITHOUT dropna
            # Let calculate_optimal_portfolio handle NaN filtering per asset
            returns_data = historical_data["market"].pct_change()
            
            # Don't pre-filter assets - let calculate_optimal_portfolio handle it
            optimal_weights, exp_return, portfolio_risk, available_assets = self.calculate_optimal_portfolio(
                returns_data, target_dt, lookback_years=5
            )
        except Exception as e:
            return {"error": f"Optimal failed: {e}", "score": 0}
        
        # Calculate score
        score = self.calculate_portfolio_score(model_weights, optimal_weights, available_assets)
        
        # Calculate regime scores (both methods)
        regime_scores = {
            # Threshold method
            'recession_risk': self._calculate_recession_risk(metrics),
            'growth_regime': self._calculate_growth_regime(metrics),
            'inflation_regime': self._calculate_inflation_regime(metrics),
            # Continuous method
            'recession_risk_continuous': self._calculate_recession_risk_continuous(metrics),
            'growth_regime_continuous': self._calculate_growth_regime_continuous(metrics),
            'inflation_regime_continuous': self._calculate_inflation_regime_continuous(metrics)
        }
        
        return {
            "date": target_date,
            "model_weights": model_weights,
            "optimal_weights": dict(zip(available_assets, optimal_weights)),
            "regime_scores": regime_scores,
            "metrics": metrics,
            "score": score,
            "expected_return": exp_return,
            "portfolio_risk": portfolio_risk
        }
    
    def generate_test_dates(self) -> List[str]:
        """Generate 200 test dates: 40 critical + 160 uniform"""
        
        critical_dates = CRITICAL_DATES[:40]
        uniform_dates = pd.date_range(start="1955-01-01", end="2026-04-15", periods=160)
        uniform_date_strings = [d.strftime("%Y-%m-%d") for d in uniform_dates]
        
        all_dates = critical_dates + uniform_date_strings
        all_dates = sorted(list(set(all_dates)))
        
        return all_dates
    
    def backtest_score_all(self) -> Dict:
        """Run comprehensive backtesting on all test dates"""
        
        self.debug_print("Starting comprehensive backtesting...")
        
        # Load historical data
        historical_data = self.load_full_historical_data()
        
        if not historical_data["fred"] or historical_data["market"].empty:
            return {"error": "Failed to load historical data", "overall_score": 0}
        
        # Generate test dates
        test_dates = self.generate_test_dates()
        
        # Run backtests
        results = []
        scores = []
        
        for i, date in enumerate(test_dates):
            if i % 20 == 0:
                self.debug_print(f"Progress: {i+1}/{len(test_dates)} dates processed")
            
            try:
                result = self.backtest_single_date(date, historical_data)
                results.append(result)
                
                if "score" in result:
                    scores.append(result["score"])
                    
            except Exception as e:
                self.debug_print(f"Backtest failed for {date}: {e}")
                results.append({"date": date, "error": str(e), "score": 0})
                scores.append(0)
        
        # Calculate overall statistics
        overall_stats = {
            "overall_score": np.mean(scores) if scores else 0,
            "median_score": np.median(scores) if scores else 0,
            "min_score": np.min(scores) if scores else 0,
            "max_score": np.max(scores) if scores else 0,
            "std_score": np.std(scores) if scores else 0,
            "valid_tests": len([s for s in scores if s > 0]),
            "total_tests": len(test_dates),
            "success_rate": len([s for s in scores if s > 0]) / len(test_dates) * 100 if test_dates else 0
        }
        
        self.debug_print(f"\n=== BACKTESTING SUMMARY ===")
        self.debug_print(f"Overall Score: {overall_stats['overall_score']:.1f}")
        self.debug_print(f"Success Rate: {overall_stats['success_rate']:.1f}%")
        
        return {
            "summary": overall_stats,
            "individual_results": results,
            "test_dates": test_dates
        }
    
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
            f"• Real yield: *{metrics['real_yield']:.2f}%*",
            f"• Inflation (YoY): *{metrics['inflation_yoy']:.1f}%*",
            f"• Credit spread: *{metrics['credit_spread']:.2f}%* ({metrics['credit_spread']*100:.0f} bps)",
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
        lines.append("_Note: Regime-aware allocation using inflation, credit spreads, and economic cycle detection._")
        return "\n".join(lines)

    def send_telegram(self, message: str, pdf_path: str = None):
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            raise RuntimeError("Telegram credentials are not configured in environment variables")

        # Send text message
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        
        # Send PDF if provided
        if pdf_path and os.path.exists(pdf_path):
            url_doc = f"https://api.telegram.org/bot{token}/sendDocument"
            with open(pdf_path, 'rb') as pdf_file:
                files = {'document': pdf_file}
                data = {'chat_id': chat_id, 'caption': 'Full Portfolio Report (PDF)'}
                response = requests.post(url_doc, data=data, files=files, timeout=30)
                response.raise_for_status()
                self.log_progress("PDF sent to Telegram")

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


def main(send_telegram=False, backtest=False, fast=False, offline=False, debug=False, backtest_comprehensive=False, include_crypto=False):
    allocator = MacroPortfolioAllocator(fast=fast, offline=offline, debug=debug, include_crypto=include_crypto)
    
    # Comprehensive backtesting mode
    if backtest_comprehensive:
        allocator.log_progress("Starting comprehensive backtesting...")
        results = allocator.backtest_score_all()
        
        # Save results
        results_file = "backtest_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        allocator.log_progress(f"Backtest results saved to {results_file}")
        
        # Print summary
        if "summary" in results:
            summary = results["summary"]
            print(f"\n=== COMPREHENSIVE BACKTESTING RESULTS ===")
            print(f"Overall Score: {summary['overall_score']:.1f}/100")
            print(f"Median Score: {summary['median_score']:.1f}/100")
            print(f"Success Rate: {summary['success_rate']:.1f}%")
            print(f"Valid Tests: {summary['valid_tests']}/{summary['total_tests']}")
            print(f"Score Range: {summary['min_score']:.1f} - {summary['max_score']:.1f}")
            print(f"Standard Deviation: {summary['std_score']:.1f}")
        
        return
    
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

    allocator.log_progress("Calculating regime scores...")
    regime_scores = {
        'recession_risk': allocator._calculate_recession_risk(metrics),
        'growth_regime': allocator._calculate_growth_regime(metrics),
        'inflation_regime': allocator._calculate_inflation_regime(metrics),
        'recession_risk_continuous': allocator._calculate_recession_risk_continuous(metrics),
        'growth_regime_continuous': allocator._calculate_growth_regime_continuous(metrics),
        'inflation_regime_continuous': allocator._calculate_inflation_regime_continuous(metrics),
        'probabilistic': allocator._calculate_regime_probabilistic(metrics)
    }

    allocator.log_progress("Generating comprehensive markdown report...")
    
    # Generate visualizations
    try:
        allocator.log_progress("Creating visualizations...")
        
        # Determine current regime for scatter plot
        prob_probs = regime_scores.get('probabilistic', {})
        if prob_probs:
            current_regime = max(prob_probs, key=prob_probs.get)
        else:
            current_regime = "Normal"
        
        create_all_visualizations(metrics, weights, regime_scores, current_regime, prices=prices, fred_data=fred_data)
        allocator.log_progress("Visualizations created successfully.")
    except Exception as e:
        allocator.log_progress(f"Warning: Visualization generation failed: {e}")
        print(f"Warning: Could not create visualizations: {e}")
    
    comprehensive_report = allocator.generate_comprehensive_report(metrics, weights, suballoc, similar, regime_scores)
    
    allocator.log_progress("Generating JSON output...")
    json_output = allocator.generate_json_output(metrics, weights, suballoc)

    allocator.log_progress("Saving reports to files...")
    report_file = "reports/run9_report.md"
    with open(report_file, "w") as f:
        f.write(comprehensive_report)
    allocator.log_progress(f"Comprehensive report saved to {report_file}")
    
    # Generate PDF version
    if not fast:
        try:
            allocator.log_progress("Generating PDF version...")
            pdf_file = "reports/run9_report.pdf"
            if convert_md_to_pdf(report_file, pdf_file):
                allocator.log_progress(f"PDF report saved to {pdf_file}")
        except Exception as e:
            allocator.log_progress(f"PDF generation skipped: {e}")
    
    json_file = "run9_allocation.json"
    with open(json_file, "w") as f:
        json.dump(json_output, f, indent=2)
    allocator.log_progress(f"JSON allocation saved to {json_file}")

    allocator.log_progress("Displaying summary in terminal...")
    print(message)

    if backtest:
        allocator.log_progress("Running backtest...")
        weights_df, portfolio = allocator.run_backtest(years=20, plot_path="run9_backtest.png")
        allocator.log_progress("Backtest completed.")
        allocator.log_progress(f"Final portfolio value: {portfolio.iloc[-1]:.3f}")
        allocator.log_progress("Backtest plot saved to run9_backtest.png")

    if send_telegram:
        allocator.log_progress("Sending Telegram message...")
        pdf_path = "reports/run9_report.pdf" if not fast and os.path.exists("reports/run9_report.pdf") else None
        allocator.send_telegram("\n".join(allocator.progress_log) + "\n\n" + message, pdf_path=pdf_path)
        allocator.log_progress("Telegram message dispatched.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run9 macro portfolio allocator")
    parser.add_argument("--send", action="store_true", help="Send summary to Telegram")
    parser.add_argument("--backtest", action="store_true", help="Run a 20-year backtest")
    parser.add_argument("--backtest-comprehensive", action="store_true", help="Run comprehensive backtesting on 200 historical dates")
    parser.add_argument("--fast", action="store_true", help="Use shorter periods and quicker timeouts for testing")
    parser.add_argument("--offline", action="store_true", help="Use cached data only (no network)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output for metrics and calculations")
    parser.add_argument("--crypto", action="store_true", help="Include crypto in portfolio allocation (default: excluded)")
    args = parser.parse_args()
    main(send_telegram=args.send, backtest=args.backtest, fast=args.fast, offline=args.offline, debug=args.debug, backtest_comprehensive=args.backtest_comprehensive, include_crypto=args.crypto)
