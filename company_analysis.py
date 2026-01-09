import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class ValueInvestingAnalyzer:
    """
    Value Investing Analyzer
    Automatically fetch data, calculate factors, generate analysis report
    """
    
    def __init__(self, ticker, benchmark_ticker='^GSPC', period='10y', output_dir='reports'):
        """
        Initialize analyzer
        """
        self.ticker = ticker
        self.benchmark_ticker = benchmark_ticker
        self.period = period
        self.output_dir = output_dir
        self.stock = None
        self.historical_data = None
        self.financials = None
        self.balance_sheet = None
        self.cashflow = None
        self.benchmark_data = None
        self.analysis_results = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def fetch_data(self):
        """Fetch and align all necessary data"""
        print(f"Fetching data for {self.ticker}...")
        
        # Get stock object
        self.stock = yf.Ticker(self.ticker)
        
        # Get historical price data
        self.historical_data = self.stock.history(period=self.period)
        
        # Get financial statements
        try:
            self.financials = self.stock.financials.T
            self.balance_sheet = self.stock.balance_sheet.T
            self.cashflow = self.stock.cashflow.T
        except Exception as e:
            print(f"Financial statements warning: {e}")
            # Create empty DataFrames to avoid errors
            self.financials = pd.DataFrame()
            self.balance_sheet = pd.DataFrame()
            self.cashflow = pd.DataFrame()
            
        # Get benchmark data
        try:
            self.benchmark_data = yf.Ticker(self.benchmark_ticker).history(period=self.period)
        except Exception as e:
            print(f"Benchmark data warning: {e}")
            self.benchmark_data = pd.DataFrame()
        
        print("Data fetching completed!")
        print(f"Stock data shape: {self.historical_data.shape}")
        print(f"Benchmark data shape: {self.benchmark_data.shape}")
        return self
    
    def calculate_valuation_factors(self):
        """Calculate basic valuation factors"""
        print("Calculating valuation factors...")
        
        factors = {}
        
        try:
            # Get valuation data from info
            info = self.stock.info
            
            # Current price
            current_price = self.historical_data['Close'].iloc[-1] if not self.historical_data.empty else np.nan
            factors['current_price'] = current_price
            
            # P/E Ratio
            factors['pe_ratio'] = info.get('trailingPE', info.get('forwardPE', np.nan))
            
            # P/B Ratio
            factors['pb_ratio'] = info.get('priceToBook', np.nan)
            
            # P/S Ratio
            factors['ps_ratio'] = info.get('priceToSalesTrailing12Months', np.nan)
            
            # Dividend Yield
            dividend_yield = info.get('dividendYield', 0)
            factors['dividend_yield'] = dividend_yield * 100 if dividend_yield else 0
            
            # EV/EBITDA
            factors['ev_to_ebitda'] = info.get('enterpriseToEbitda', np.nan)
            
            # Market Cap
            factors['market_cap'] = info.get('marketCap', np.nan)
            
            print(f"Valuation factors calculated: P/E={factors.get('pe_ratio', 'N/A'):.2f}, "
                  f"P/B={factors.get('pb_ratio', 'N/A'):.2f}")
                  
        except Exception as e:
            print(f"Valuation factors calculation error: {e}")
            factors = {
                'pe_ratio': np.nan,
                'pb_ratio': np.nan,
                'ps_ratio': np.nan,
                'dividend_yield': 0,
                'ev_to_ebitda': np.nan,
                'current_price': np.nan,
                'market_cap': np.nan
            }
        
        self.analysis_results['valuation_factors'] = factors
        return factors
    
    def calculate_quality_factors(self):
        """Calculate quality and safety factors (FIXED VERSION)"""
        print("Calculating quality factors...")
        
        factors = {}
        
        try:
            info = self.stock.info
            
            # Return on Equity (ROE) - from info
            factors['roe'] = info.get('returnOnEquity', np.nan)
            if factors['roe']:
                factors['roe'] *= 100
            
            # Gross Margin - from info
            factors['gross_margin'] = info.get('grossMargins', np.nan)
            if factors['gross_margin']:
                factors['gross_margin'] *= 100
            
            # FIXED: Calculate Debt Ratio properly
            debt_ratio_calculated = False
            
            # Method 1: Try to get from balance sheet (more accurate)
            if not self.balance_sheet.empty:
                try:
                    latest_bs = self.balance_sheet.iloc[0]
                    
                    # Try different column names
                    total_liabilities = None
                    total_assets = None
                    
                    # Common column names for total liabilities
                    liability_cols = ['Total Liabilities Net Minority Interest', 
                                     'Total Liabilities',
                                     'Liabilities']
                    
                    for col in liability_cols:
                        if col in latest_bs:
                            total_liabilities = latest_bs[col]
                            break
                    
                    # Common column names for total assets
                    asset_cols = ['Total Assets', 'Assets']
                    for col in asset_cols:
                        if col in latest_bs:
                            total_assets = latest_bs[col]
                            break
                    
                    if total_liabilities is not None and total_assets is not None and total_assets > 0:
                        # Validate: debt shouldn't be astronomically high
                        if abs(total_liabilities) < abs(total_assets) * 100:  # Debt < 100x assets
                            factors['debt_ratio'] = (total_liabilities / total_assets) * 100
                            debt_ratio_calculated = True
                except Exception as e:
                    print(f"Balance sheet calculation warning: {e}")
            
            # Method 2: If balance sheet calculation failed, try from info with validation
            if not debt_ratio_calculated:
                total_debt = info.get('totalDebt', None)
                total_assets = info.get('totalAssets', None)
                
                if (total_debt is not None and total_assets is not None and 
                    total_assets > 0):
                    
                    # Validate the values
                    if abs(total_debt) < abs(total_assets) * 100:
                        factors['debt_ratio'] = (total_debt / total_assets) * 100
                        debt_ratio_calculated = True
            
            # Method 3: If still not calculated, set to NaN
            if not debt_ratio_calculated:
                factors['debt_ratio'] = np.nan
            
            # Free Cash Flow
            factors['free_cashflow'] = info.get('freeCashflow', np.nan)
            
            # Free Cash Flow Yield with validation
            market_cap = self.analysis_results['valuation_factors'].get('market_cap', np.nan)
            if (factors['free_cashflow'] and market_cap and 
                market_cap > 0 and not np.isnan(factors['free_cashflow'])):
                
                if abs(factors['free_cashflow']) < abs(market_cap) * 10:
                    factors['fcf_yield'] = (factors['free_cashflow'] / market_cap) * 100
                else:
                    factors['fcf_yield'] = np.nan
            else:
                factors['fcf_yield'] = np.nan
            
            print(f"Quality factors calculated: ROE={factors.get('roe', 'N/A'):.1f}%, "
                  f"Debt Ratio={factors.get('debt_ratio', 'N/A'):.1f}%")
                  
        except Exception as e:
            print(f"Quality factors calculation error: {e}")
            factors = {
                'roe': np.nan,
                'gross_margin': np.nan,
                'debt_ratio': np.nan,
                'free_cashflow': np.nan,
                'fcf_yield': np.nan
            }
        
        self.analysis_results['quality_factors'] = factors
        return factors
    
    def calculate_market_factors(self):
        """Calculate market and sentiment factors"""
        print("Calculating market factors...")
        
        factors = {}
        
        try:
            # Calculate price history percentile
            if not self.historical_data.empty:
                price_data = self.historical_data['Close']
                current_price = price_data.iloc[-1]
                
                # 52-week price range
                if len(price_data) >= 252:
                    factors['52w_high'] = price_data.tail(252).max()
                    factors['52w_low'] = price_data.tail(252).min()
                    factors['current_vs_52w'] = ((current_price - factors['52w_low']) / 
                                                (factors['52w_high'] - factors['52w_low']) * 100 
                                                if (factors['52w_high'] - factors['52w_low']) > 0 else 50)
                else:
                    factors['52w_high'] = price_data.max()
                    factors['52w_low'] = price_data.min()
                    factors['current_vs_52w'] = 50
                
                # Historical percentile (simplified)
                factors['price_percentile'] = (price_data < current_price).mean() * 100
            
            # Beta coefficient
            info = self.stock.info
            factors['beta'] = info.get('beta', np.nan)
            
            print(f"Market factors calculated: 52-week position={factors.get('current_vs_52w', 'N/A'):.1f}%, "
                  f"Beta={factors.get('beta', 'N/A'):.2f}")
                  
        except Exception as e:
            print(f"Market factors calculation error: {e}")
            factors = {
                '52w_high': np.nan,
                '52w_low': np.nan,
                'current_vs_52w': np.nan,
                'price_percentile': np.nan,
                'beta': np.nan
            }
        
        self.analysis_results['market_factors'] = factors
        return factors
    
    def validate_financial_data(self):
        """Validate financial data for reasonableness"""
        print("\nValidating financial data...")
        
        validation_results = {}
        
        # Get current factors
        valuation = self.analysis_results.get('valuation_factors', {})
        quality = self.analysis_results.get('quality_factors', {})
        
        # Validate P/E ratio
        pe = valuation.get('pe_ratio', np.nan)
        if not np.isnan(pe):
            if pe < 0:
                validation_results['pe'] = f"⚠️ Negative P/E: {pe:.2f}"
            elif pe > 1000:
                validation_results['pe'] = f"⚠️ Extremely high P/E: {pe:.2f}"
            else:
                validation_results['pe'] = f"✅ Reasonable P/E: {pe:.2f}"
        
        # Validate Debt Ratio
        debt_ratio = quality.get('debt_ratio', np.nan)
        if not np.isnan(debt_ratio):
            if debt_ratio < 0:
                validation_results['debt_ratio'] = f"❌ Negative debt ratio: {debt_ratio:.2f}%"
            elif debt_ratio > 10000:
                validation_results['debt_ratio'] = f"❌ Impossible debt ratio: {debt_ratio:.2f}%"
            elif debt_ratio > 100:
                validation_results['debt_ratio'] = f"⚠️ Very high debt ratio: {debt_ratio:.2f}%"
            else:
                validation_results['debt_ratio'] = f"✅ Reasonable debt ratio: {debt_ratio:.2f}%"
        
        # Validate Market Cap
        market_cap = valuation.get('market_cap', np.nan)
        if not np.isnan(market_cap):
            if market_cap < 0:
                validation_results['market_cap'] = f"❌ Negative market cap: ${market_cap:,.0f}"
            elif market_cap > 1e15:
                validation_results['market_cap'] = f"❌ Impossible market cap: ${market_cap:,.0f}"
            else:
                validation_results['market_cap'] = f"✅ Reasonable market cap: ${market_cap:,.0f}"
        
        # Print validation results
        print("Data Validation Results:")
        for factor, message in validation_results.items():
            print(f"  {factor}: {message}")
        
        return validation_results
    
    def align_data(self):
        """Align stock and benchmark data"""
        if self.historical_data.empty or self.benchmark_data.empty:
            print("Warning: No data available, cannot align")
            return pd.Series(), pd.Series()
            
        # Ensure both have Close price data
        if 'Close' not in self.historical_data.columns or 'Close' not in self.benchmark_data.columns:
            print("Warning: Missing Close price data")
            return pd.Series(), pd.Series()
            
        # Find common time index
        common_index = self.historical_data.index.intersection(self.benchmark_data.index)
        
        if len(common_index) > 10:
            aligned_stock = self.historical_data.loc[common_index, 'Close']
            aligned_benchmark = self.benchmark_data.loc[common_index, 'Close']
            return aligned_stock, aligned_benchmark
        else:
            print("Warning: Few common data points, using simple alignment")
            # Take shorter data length
            min_len = min(len(self.historical_data), len(self.benchmark_data))
            aligned_stock = self.historical_data['Close'].iloc[-min_len:]
            aligned_benchmark = self.benchmark_data['Close'].iloc[-min_len:]
            return aligned_stock, aligned_benchmark
    
    def generate_valuation_radar(self):
        """Generate valuation radar chart"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left chart: Valuation factors comparison
        valuation = self.analysis_results['valuation_factors']
        
        # Prepare radar chart data
        radar_categories = ['P/E', 'P/B', 'P/S', 'Div Yield', 'EV/EBITDA']
        
        # Get actual values
        actual_values = [
            valuation.get('pe_ratio', 0),
            valuation.get('pb_ratio', 0),
            valuation.get('ps_ratio', 0),
            valuation.get('dividend_yield', 0),
            valuation.get('ev_to_ebitda', 0)
        ]
        
        # Set reasonable maximum values
        max_values = [30, 5, 10, 5, 20]
        
        # Normalize data
        normalized_values = []
        for i, val in enumerate(actual_values):
            if np.isnan(val) or val == 0:
                normalized_values.append(0)
            else:
                normalized = min(val / max_values[i], 1.0)
                normalized_values.append(normalized)
        
        # Radar chart plotting
        N = len(radar_categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        values = normalized_values + normalized_values[:1]
        
        # Create radar chart
        ax = plt.subplot(1, 2, 1, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2, label='Current Valuation')
        ax.fill(angles, values, alpha=0.25)
        
        # Add grid lines
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_categories)
        ax.set_ylim(0, 1)
        ax.set_title('Valuation Factors Radar Chart', size=14, weight='bold')
        ax.legend(loc='upper right')
        
        # Right chart: Quality factors bar chart
        quality = self.analysis_results['quality_factors']
        
        quality_categories = ['ROE (%)', 'Gross Margin (%)', 'Debt Ratio (%)', 'FCF Yield (%)']
        quality_values = [
            quality.get('roe', 0),
            quality.get('gross_margin', 0),
            quality.get('debt_ratio', 0),
            quality.get('fcf_yield', 0)
        ]
        
        # Set colors (green for good, red for caution)
        colors = []
        for i, cat in enumerate(quality_categories):
            val = quality_values[i]
            if cat == 'Debt Ratio (%)':
                colors.append('red' if val > 60 else ('orange' if val > 40 else 'green'))
            else:
                colors.append('green' if val > 0 else ('orange' if not np.isnan(val) else 'gray'))
        
        bars = axes[1].bar(quality_categories, quality_values, color=colors)
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_title('Quality Factors Analysis', size=14, weight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, quality_values):
            if not np.isnan(val):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(self.output_dir, f"{self.ticker}_valuation_chart.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Valuation chart saved: {chart_path}")
        
        return fig
    
    def generate_price_analysis(self):
        """Generate price analysis chart"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Align data
        aligned_stock, aligned_benchmark = self.align_data()
        
        if aligned_stock.empty or aligned_benchmark.empty:
            # If alignment fails, use only stock data
            if not self.historical_data.empty:
                price_data = self.historical_data['Close']
            else:
                axes[0, 0].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                              transform=axes[0, 0].transAxes, fontsize=12)
                plt.tight_layout()
                
                # Save chart even if empty
                chart_path = os.path.join(self.output_dir, f"{self.ticker}_price_chart.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                return fig
        else:
            price_data = aligned_stock
        
        # 1. Price trend
        axes[0, 0].plot(price_data.index, price_data.values, linewidth=2, label=self.ticker, color='blue')
        
        # Add moving averages
        window_50 = min(50, len(price_data))
        window_200 = min(200, len(price_data))
        
        if window_50 > 1:
            ma_50 = price_data.rolling(window=window_50).mean()
            axes[0, 0].plot(ma_50.index, ma_50.values, 'orange', linewidth=1, label=f'MA{window_50}')
        
        if window_200 > 1:
            ma_200 = price_data.rolling(window=window_200).mean()
            axes[0, 0].plot(ma_200.index, ma_200.values, 'red', linewidth=1, label=f'MA{window_200}')
        
        axes[0, 0].set_title(f'{self.ticker} Price Trend', size=12, weight='bold')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Relative performance (if benchmark data available)
        if not aligned_benchmark.empty and len(aligned_stock) == len(aligned_benchmark):
            try:
                stock_norm = price_data / price_data.iloc[0] * 100
                benchmark_norm = aligned_benchmark / aligned_benchmark.iloc[0] * 100
                
                axes[0, 1].plot(stock_norm.index, stock_norm.values, 'b', linewidth=2, label=self.ticker)
                axes[0, 1].plot(benchmark_norm.index, benchmark_norm.values, 'g', linewidth=2, 
                               label=self.benchmark_ticker)
                
                # Simple fill, no complex where condition
                axes[0, 1].fill_between(stock_norm.index, stock_norm.values, benchmark_norm.values, 
                                       alpha=0.3)
                
                axes[0, 1].set_title('Relative Performance (Normalized to 100)', size=12, weight='bold')
                axes[0, 1].set_ylabel('Performance Index')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            except:
                axes[0, 1].text(0.5, 0.5, 'Relative performance calculation failed', ha='center', va='center', 
                              transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Relative Performance', size=12, weight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'No benchmark data', ha='center', va='center', 
                          transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Relative Performance', size=12, weight='bold')
        
        # 3. Return distribution histogram
        try:
            returns = price_data.pct_change().dropna() * 100
            if len(returns) > 10:
                bins = min(30, len(returns)//5)
                axes[1, 0].hist(returns, bins=bins, edgecolor='black', alpha=0.7)
                
                mean_return = returns.mean()
                median_return = returns.median()
                
                axes[1, 0].axvline(x=mean_return, color='red', linestyle='--', 
                                 label=f'Mean: {mean_return:.2f}%')
                axes[1, 0].axvline(x=median_return, color='green', linestyle='--', 
                                 label=f'Median: {median_return:.2f}%')
                
                axes[1, 0].set_xlabel('Daily Return (%)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Daily Return Distribution', size=12, weight='bold')
                axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                              transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Daily Return Distribution', size=12, weight='bold')
        except:
            axes[1, 0].text(0.5, 0.5, 'Calculation error', ha='center', va='center', 
                          transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Daily Return Distribution', size=12, weight='bold')
        
        # 4. Rolling volatility
        try:
            if len(returns) >= 21:
                rolling_vol = returns.rolling(window=21).std() * np.sqrt(252)
                axes[1, 1].plot(rolling_vol.index, rolling_vol.values, 'purple', linewidth=2)
                
                avg_vol = rolling_vol.mean()
                axes[1, 1].axhline(y=avg_vol, color='red', linestyle='--', 
                                  label=f'Average: {avg_vol:.1f}%')
                
                axes[1, 1].set_xlabel('Date')
                axes[1, 1].set_ylabel('Annualized Volatility (%)')
                axes[1, 1].set_title('21-Day Rolling Volatility', size=12, weight='bold')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                              transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('21-Day Rolling Volatility', size=12, weight='bold')
        except:
            axes[1, 1].text(0.5, 0.5, 'Calculation error', ha='center', va='center', 
                          transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('21-Day Rolling Volatility', size=12, weight='bold')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(self.output_dir, f"{self.ticker}_price_analysis_chart.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Price analysis chart saved: {chart_path}")
        
        return fig
    
    def generate_factor_summary_table(self):
        """Generate factor summary table"""
        valuation = self.analysis_results.get('valuation_factors', {})
        quality = self.analysis_results.get('quality_factors', {})
        market = self.analysis_results.get('market_factors', {})
        
        summary_data = []
        
        # Valuation factors
        summary_data.append(['Valuation Factors', 'Value', 'Assessment'])
        
        pe = valuation.get('pe_ratio', np.nan)
        summary_data.append(['P/E Ratio', f"{pe:.2f}" if not np.isnan(pe) else 'N/A', 
                           self._evaluate_pe(pe)])
        
        pb = valuation.get('pb_ratio', np.nan)
        summary_data.append(['P/B Ratio', f"{pb:.2f}" if not np.isnan(pb) else 'N/A', 
                           self._evaluate_pb(pb)])
        
        ps = valuation.get('ps_ratio', np.nan)
        summary_data.append(['P/S Ratio', f"{ps:.2f}" if not np.isnan(ps) else 'N/A',
                           self._evaluate_ps(ps)])
        
        dividend = valuation.get('dividend_yield', 0)
        summary_data.append(['Dividend Yield (%)', f"{dividend:.2f}%" if not np.isnan(dividend) else 'N/A',
                           self._evaluate_dividend(dividend)])
        
        ev_ebitda = valuation.get('ev_to_ebitda', np.nan)
        summary_data.append(['EV/EBITDA', f"{ev_ebitda:.2f}" if not np.isnan(ev_ebitda) else 'N/A',
                           self._evaluate_ev_ebitda(ev_ebitda)])
        
        # Quality factors
        summary_data.append(['\nQuality Factors', 'Value', 'Assessment'])
        
        roe = quality.get('roe', np.nan)
        summary_data.append(['ROE (%)', f"{roe:.2f}%" if not np.isnan(roe) else 'N/A',
                           self._evaluate_roe(roe)])
        
        margin = quality.get('gross_margin', np.nan)
        summary_data.append(['Gross Margin (%)', f"{margin:.2f}%" if not np.isnan(margin) else 'N/A',
                           self._evaluate_margin(margin)])
        
        debt = quality.get('debt_ratio', np.nan)
        summary_data.append(['Debt Ratio (%)', f"{debt:.2f}%" if not np.isnan(debt) else 'N/A',
                           self._evaluate_debt(debt)])
        
        fcf = quality.get('fcf_yield', np.nan)
        summary_data.append(['FCF Yield (%)', f"{fcf:.2f}%" if not np.isnan(fcf) else 'N/A',
                           self._evaluate_fcf(fcf)])
        
        # Market factors
        summary_data.append(['\nMarket Factors', 'Value', 'Assessment'])
        
        pos_52w = market.get('current_vs_52w', np.nan)
        summary_data.append(['52-Week Position (%)', f"{pos_52w:.1f}%" if not np.isnan(pos_52w) else 'N/A',
                           self._evaluate_52w(pos_52w)])
        
        beta = market.get('beta', np.nan)
        summary_data.append(['Beta Coefficient', f"{beta:.2f}" if not np.isnan(beta) else 'N/A',
                           self._evaluate_beta(beta)])
        
        percentile = market.get('price_percentile', np.nan)
        summary_data.append(['Historical Price Percentile', f"{percentile:.1f}%" if not np.isnan(percentile) else 'N/A',
                           self._evaluate_percentile(percentile)])
        
        return pd.DataFrame(summary_data[1:], columns=summary_data[0])
    
    def _evaluate_pe(self, pe):
        if np.isnan(pe):
            return "Data missing"
        if pe < 15:
            return "⭐ Low (Value range)"
        elif pe < 25:
            return "✅ Reasonable"
        elif pe < 40:
            return "⚠️ High"
        else:
            return "❌ Very high"
    
    def _evaluate_pb(self, pb):
        if np.isnan(pb):
            return "Data missing"
        if pb < 1.5:
            return "⭐ Low"
        elif pb < 3:
            return "✅ Reasonable"
        elif pb < 5:
            return "⚠️ High"
        else:
            return "❌ Very high"
    
    def _evaluate_roe(self, roe):
        if np.isnan(roe):
            return "Data missing"
        if roe > 20:
            return "⭐ Excellent"
        elif roe > 15:
            return "✅ Good"
        elif roe > 10:
            return "⚠️ Average"
        else:
            return "❌ Poor"
    
    def _evaluate_dividend(self, yield_):
        if np.isnan(yield_):
            return "Data missing"
        if yield_ > 4:
            return "⭐ High dividend"
        elif yield_ > 2:
            return "✅ Reasonable"
        elif yield_ > 0:
            return "⚠️ Low"
        else:
            return "No dividend"
    
    def _evaluate_debt(self, debt_ratio):
        """Evaluate Debt Ratio with industry context"""
        if np.isnan(debt_ratio):
            return "Data missing"
        
        # Different thresholds for different industries
        current_ticker = self.ticker
        
        # For financial companies (banks, insurance), higher debt ratios are normal
        financial_keywords = ['BANK', 'FINANCIAL', 'INSURANCE', 'JPM', 'BAC', 'WFC', 'C']
        is_financial = any(keyword in current_ticker.upper() for keyword in financial_keywords)
        
        # For utilities, higher debt is also common
        utility_keywords = ['UTILITY', 'POWER', 'ENERGY', 'NEE', 'DUK', 'SO']
        is_utility = any(keyword in current_ticker.upper() for keyword in utility_keywords)
        
        if is_financial:
            # Financial industry standards
            if debt_ratio > 90:
                return "❌ Very high (even for financial)"
            elif debt_ratio > 85:
                return "⚠️ High"
            elif debt_ratio > 80:
                return "✅ Normal for financial"
            else:
                return "⭐ Low for financial"
        
        elif is_utility:
            # Utility industry standards
            if debt_ratio > 70:
                return "❌ Very high"
            elif debt_ratio > 60:
                return "⚠️ High"
            elif debt_ratio > 50:
                return "✅ Normal for utility"
            else:
                return "⭐ Low for utility"
        
        else:
            # General standards for non-financial companies
            if debt_ratio > 70:
                return "❌ Very high"
            elif debt_ratio > 50:
                return "⚠️ High"
            elif debt_ratio > 30:
                return "✅ Reasonable"
            else:
                return "⭐ Low debt"
    
    def _evaluate_52w(self, position):
        if np.isnan(position):
            return "Data missing"
        if position < 30:
            return "📉 Near low"
        elif position < 70:
            return "↔️ Middle range"
        else:
            return "📈 Near high"
    
    def _evaluate_ps(self, ps):
        if np.isnan(ps): return "Data missing"
        return "⭐ Low" if ps < 2 else "✅ Reasonable" if ps < 5 else "⚠️ High"
    
    def _evaluate_ev_ebitda(self, ev_ebitda):
        if np.isnan(ev_ebitda): return "Data missing"
        return "⭐ Low" if ev_ebitda < 10 else "✅ Reasonable" if ev_ebitda < 20 else "⚠️ High"
    
    def _evaluate_margin(self, margin):
        if np.isnan(margin): return "Data missing"
        return "⭐ High" if margin > 40 else "✅ Good" if margin > 30 else "⚠️ Average"
    
    def _evaluate_fcf(self, fcf_yield):
        if np.isnan(fcf_yield): return "Data missing"
        return "⭐ Excellent" if fcf_yield > 8 else "✅ Good" if fcf_yield > 5 else "⚠️ Average"
    
    def _evaluate_beta(self, beta):
        if np.isnan(beta): return "Data missing"
        return "Low volatility" if beta < 0.8 else "Market neutral" if beta < 1.2 else "High volatility"
    
    def _evaluate_percentile(self, percentile):
        if np.isnan(percentile): return "Data missing"
        return "📉 Historical low" if percentile < 30 else "↔️ Historical middle" if percentile < 70 else "📈 Historical high"
    
    def generate_summary_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print(f"Value Investing Analysis Report - {self.ticker}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Display factor summary table
        try:
            summary_table = self.generate_factor_summary_table()
            print("\n📊 Factor Analysis Summary:")
            print(summary_table.to_string(index=False))
        except Exception as e:
            print(f"\nFactor summary table generation error: {e}")
            summary_table = pd.DataFrame()
        
        # Investment recommendation
        print("\n💡 Investment Recommendation:")
        self._generate_investment_recommendation()
        
        # Risk warning
        print("\n⚠️ Risk Warning:")
        print("1. Historical analysis doesn't guarantee future performance")
        print("2. Financial statement data may be lagging")
        print("3. Market sentiment and macro factors can change rapidly")
        print("4. Combine with qualitative analysis and latest financial reports")
        
        return summary_table
    
    def _generate_investment_recommendation(self):
        """Generate investment recommendation"""
        v = self.analysis_results.get('valuation_factors', {})
        q = self.analysis_results.get('quality_factors', {})
        m = self.analysis_results.get('market_factors', {})
        
        score = 0
        reasons = []
        
        try:
            # Valuation scoring
            pe = v.get('pe_ratio', 100)
            if not np.isnan(pe):
                if pe < 15: 
                    score += 2
                    reasons.append("Low P/E ratio")
                elif pe < 25: 
                    score += 1
            
            pb = v.get('pb_ratio', 10)
            if not np.isnan(pb):
                if pb < 1.5: 
                    score += 2
                    reasons.append("Reasonable P/B ratio")
                elif pb < 3: 
                    score += 1
            
            dividend = v.get('dividend_yield', 0)
            if not np.isnan(dividend) and dividend > 3: 
                score += 1
                reasons.append("Attractive dividend yield")
            
            # Quality scoring
            roe = q.get('roe', 0)
            if not np.isnan(roe):
                if roe > 15: 
                    score += 2
                    reasons.append("Strong profitability")
                elif roe > 10: 
                    score += 1
            
            debt = q.get('debt_ratio', 100)
            if not np.isnan(debt):
                # Adjusted scoring for different industries
                current_ticker = self.ticker
                is_financial = any(keyword in current_ticker.upper() 
                                 for keyword in ['BANK', 'FINANCIAL', 'INSURANCE'])
                is_utility = any(keyword in current_ticker.upper() 
                               for keyword in ['UTILITY', 'POWER', 'ENERGY'])
                
                if is_financial:
                    if debt < 85:
                        score += 1
                        reasons.append("Reasonable debt for financial")
                elif is_utility:
                    if debt < 60:
                        score += 1
                        reasons.append("Reasonable debt for utility")
                else:
                    if debt < 50: 
                        score += 1
                        reasons.append("Solid financial structure")
            
            fcf = q.get('fcf_yield', 0)
            if not np.isnan(fcf):
                if fcf > 5: 
                    score += 2
                    reasons.append("Strong free cash flow")
                elif fcf > 3: 
                    score += 1
            
            # Market sentiment scoring (contrarian)
            pos_52w = m.get('current_vs_52w', 50)
            if not np.isnan(pos_52w) and pos_52w < 40: 
                score += 1
                reasons.append("Stock price at relative low")
            
            # Generate recommendation
            print(f"Composite Score: {score}/10")
            
            if score >= 8:
                print("Recommendation: 🎯 STRONG BUY")
                print("Reasons: " + ", ".join(reasons[:3]))
                print("Strategy: Consider phased position building, suitable for long-term holding")
            elif score >= 6:
                print("Recommendation: ✅ BUY")
                print("Reasons: " + ", ".join(reasons[:3]))
                print("Strategy: Wait for appropriate price entry, set stop-loss")
            elif score >= 4:
                print("Recommendation: ⚠️ HOLD")
                print("Strategy: If already holding, continue to monitor; if not, wait for better opportunity")
            else:
                print("Recommendation: ❌ AVOID")
                print("Strategy: Fundamentals or valuation not attractive, consider other opportunities")
                
        except Exception as e:
            print(f"Investment recommendation generation error: {e}")
            print("Recommendation: ⚠️ Need more data for analysis")
    
    def save_report(self, filename=None):
        """Save report as text file"""
        if filename is None:
            filename = f"{self.ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write(f"Value Investing Analysis Report - {self.ticker}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*60 + "\n\n")
                
                # Save factor summary
                summary_table = self.generate_factor_summary_table()
                f.write("📊 FACTOR ANALYSIS SUMMARY:\n")
                f.write(summary_table.to_string(index=False) + "\n\n")
                
                # Save investment recommendation
                f.write("💡 INVESTMENT RECOMMENDATION:\n")
                # Get valuation factors for detailed analysis
                v = self.analysis_results.get('valuation_factors', {})
                q = self.analysis_results.get('quality_factors', {})
                m = self.analysis_results.get('market_factors', {})
                
                # Add detailed analysis
                f.write(f"\nDetailed Analysis:\n")
                f.write(f"- Current Price: ${v.get('current_price', 'N/A'):.2f}\n")
                f.write(f"- Market Cap: ${v.get('market_cap', 'N/A'):,.0f}\n")
                f.write(f"- P/E Ratio: {v.get('pe_ratio', 'N/A'):.2f}\n")
                f.write(f"- P/B Ratio: {v.get('pb_ratio', 'N/A'):.2f}\n")
                f.write(f"- Dividend Yield: {v.get('dividend_yield', 'N/A'):.2f}%\n")
                f.write(f"- ROE: {q.get('roe', 'N/A'):.1f}%\n")
                f.write(f"- Debt Ratio: {q.get('debt_ratio', 'N/A'):.1f}%\n")
                f.write(f"- 52-Week Position: {m.get('current_vs_52w', 'N/A'):.1f}%\n")
                
                f.write(f"\nChart files saved:\n")
                f.write(f"- Valuation Chart: {self.ticker}_valuation_chart.png\n")
                f.write(f"- Price Analysis Chart: {self.ticker}_price_analysis_chart.png\n")
                
            print(f"\nReport saved: {filepath}")
            return True
        except Exception as e:
            print(f"Report saving failed: {e}")
            return False
    
    def run_full_analysis(self):
        """Run complete analysis workflow (FIXED VERSION)"""
        print("Starting value investing analysis workflow...")
        print("-" * 40)
        
        # 1. Fetch data
        self.fetch_data()
        
        # 2. Check if data is valid
        if self.historical_data.empty:
            print(f"Error: Cannot fetch historical data for {self.ticker}")
            return None, None
            
        # 3. Calculate all factors
        try:
            self.calculate_valuation_factors()
            self.calculate_quality_factors()
            self.calculate_market_factors()
        except Exception as e:
            print(f"Factor calculation error: {e}")
            self.analysis_results = {
                'valuation_factors': {},
                'quality_factors': {},
                'market_factors': {}
            }
        
        # 4. Validate the calculated data
        self.validate_financial_data()
        
        # 5. Generate visualization charts
        print("\nGenerating analysis charts...")
        try:
            fig1 = self.generate_valuation_radar()
        except Exception as e:
            print(f"Radar chart generation error: {e}")
            fig1 = None
            
        try:
            fig2 = self.generate_price_analysis()
        except Exception as e:
            print(f"Price analysis chart generation error: {e}")
            fig2 = None
        
        # 6. Generate text report
        try:
            summary = self.generate_summary_report()
        except Exception as e:
            print(f"Report generation error: {e}")
            summary = pd.DataFrame()
        
        # 7. Ask user if they want to display charts
        show_charts = input("\nDo you want to display charts? (y/n): ").lower() == 'y'
        if show_charts and (fig1 is not None or fig2 is not None):
            plt.show()
        else:
            print("Charts saved to files. Not displaying on screen.")
        
        # 8. Save report
        self.save_report()
        
        return self.analysis_results, summary

# ============================================================================
# Usage Examples
# ============================================================================
if __name__ == "__main__":
    print("VALUE INVESTING ANALYSIS TOOL")
    print("=" * 60)
    
    # Example 1: Analyze Apple
    print("\nExample 1: Analyzing Apple (AAPL)")
    print("-" * 40)
    
    analyzer = ValueInvestingAnalyzer(
        ticker='MRNA',  # Apple
        benchmark_ticker='^GSPC',  # S&P 500
        period='3y',  # 3 years data
        output_dir='analysis_reports'
    )
    
    results, summary = analyzer.run_full_analysis()
    
    # Example 2: User can analyze custom stocks
    print("\n" + "="*60)
    custom_analysis = input("\nDo you want to analyze another stock? (y/n): ").lower() == 'y'
    
    while custom_analysis:
        ticker = input("Enter stock ticker (e.g., MSFT, GOOGL, TSLA): ").upper()
        if not ticker:
            break
            
        print(f"\nAnalyzing {ticker}...")
        print("-" * 40)
        
        custom_analyzer = ValueInvestingAnalyzer(
            ticker=ticker,
            benchmark_ticker='^GSPC',
            period='3y',
            output_dir='analysis_reports'
        )
        
        custom_results, custom_summary = custom_analyzer.run_full_analysis()
        
        custom_analysis = input("\nAnalyze another stock? (y/n): ").lower() == 'y'
    
    print("\nAnalysis complete! All reports and charts saved in 'analysis_reports' folder.")
    print("Files created:")
    print("- [ticker]_valuation_chart.png")
    print("- [ticker]_price_analysis_chart.png")
    print("- [ticker]_analysis_[timestamp].txt")