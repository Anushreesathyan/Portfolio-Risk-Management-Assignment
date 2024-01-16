import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def download_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data['Adj Close']

def calculate_daily_returns(stock_prices):
    return stock_prices.pct_change().dropna()

def calculate_portfolio_var(returns, weights):
    portfolio_var = np.dot(weights.T, np.dot(returns.cov(), weights))
    return portfolio_var

def calculate_portfolio_volatility(returns, weights):
    portfolio_var = calculate_portfolio_var(returns, weights)
    portfolio_volatility = np.sqrt(portfolio_var)
    return portfolio_volatility

def calculate_value_at_risk(returns, weights, confidence_level=0.95):
    portfolio_volatility = calculate_portfolio_volatility(returns, weights)
    z_score = np.percentile(returns @ weights, (1 - confidence_level) * 100)
    var = portfolio_volatility * z_score
    return var

def calculate_beta(stock_returns, market_returns):
    covariance_matrix = np.cov(stock_returns, market_returns)
    beta = covariance_matrix[0, 1] / np.var(market_returns)
    return beta

def main():
    # Input parameters
    tickers = ['AAPL', 'GOOGL', 'MSFT']  # Replace with your stock tickers
    weights = np.array([0.4, 0.4, 0.2])  # Replace with your portfolio weights
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    # Download stock data
    stock_prices = pd.DataFrame({ticker: download_stock_data(ticker, start_date, end_date) for ticker in tickers})

    # Calculate daily returns
    returns = calculate_daily_returns(stock_prices)

    # Calculate VaR
    confidence_level = 0.95
    var = calculate_value_at_risk(returns, weights, confidence_level)
    print(f"Value at Risk (VaR) at {confidence_level * 100}% confidence level: {var:.2%}")

    # Calculate Beta
    market_ticker = '^GSPC'  # Replace with your market index
    market_prices = download_stock_data(market_ticker, start_date, end_date)
    market_returns = calculate_daily_returns(market_prices)
    beta = calculate_beta(returns[tickers[0]], market_returns)
    print(f"Beta: {beta:.4f}")

    # Visualization: Cumulative Portfolio Returns
    cumulative_returns = (1 + returns @ weights).cumprod()
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label='Portfolio')
    plt.plot((1 + market_returns).cumprod(), label='Market', linestyle='--')
    plt.legend()
    plt.title('Cumulative Portfolio Returns vs. Market')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.show()

if __name__ == "__main__":
    main()
