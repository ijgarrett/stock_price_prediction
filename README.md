u# Stock Price Prediction with LSTM

This project uses historical stock data to predict the next day’s closing price of a stock using an LSTM (Long Short-Term Memory) neural network in PyTorch. I completed this project in September 2025, implementing a full pipeline for feature engineering, sequence creation, model training, evaluation, and visualization.

## Project Overview

The goal is to predict the next day’s closing price of a stock using past stock prices and engineered features. The model uses a sliding window of past 30 days to forecast the next closing price. Additional features beyond the raw closing price, such as returns, momentum, moving averages, volatility, momentum, and volume, are included to improve predictive performance.

## Dataset

- Historical stock price data fetched from Yahoo Finance using `yfinance`
- Example stock: Apple Inc. (`AAPL`)
- Time period: 2020-01-01 to present
- Features:
  - `Close`: daily closing price
  - `Volume`: total shares traded
  - `Return`: daily percent change in closing price
  - `MA5`: 5-day moving average of closing price
  - `Volatility`: 5-day rolling standard deviation of returns
  - `Momentum`: difference between current close and close 5 days ago

## Tools and Libraries

- Python  
- PyTorch (LSTM model implementation)  
- pandas & NumPy (data handling)  
- scikit-learn (`StandardScaler` for feature normalization)  
- matplotlib (visualization)  
- yfinance (data fetching)  

## Process and Methodology

### 1. Data Preprocessing
- Collected historical stock data using `yfinance`
- Selected relevant features (`Close`, `Volume`, engineered features)
- Computed additional features:
  - **Return:** `df["Close"].pct_change().fillna(0)`  
  - **MA5:** `df["Close"].rolling(window=5).mean().fillna(df["Close"].iloc[0])`  
  - **Volatility:** `df["Return"].rolling(window=5).std().fillna(0)`  
  - **Momentum:** `df["Close"].diff(5).fillna(0)`  
- Filled remaining missing values with 0
- Scaled all features using `StandardScaler`

### 2. Sequence Creation
- Used a sliding window of 30 days to create input sequences
- Each sequence includes the last 29 days as features and predicts the next day’s closing price
- Converted sequences to PyTorch tensors for model training

### 3. Model Architecture
- LSTM-based neural network:
  - Input layer: number of features (6 in total)
  - Hidden dimension: 40
  - 2 LSTM layers stacked
  - Dropout: 0.15 to prevent overfitting
  - Fully connected layer mapping last hidden state to predicted price
- Optimizer: Adam with learning rate = 0.01 and weight decay = 0.001
- Loss function: Mean Squared Error (MSE)

### 4. Training
- Trained for 300 epochs
- Monitored train loss

### 5. Evaluation
- Predictions scaled back to original price using inverse transform
- Computed RMSE (Root Mean Squared Error) on train and test sets
- Visualized:
  - Actual vs predicted prices (test set)
  - Prediction error with RMSE line

## Final Model Performance

- Train RMSE: varies depending on stock and hyperparameters
- Test RMSE: varies depending on stock and hyperparameters
- Visualization shows model captures general trends but may lag on sharp spikes
- This shows that LSTMs generally can't predict market crashes or spikes well before they happen

## Files in This Project

- `stock_lstm_prediction.ipynb` (full notebook with data fetching, preprocessing, training, and plotting)
- README.md

## Timeline

9/3/25 - 9/9/25  

## Future Improvements

- Add validation split to monitor overfitting  
- Tune LSTM hyperparameters (hidden_dim, num_layers, dropout, learning rate)  
- Incorporate additional technical indicators (e.g., RSI, MACD)  
- Experiment with sequence length beyond 30 days  
- Test alternative models (GRU, Transformer-based time series models)  
- Implement early stopping during training  
