# Stock Price Prediction with Deep Learning Models

## Overview
This project implements three state-of-the-art deep learning architectures (LSTM, GRU, and Transformer) to predict stock prices using historical data from Yahoo Finance. The models are designed to capture different aspects of temporal dependencies in stock price movements, enabling comprehensive price forecasting and model performance comparison.

## Features
- Historical stock data retrieval via yfinance API
- Robust data preprocessing pipeline with advanced scaling techniques
- Three implemented deep learning architectures:
  - LSTM (Long Short-Term Memory) for long-term dependency learning
  - GRU (Gated Recurrent Unit) for efficient sequence processing
  - Transformer with temporal attention for parallel processing
- Multi-day price prediction capabilities
- Interactive visualization tools for prediction analysis
- Comprehensive performance metrics evaluation

## Requirements

### Dependencies
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- yfinance
- scikit-learn
- Matplotlib

### Installation
```bash
pip install tensorflow numpy pandas yfinance scikit-learn matplotlib
```

## Model Architectures

### LSTM Model
- Stacked LSTM layers for hierarchical feature extraction
- Dropout layers for regularization
- Batch normalization for training stability
- Dense layers for final price prediction

### GRU Model
- Dual GRU layers with optimized hyperparameters
- Dropout and batch normalization for improved generalization
- Dense layers with activation functions tuned for financial data
- Gradient clipping to prevent exploding gradients

### Transformer Model
- Custom temporal attention mechanism for price trend analysis
- Positional encoding to maintain sequence order information
- Multi-head attention layers for parallel processing
- Feed-forward networks with residual connections
- Layer normalization for stable training

## Data Preparation

### Data Collection
- Automated historical stock data download using yfinance
- Support for multiple stock symbols and date ranges
- Handling of missing values and stock splits

### Preprocessing
- MinMax scaling for price normalization
- Sequence creation with configurable window sizes
- Train/validation/test split with time-aware shuffling
- Feature engineering for technical indicators

## Making Predictions

### Prediction Capabilities
- Single and multi-day price predictions
- Rolling window prediction approach
- Confidence intervals for predictions
- Support for batch prediction


## Visualization

### Available Plots
- Actual vs predicted price comparison
- Model performance comparison


## Performance Metrics

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

### Model Monitoring
- Training loss tracking
- Validation metrics logging
- Early stopping implementation
- Model checkpointing
