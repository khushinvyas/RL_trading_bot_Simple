# Trading Bot Project Report

## 1. Problem Description
This project implements a reinforcement learning trading bot that learns to make buy/sell/hold decisions for stock trading. The goal is to maximize portfolio returns while considering transaction costs and market conditions.

## 2. Environment and Agent

### States:
The observation space includes:
- Current stock price
- Trading volume
- Number of shares owned
- Current cash balance
- Technical indicators (SMA20, SMA50, MACD, Signal line)

### Actions:
- 0: Hold (do nothing)
- 1: Buy (purchase shares worth 10% of current assets)
- 2: Sell (liquidate all current position)

### Objective:
Maximize total portfolio value (cash + stock value) through optimal trading decisions.

### Model:
The implementation supports PPO, A2C, and DQN algorithms from Stable Baselines3.

### Discount Factor:
The default γ=0.99 from Stable Baselines3 is used.

## 3. MDP Formulation
- **State Space**: 8-dimensional continuous space (price, volume, holdings, balance, indicators)
- **Action Space**: Discrete 3 actions (hold, buy, sell)
- **Reward Function**: 
  - Base reward: Logarithmic returns between steps
  - Bonus: +0.1 for taking buy/sell actions
  - Penalties: For holding too long, buying with low cash, or selling with no position
- **Transition Dynamics**: Determined by market prices and trading impact

## 4. Method Description
The implementation uses:
- Gymnasium environment for trading simulation
- Yahoo Finance API for historical data
- Technical indicators (SMAs, MACD) for state features
- Three RL algorithms (PPO, A2C, DQN) for comparison
- Custom callback for training monitoring
- Golden Cross strategy as baseline

## 5. Results
Performance metrics are logged in `trading_logs_revised/` including:
- Portfolio value over time
- Trade execution details
- Model comparison metrics

## 6. Graphs
See included visualizations in `trading_logs_revised/`:
- `golden_cross_strategy.png`: Baseline strategy performance
- `*_evaluation.png`: RL model evaluations
- `model_comparison_portfolio.png`: Performance comparison