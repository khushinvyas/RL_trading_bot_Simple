import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import logging
import os
import time
from datetime import datetime

# --- Configuration ---
LOG_DIR = "trading_logs_revised"
TICKER = 'GOOG' # Change ticker if needed
START_DATE = '2022-01-01'
END_DATE = '2024-01-01'
INITIAL_BALANCE = 100000
COMMISSION = 0.001
SLIPPAGE = 0.001
TRADE_SIZE_PERCENT = 0.1 # Buy shares worth 10% of current total assets per buy signal
TRAINING_TIMESTEPS = 100000 # Increased default timesteps

# --- Setup Logging ---
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBotRevised")

# --- Custom Callback ---
class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_count = 0
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self):
        # Ensure reward is accessed correctly (assuming VecEnv)
        if isinstance(self.locals["rewards"], (list, np.ndarray)):
            reward = self.locals["rewards"][0]
        else:
            reward = self.locals["rewards"]

        self.current_episode_reward += reward

        # Check dones (assuming VecEnv)
        if isinstance(self.locals["dones"], (list, np.ndarray)):
             done = self.locals["dones"][0]
        else:
             done = self.locals["dones"]

        if done:
            self.episode_count += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0

            if self.episode_count % 10 == 0:
                if self.episode_rewards: # Avoid error if list is empty
                   mean_reward = np.mean(self.episode_rewards[-10:])
                   logger.info(f"Episode {self.episode_count}, Mean Reward (last 10): {mean_reward:.2f}")
                else:
                    logger.info(f"Episode {self.episode_count} completed.")

        return True

# --- Stock Trading Environment ---
class StockTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, df, initial_balance=INITIAL_BALANCE, commission=COMMISSION, slippage=SLIPPAGE, trade_size_percent=TRADE_SIZE_PERCENT):
        super(StockTradingEnv, self).__init__()

        self.df = df # Expects df with indicators already added and NaNs dropped
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.trade_size_percent = trade_size_percent

        # Define action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Define observation space: [Price, Volume, StockOwned, Balance, SMA20, SMA50, MACD, Signal]
        # Increased shape from 7 to 8 to include Volume
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # Internal state variables will be set in reset
        self.current_step = 0
        self.balance = 0
        self.stock_owned = 0
        self.total_asset_history = []
        self.trades = []


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Important for reproducibility

        self.current_step = 0 # Start from the first available data point after dropna
        self.balance = self.initial_balance
        self.stock_owned = 0
        self.total_asset_history = [self.initial_balance]
        self.trades = []

        return self._get_obs(), {} # Return observation and info dict

    def _get_obs(self):
        """Get current observation state"""
        step_data = self.df.iloc[self.current_step]
        obs = np.array([
        float(step_data["adj_close"].iloc[0]),  # Convert to scalar
        float(step_data["Volume"].iloc[0]),
        float(self.stock_owned),
        float(self.balance),
        float(step_data["sma20"].iloc[0]),
        float(step_data["sma50"].iloc[0]),
        float(step_data["macd"].iloc[0]),
        float(step_data["signal"].iloc[0])
        ], dtype=np.float32)
        return obs

    def _get_current_price(self):
        return self.df.iloc[self.current_step]["adj_close"]

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            done = True
            # Initialize final_price before liquidation logic
            final_price = self._get_current_price()
            if self.stock_owned > 0:
                proceeds = self.stock_owned * final_price * (1 - self.commission - self.slippage)
                self.balance += proceeds
                logger.info(f"Final liquidation: Sold {self.stock_owned} shares at {final_price:.2f}")
                self.trades.append((self.current_step, "sell", self.stock_owned, final_price))
                self.stock_owned = 0
            reward = 0
            info = {'total_asset': self.balance}
            return self._get_obs(), reward, True, False, info

        current_price = self._get_current_price()  # Now a scalar
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0] if not current_price.empty else 0.0
        else:
            current_price = current_price

        info = {}

        # Calculate current total asset value before taking action
        current_total_asset = self.balance + self.stock_owned * current_price

        # Execute trade based on action
        if action == 1:  # Buy
            # Calculate trade amount based on percentage of total assets
            trade_value = current_total_asset * self.trade_size_percent
            shares_to_buy = int(trade_value // current_price)

            if isinstance(shares_to_buy, pd.Series):
                shares_to_buy_value = shares_to_buy.iloc[0] if not shares_to_buy.empty else 0
            else:
                shares_to_buy_value = shares_to_buy

            cost = shares_to_buy_value * current_price * (1 + self.commission + self.slippage)

            if isinstance(cost, pd.Series):
                cost_value = cost.iloc[0] if not cost.empty else 0
            else:
                cost_value = cost

            if cost_value <= self.balance and shares_to_buy_value > 0:
                self.balance -= cost_value
                self.stock_owned += shares_to_buy_value
                self.trades.append((self.current_step, "buy", shares_to_buy_value, current_price))
                logger.debug(f"Step {self.current_step}: Bought {shares_to_buy_value} shares at {current_price:.2f}")

        elif action == 2 and self.stock_owned > 0:  # Sell All
            proceeds = self.stock_owned * current_price * (1 - self.commission - self.slippage)
            self.balance += proceeds
            logger.debug(f"Step {self.current_step}: Sold {self.stock_owned} shares at {current_price:.2f}")
            self.trades.append((self.current_step, "sell", self.stock_owned, current_price))
            self.stock_owned = 0

        # Action 0 (Hold): Do nothing

        # Move to next step
        self.current_step += 1

        # Calculate reward with improved scaling and incentives
        new_total_asset = self.balance + self.stock_owned * self._get_current_price()
        
        # Base reward: logarithmic returns for better scaling
        if self.total_asset_history:
            ratio = float(new_total_asset) / float(self.total_asset_history[-1])
            reward = np.log(ratio) * 100 if ratio > 0 else 0
        else:
            reward = 0
        
        # Add incentives for good trades
        if action == 1:  # Buy
            reward += 0.1  # Small positive reinforcement for taking action
        elif action == 2:  # Sell
            reward += 0.1  # Small positive reinforcement for taking action
        
        # Penalties for bad behavior
        if action == 0 and self.stock_owned > 0:  # Holding too long
            reward -= 0.005 * self.stock_owned
        elif action == 1 and self.balance < (self.initial_balance * 0.1):  # Buying with low cash
            reward -= 0.2
        elif action == 2 and self.stock_owned == 0:  # Selling when no position
            reward -= 0.5
            
        self.total_asset_history.append(new_total_asset)

        # Update info dictionary
        info['total_asset'] = new_total_asset
        info['balance'] = self.balance
        info['stock_owned'] = self.stock_owned
        info['trades'] = self.trades # Pass trades info if needed

        # Gymnasium step returns: obs, reward, terminated, truncated, info
        done = self.current_step >= len(self.df) - 1
        terminated = done # Episode ends naturally
        truncated = False # Not using time limits here other than data end

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        """Display trading information"""
        current_price = self._get_current_price()
        total_asset_value = self.balance + self.stock_owned * current_price

        print("-" * 30)
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Stock Owned: {self.stock_owned}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Total Asset Value: ${total_asset_value:.2f}")
        print(f"Initial Asset Value: ${self.initial_balance:.2f}")
        print(f"Return: {((total_asset_value / self.initial_balance) - 1) * 100:.2f}%")

        if self.trades:
            print("Last 5 Trades:")
            for trade in self.trades[-5:]:
                step, action, amount, price = trade
                print(f"  Step {step}: {action.upper()} {amount} shares at ${price:.2f}")
        print("-" * 30)

# --- Golden Cross Strategy ---
class GoldenCrossStrategy:
    def __init__(self):
        self.name = "GoldenCross"

    def generate_signals(self, df):
        signals = pd.DataFrame(index=df.index)
        price_col = 'adj_close' # Assuming adj_close is present after preparation
        signals['price'] = df[price_col]
        signals['signal'] = 0

        # Ensure sma20 and sma50 are present
        if 'sma20' not in df.columns or 'sma50' not in df.columns:
             raise ValueError("SMA20 and SMA50 must be calculated before generating signals.")

        # Buy signal
        signals.loc[(df['sma20'] > df['sma50']) & (df['sma20'].shift(1) <= df['sma50'].shift(1)), 'signal'] = 1
        # Sell signal
        signals.loc[(df['sma20'] < df['sma50']) & (df['sma20'].shift(1) >= df['sma50'].shift(1)), 'signal'] = -1

        signals['position'] = signals['signal'].replace(to_replace=0, method='ffill')
        signals['position'] = signals['position'].fillna(0)

        return signals

    def backtest(self, df, initial_balance=INITIAL_BALANCE, commission=COMMISSION):
        # Note: This backtest is simplified and may not perfectly match RL env trades/costs
        signals = self.generate_signals(df)

        portfolio = pd.DataFrame(index=signals.index)
        portfolio['holdings'] = 0.0
        portfolio['cash'] = float(initial_balance)
        portfolio['total'] = float(initial_balance)
        portfolio['position'] = 0 # Shares held

        for i in range(1, len(signals)):
             # Carry forward previous day's values
            portfolio.loc[signals.index[i], 'cash'] = portfolio.loc[signals.index[i-1], 'cash']
            portfolio.loc[signals.index[i], 'position'] = portfolio.loc[signals.index[i-1], 'position']

            price = signals.loc[signals.index[i], 'price']

            # Check for Buy Signal
            if signals.loc[signals.index[i], 'signal'] == 1 and portfolio.loc[signals.index[i-1], 'position'] == 0:
                # Simplified: Buy with all available cash (adjust for more realistic sizing if needed)
                shares_to_buy = int(portfolio.loc[signals.index[i-1], 'cash'] / (price * (1 + commission)))
                cost = shares_to_buy * price * (1 + commission)
                if shares_to_buy > 0:
                    portfolio.loc[signals.index[i], 'cash'] -= cost
                    portfolio.loc[signals.index[i], 'position'] = shares_to_buy

            # Check for Sell Signal
            elif signals.loc[signals.index[i], 'signal'] == -1 and portfolio.loc[signals.index[i-1], 'position'] > 0:
                 shares_to_sell = portfolio.loc[signals.index[i-1], 'position']
                 proceeds = shares_to_sell * price * (1 - commission)
                 portfolio.loc[signals.index[i], 'cash'] += proceeds
                 portfolio.loc[signals.index[i], 'position'] = 0

            # Update holdings and total value
            portfolio.loc[signals.index[i], 'holdings'] = portfolio.loc[signals.index[i], 'position'] * price
            portfolio.loc[signals.index[i], 'total'] = portfolio.loc[signals.index[i], 'holdings'] + portfolio.loc[signals.index[i], 'cash']

        portfolio['returns'] = portfolio['total'].pct_change().fillna(0)

        # Calculate metrics
        total_return = (portfolio['total'].iloc[-1] / initial_balance) - 1
        # Avoid division by zero if std is 0
        std_dev = portfolio['returns'].std()
        sharpe_ratio = (np.sqrt(252) * portfolio['returns'].mean() / std_dev) if std_dev != 0 else 0
        max_drawdown = (portfolio['total'] / portfolio['total'].cummax() - 1).min()

        logger.info(f"Golden Cross Strategy - Total Return: {total_return*100:.2f}%")
        logger.info(f"Golden Cross Strategy - Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Golden Cross Strategy - Max Drawdown: {max_drawdown*100:.2f}%")

        # Extract trades for logging/comparison (simplified)
        trades = signals[signals['signal'] != 0].copy()
        trades['action'] = trades['signal'].apply(lambda x: 'buy' if x == 1 else 'sell')

        return portfolio, trades # Return portfolio and simplified trade signals

    def plot_strategy(self, df, portfolio):
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

            price_col = 'adj_close'
            ax1.plot(df.index, df[price_col], label='Price')
            if 'sma20' in df.columns: ax1.plot(df.index, df['sma20'], label='SMA20', alpha=0.7)
            if 'sma50' in df.columns: ax1.plot(df.index, df['sma50'], label='SMA50', alpha=0.7)

            signals = self.generate_signals(df) # Regenerate to ensure alignment
            buy_signals = signals[signals['signal'] == 1]
            sell_signals = signals[signals['signal'] == -1]

            ax1.scatter(buy_signals.index, buy_signals['price'], marker='^', color='green', s=100, label='Buy Signal', zorder=5)
            ax1.scatter(sell_signals.index, sell_signals['price'], marker='v', color='red', s=100, label='Sell Signal', zorder=5)

            ax1.set_title('Golden Cross Strategy')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)

            if portfolio is not None:
                ax2.plot(portfolio.index, portfolio['total'], label='Portfolio Value')
                ax2.set_ylabel('Portfolio Value')
                ax2.set_xlabel('Date')
                ax2.legend()
                ax2.grid(True)

            plt.tight_layout()
            plt.savefig(f"{LOG_DIR}/golden_cross_strategy.png")
            plt.close(fig) # Close the figure to free memory
            logger.info(f"Saved Golden Cross plot to {LOG_DIR}/golden_cross_strategy.png")
        except Exception as e:
            logger.error(f"Failed to plot Golden Cross strategy: {e}")


# --- Data Handling ---
def _add_indicators(df):
    """Add technical indicators to the dataframe"""
    if df.empty:
        logger.warning("Dataframe is empty, cannot add indicators.")
        return df

    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    if price_col not in df.columns:
        raise ValueError(f"Price column ('Adj Close' or 'Close') not found in data. Columns: {df.columns}")
    df['adj_close'] = df[price_col] # Standardize price column name

    df['sma20'] = df['adj_close'].rolling(window=20).mean()
    df['sma50'] = df['adj_close'].rolling(window=50).mean()

    df['ema12'] = df['adj_close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['adj_close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # Ensure Volume is present
    if 'Volume' not in df.columns:
        logger.warning("Volume column not found, adding placeholder with zeros.")
        df['Volume'] = 0
    else:
        # Optional: Normalize volume (e.g., scale between 0 and 1)
        # df['Volume'] = (df['Volume'] - df['Volume'].min()) / (df['Volume'].max() - df['Volume'].min())
        pass # Keep volume as is for now

    # Drop rows with NaN values created by indicators
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_len - len(df)} rows due to indicator NaNs.")

    # Ensure required columns for environment exist
    required_env_cols = ['adj_close', 'Volume', 'sma20', 'sma50', 'macd', 'signal']
    missing_cols = [col for col in required_env_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after adding indicators: {missing_cols}")

    return df.reset_index() # Reset index after dropna, keeps original date if needed

def download_and_prepare_data(ticker, start_date, end_date, max_retries=3):
    """Download data, add indicators, and prepare for the environment"""
    logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
    data = None
    for attempt in range(max_retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if data.empty:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: Downloaded data is empty for {ticker}.")
                if attempt < max_retries - 1:
                    time.sleep(5) # Wait before retrying
                    continue
                else:
                    raise ValueError(f"Failed to download data for {ticker}: Data is empty after {max_retries} attempts.")

            logger.info(f"Successfully downloaded {len(data)} data points for {ticker}.")
            logger.debug(f"Downloaded columns: {data.columns}")
            logger.debug(f"Data sample:\n{data.head()}")

            # Basic data validation
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                 logger.warning(f"Downloaded data missing some standard columns. Available: {data.columns}")
                 # Allow proceeding if Close/Adj Close and Volume are present

            # Add indicators and handle NaNs
            data_with_indicators = _add_indicators(data.copy()) # Pass a copy

            if data_with_indicators.empty:
                raise ValueError("Data became empty after adding indicators and dropping NaNs.")

            logger.info(f"Data prepared with indicators. Final shape: {data_with_indicators.shape}")
            return data_with_indicators

        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{max_retries} failed to download or process data for {ticker}: {e}")
            if attempt < max_retries - 1:
                logger.info("Waiting 5 seconds before retry...")
                time.sleep(5)
            else:
                logger.error(f"Failed to get valid data for {ticker} after {max_retries} attempts.")
                # IMPORTANT: Raise the error instead of returning sample data
                raise e

    # This part should not be reached if logic is correct, but as a safeguard:
    raise RuntimeError("Unexpected exit from data download loop.")


# --- Model Training and Evaluation ---
def create_env_fn(data):
    """Function to create a new environment instance"""
    def _init():
        return StockTradingEnv(df=data.copy()) # Pass a copy to avoid modifying original df
    return _init

def train_model(model_class, env_fn, total_timesteps, model_name="Model"):
    """Train a RL model"""
    logger.info(f"--- Training {model_name} for {total_timesteps} timesteps ---")
    try:
        env = DummyVecEnv([env_fn]) # Wrap the environment creation function
        model = model_class("MlpPolicy", env, verbose=0, tensorboard_log=f"{LOG_DIR}/tensorboard/")
        callback = TrainingCallback()

        model.learn(total_timesteps=total_timesteps, callback=callback, tb_log_name=model_name)
        logger.info(f"Completed training {model_name}")
        return model

    except Exception as e:
        logger.error(f"Error during training {model_name}: {e}", exc_info=True)
        return None # Return None if training fails

def evaluate_model(model, env, model_name, num_episodes=10):
    all_rewards = []
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_rewards = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_rewards.append(reward)
        all_rewards.append(sum(episode_rewards))  # Sum the rewards for the episode

    mean_reward = np.mean(all_rewards)
    final_asset = env.balance + env.stock_owned * env._get_current_price()
    roi = (final_asset - env.initial_balance) / env.initial_balance * 100

    logger.info(f"{model_name} - Evaluation Complete")
    logger.info(f"{model_name} - Final Asset: ${float(final_asset):.2f}")
    logger.info(f"{model_name} - ROI: {float(roi):.2f}%")
    # Corrected line: use mean_reward instead of total_reward
    logger.info(f"{model_name} - Mean Reward (per episode): {mean_reward:.4f}")

    return final_asset, roi, mean_reward, len(env.trades) # Return mean reward


def plot_model_results(data, results, model_name="Model"):
    """Plot model evaluation results"""
    if results is None:
        logger.warning(f"No results to plot for {model_name}.")
        return

    logger.info(f"Plotting results for {model_name}")
    try:
        # Handle both dictionary and tuple inputs
        if isinstance(results, tuple):
            # Assume tuple contains (obs, reward, terminated, truncated, info)
            trades = results[4].get('trades', []) if len(results) > 4 and isinstance(results[4], dict) else []
            assets = results[4].get('assets', []) if len(results) > 4 and isinstance(results[4], dict) else []
        else:
            trades = results.get('trades', [])
            assets = results.get('assets', [])
        price_data = data['adj_close'] # Use prepared data with matching index

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

        # Plot price
        ax1.plot(price_data.index, price_data.values, label='Price', alpha=0.8)

        # Extract buy/sell points from actual trades during evaluation
        buy_indices = [trade[0] for trade in trades if trade[1] == 'buy']
        sell_indices = [trade[0] for trade in trades if trade[1] == 'sell']
        # Get prices at trade steps from the main dataframe using the index/step number
        buy_prices = [data.iloc[idx]['adj_close'] for idx in buy_indices if idx < len(data)]
        sell_prices = [data.iloc[idx]['adj_close'] for idx in sell_indices if idx < len(data)]
        # Match indices for plotting
        buy_plot_indices = [price_data.index[idx] for idx in buy_indices if idx < len(price_data)]
        sell_plot_indices = [price_data.index[idx] for idx in sell_indices if idx < len(price_data)]


        if buy_plot_indices:
            ax1.scatter(buy_plot_indices, buy_prices, marker='^', color='green', s=100, label='Buy', zorder=5)
        if sell_plot_indices:
            ax1.scatter(sell_plot_indices, sell_prices, marker='v', color='red', s=100, label='Sell', zorder=5)

        ax1.set_title(f'{model_name} Trading Strategy')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)

        # Plot asset value over time (use index from price data for x-axis)
        # Ensure assets list length matches data index length for plotting
        plot_assets = assets[:len(price_data.index)] # Adjust if lengths mismatch
        ax2.plot(price_data.index[:len(plot_assets)], plot_assets, label='Portfolio Value')
        ax2.set_title(f'{model_name} Portfolio Value')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plot_filename = f"{LOG_DIR}/{model_name}_evaluation.png"
        plt.savefig(plot_filename)
        plt.close(fig)
        logger.info(f"Saved evaluation plot to {plot_filename}")

    except Exception as e:
        logger.error(f"Failed to plot results for {model_name}: {e}", exc_info=True)


def compare_models(data_index, models_results):
    """Compare performance of different models and save comparison."""
    logger.info("--- Comparing Models ---")
    valid_results = {name: res for name, res in models_results.items() if res is not None}

    if not valid_results:
        logger.warning("No valid model results available for comparison.")
        return None

    # Plot Asset Curves
    try:
        plt.figure(figsize=(14, 7))
        for name, results in valid_results.items():
             # Handle both tuple and dict formats
             if isinstance(results, tuple):
                 assets = results[0]
                 roi = float(results[1].iloc[0]) if len(results) > 1 and isinstance(results[1], pd.Series) else (float(results[1]) if len(results) > 1 else 0)
             else:
                 assets = results.get('assets', [])
                 roi = float(results.get('roi', 0).iloc[0]) if isinstance(results.get('roi', 0), pd.Series) else float(results.get('roi', 0))
             
             # Ensure asset length matches data index for plotting
             plot_assets = assets[:len(data_index)]
             plot_index = data_index[:len(plot_assets)]
             plt.plot(plot_index, plot_assets, label=f'{name} (ROI: {roi:.2f}%)')

        plt.title('Model Comparison: Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)

        comp_plot_filename = f"{LOG_DIR}/model_comparison_portfolio.png"
        plt.savefig(comp_plot_filename)
        plt.close()
        logger.info(f"Saved comparison plot to {comp_plot_filename}")
    except Exception as e:
        logger.error(f"Failed to plot model comparison: {e}", exc_info=True)

    # Create and Save Comparison Table
    try:
        comparison_data = []
        for name, results in valid_results.items():
             # Handle both tuple and dict formats
             if isinstance(results, tuple):
                 comparison_data.append({
                     'Model': name,
                     'Final Asset ($)': float(results[0]),
                     'ROI (%)': float(results[1]),
                     'Total Reward': float(results[2]),
                     'Num Trades': float(results[3])
                 })
             else:
                 comparison_data.append({
                     'Model': name,
                     'Final Asset ($)': float(results.get('final_asset', float('nan')).iloc[0]) if isinstance(results.get('final_asset', float('nan')), pd.Series) else float(results.get('final_asset', float('nan'))),
                     'ROI (%)': float(results.get('roi', float('nan')).iloc[0]) if isinstance(results.get('roi', float('nan')), pd.Series) else float(results.get('roi', float('nan'))),
                     'Total Reward': float(results.get('total_reward', float('nan')).iloc[0]) if isinstance(results.get('total_reward', float('nan')), pd.Series) else float(results.get('total_reward', float('nan'))),
                     'Num Trades': len(results.get('trades', []))
                 })

        comparison_df = pd.DataFrame(comparison_data)
        logger.info(f"Model Comparison Table:\n{comparison_df.to_string()}")

        csv_filename = f"{LOG_DIR}/model_comparison.csv"
        comparison_df.to_csv(csv_filename, index=False)
        logger.info(f"Saved comparison CSV to {csv_filename}")
        return comparison_df

    except Exception as e:
        logger.error(f"Failed to create or save comparison CSV: {e}", exc_info=True)
        return None


# --- Main Execution ---
def main():
    logger.info("Starting revised trading bot execution...")

    # Download and prepare data (raises error if fails)
    try:
        data = download_and_prepare_data(TICKER, START_DATE, END_DATE)
    except Exception as e:
        logger.error(f"Fatal Error: Could not prepare data. Exiting. Error: {e}", exc_info=True)
        return # Exit if data preparation fails

    if data.empty:
         logger.error("Fatal Error: Data is empty after preparation. Exiting.")
         return

    # --- Run Golden Cross Strategy ---
    logger.info("--- Running Golden Cross Strategy ---")
    gc_strategy = GoldenCrossStrategy()
    try:
        gc_portfolio, gc_trades_signals = gc_strategy.backtest(data.copy(), initial_balance=INITIAL_BALANCE, commission=COMMISSION)
        gc_strategy.plot_strategy(data.copy(), gc_portfolio)
        # Prepare Golden Cross results for comparison table
        gc_results = {
            'assets': gc_portfolio['total'].values,
            'roi': (gc_portfolio['total'].iloc[-1] / INITIAL_BALANCE - 1) * 100 if INITIAL_BALANCE > 0 else 0,
            'final_asset': gc_portfolio['total'].iloc[-1],
            'total_reward': float('nan'), # Reward concept differs from RL
            'trades': gc_trades_signals # Using the simplified signal list here
        }
    except Exception as e:
        logger.error(f"Failed to run or plot Golden Cross strategy: {e}", exc_info=True)
        gc_results = None


    # --- Train and Evaluate RL Models ---
    all_results = {"GoldenCross": gc_results} if gc_results else {} # Start with GC if successful

    # Environment creation function using the prepared data
    env_creator = create_env_fn(data)

    models_to_train = {
        "PPO": PPO,
        "A2C": A2C,
        "DQN": DQN
    }

    for name, model_class in models_to_train.items():
        model = train_model(model_class, env_creator, total_timesteps=TRAINING_TIMESTEPS, model_name=name)

        if model: # If training was successful
            # Save the trained model
            try:
                 model.save(f"{LOG_DIR}/{name.lower()}_model")
                 logger.info(f"Saved trained {name} model.")
            except Exception as e:
                 logger.error(f"Failed to save {name} model: {e}")

            # Evaluate the model
            env = StockTradingEnv(data.copy())
            eval_results = evaluate_model(model, env, model_name=name)
            all_results[name] = eval_results # Store results even if None (for completeness)

            # Plot evaluation results if evaluation was successful
            if eval_results:
                plot_model_results(data.copy(), eval_results, model_name=name) # Use a copy of data
        else:
             logger.warning(f"Training failed for {name}, skipping evaluation and plotting.")
             all_results[name] = None # Mark as failed


    # --- Compare Models ---
    # Pass the index from the prepared data for plotting axes
    comparison_df = compare_models(data['Date'], all_results) # Assuming 'Date' is the index column after reset_index

    logger.info("Trading bot execution finished.")
    if comparison_df is not None:
        logger.info(f"\nFinal Comparison:\n{comparison_df.to_string()}")
    else:
        logger.warning("Model comparison could not be generated.")


if __name__ == "__main__":
    main()