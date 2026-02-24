#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FRIDAY UNIFIED – Streamlit AI Trading Bot                ║
║   Scalping | Swing | Positional | BTST + All Advanced Features in One       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# IMPORTS (with fallbacks for optional packages)
# =============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import plotly.graph_objects as go
import datetime
import json
import time
import talib
import asyncio
import warnings
warnings.filterwarnings('ignore')
@st.cache_data(ttl=600)
def fetch_stock_data(ticker_symbol):
    try:
        # Pehle YahooQuery se koshish karein
        t = Ticker(ticker_symbol)
        data = t.history(period='1d', interval='1m')
        if not data.empty:
            # Agar data MultiIndex hai toh use clean karein
            if isinstance(data.index, pd.MultiIndex):
                data = data.xs(ticker_symbol)
            return data
    except Exception:
        pass
    
    # Agar YahooQuery fail ho toh yfinance fallback use karein
    return yf.download(ticker_symbol, period='1d', interval='1m')

# Optional imports – will be checked at runtime
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

try:
    from kiteconnect import KiteConnect
    ZERODHA_AVAILABLE = True
except ImportError:
    ZERODHA_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

# =============================================================================
# 1. AI MODEL – CustomizableBrain
# =============================================================================
class CustomizableBrain(nn.Module):
    def __init__(self, input_dim=20, model_type='lstm', hidden_size=256,
                 num_layers=2, dropout=0.2, use_attention=True):
        super().__init__()
        self.model_type = model_type
        self.use_attention = use_attention

        if model_type == 'simple':
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 3)
            )
        elif model_type == 'lstm':
            self.lstm = nn.LSTM(input_dim, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            if use_attention:
                self.attention = nn.MultiheadAttention(hidden_size, 4, batch_first=True)
            self.fc = nn.Linear(hidden_size, 3)
        elif model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=4, dim_feedforward=hidden_size*4
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.pos_encoding = nn.Parameter(torch.randn(1, 500, hidden_size))
            self.fc = nn.Linear(hidden_size, 3)
        elif model_type == 'cnn':
            self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(128, 3)

    def forward(self, x):
        if self.model_type == 'simple':
            return self.net(x[:, -1, :])
        elif self.model_type == 'lstm':
            lstm_out, _ = self.lstm(x)
            if self.use_attention:
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                out = attn_out[:, -1, :]
            else:
                out = lstm_out[:, -1, :]
            return self.fc(out)
        elif self.model_type == 'transformer':
            x = x + self.pos_encoding[:, :x.size(1), :]
            trans_out = self.transformer(x)
            return self.fc(trans_out[:, -1, :])
        elif self.model_type == 'cnn':
            x = x.permute(0, 2, 1)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

# =============================================================================
# 2. TECHNICAL INDICATORS – CustomizableTechnicalAnalysis
# =============================================================================
class CustomizableTechnicalAnalysis:
    @staticmethod
    def calculate(df, sma_periods=[20,50], ema_periods=[12,26], rsi_period=14,
                  rsi_oversold=30, rsi_overbought=70, macd_fast=12, macd_slow=26,
                  macd_signal=9, bb_period=20, bb_std=2, volume_ma_period=20,
                  atr_period=14, adx_period=14, custom_indicators=[]):
        for p in sma_periods:
            df[f'SMA_{p}'] = talib.SMA(df['Close'], timeperiod=p)
        for p in ema_periods:
            df[f'EMA_{p}'] = talib.EMA(df['Close'], timeperiod=p)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=rsi_period)
        df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = talib.MACD(
            df['Close'], fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal
        )
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(
            df['Close'], timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std
        )
        df['VOLUME_MA'] = df['Volume'].rolling(volume_ma_period).mean()
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_period)
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=adx_period)
        for func in custom_indicators:
            df = func(df)
        return df

# =============================================================================
# 3. RISK MANAGER – CustomizableRiskManager
# =============================================================================
class CustomizableRiskManager:
    def __init__(self, initial_capital=100000, max_position_pct=0.05,
                 max_portfolio_risk=0.02, max_daily_loss=0.03, max_weekly_loss=0.08,
                 max_drawdown=0.15, max_leverage=2.0, stop_loss_type='atr',
                 stop_loss_atr_multiple=2.0, stop_loss_percentage=0.05,
                 risk_reward_ratio=2.0, use_trailing_stop=False,
                 trailing_activation=0.03, trailing_distance=0.02):
        self.capital = initial_capital
        self.peak = initial_capital
        self.positions = {}
        self.daily_trades = []
        self.weekly_trades = []
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.max_daily_loss = max_daily_loss
        self.max_weekly_loss = max_weekly_loss
        self.max_drawdown = max_drawdown
        self.max_leverage = max_leverage
        self.stop_loss_type = stop_loss_type
        self.stop_loss_atr_multiple = stop_loss_atr_multiple
        self.stop_loss_percentage = stop_loss_percentage
        self.risk_reward_ratio = risk_reward_ratio
        self.use_trailing_stop = use_trailing_stop
        self.trailing_activation = trailing_activation
        self.trailing_distance = trailing_distance

    def check_trade(self, symbol, quantity, price):
        value = quantity * price
        pos_pct = value / self.capital
        if pos_pct > self.max_position_pct:
            return False, f"Position size {pos_pct:.1%} > limit {self.max_position_pct:.1%}"
        today_pnl = sum(t['pnl'] for t in self.daily_trades if t['date']==datetime.datetime.now().date())
        if today_pnl < 0 and abs(today_pnl)/self.capital > self.max_daily_loss:
            return False, "Daily loss limit exceeded"
        total_exposure = sum(p['size']*p['price'] for p in self.positions.values()) + value
        leverage = total_exposure / self.capital
        if leverage > self.max_leverage:
            return False, f"Leverage {leverage:.1f}x > limit {self.max_leverage}x"
        return True, "OK"

    def calculate_stop_loss(self, entry_price, atr=None, support=None):
        if self.stop_loss_type == 'percentage':
            return entry_price * (1 - self.stop_loss_percentage)
        elif self.stop_loss_type == 'atr' and atr:
            return entry_price - (atr * self.stop_loss_atr_multiple)
        elif self.stop_loss_type == 'support' and support:
            return min(entry_price*0.95, support)
        else:
            return entry_price * 0.95

    def calculate_take_profit(self, entry_price, stop_price):
        risk = entry_price - stop_price
        return entry_price + (risk * self.risk_reward_ratio)

    def update_pnl(self, trade):
        self.daily_trades.append(trade)
        self.weekly_trades.append(trade)
        self.capital += trade['pnl']
        if self.capital > self.peak:
            self.peak = self.capital

    def get_risk_metrics(self):
        return {
            'capital': self.capital,
            'peak': self.peak,
            'drawdown': (self.peak - self.capital)/self.peak,
            'leverage': sum(p['size']*p['price'] for p in self.positions.values())/self.capital if self.capital else 0,
            'daily_pnl': sum(t['pnl'] for t in self.daily_trades if t['date']==datetime.datetime.now().date())
        }

# =============================================================================
# 4. DATA SOURCE MANAGER – DataSourceManager
# ============================================================================
class DataSourceManager:
    def __init__(self, stocks_sources=['yfinance'], crypto_exchanges=[],
                 forex_sources=[], preferred_exchange='binance', api_keys={}):
        self.stocks_sources = stocks_sources
        self.yf = yf

    def get_stock_data(self, symbol, period='1mo', interval='1d'):
        # Ye line aapke naye fetch_stock_data function ko call karegi
        return fetch_stock_data(symbol)

    def get_crypto_price(self, symbol, exchange=None):
        if not exchange:
            exchange = self.preferred_exchange
        if exchange in getattr(self, 'ccxt_exchanges', {}):
            try:
                ticker = self.ccxt_exchanges[exchange].fetch_ticker(symbol)
                return ticker['last']
            except: pass
        return None

    def get_crypto_ohlcv(self, symbol, timeframe='1h', limit=100):
        if self.preferred_exchange in getattr(self, 'ccxt_exchanges', {}):
            try:
                ohlcv = self.ccxt_exchanges[self.preferred_exchange].fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
            except: pass
        return pd.DataFrame()

# =============================================================================
# 5. ALERT SYSTEM – AlertSystem (Enhanced with Telegram/Discord)
# =============================================================================
class AlertSystem:
    def __init__(self, email_alerts=False, sms_alerts=False, telegram_alerts=False,
                 voice_alerts=False, email='', telegram_bot_token='', telegram_chat_id='',
                 discord_webhook='', custom_handlers=[]):
        self.email_alerts = email_alerts
        self.sms_alerts = sms_alerts
        self.telegram_alerts = telegram_alerts
        self.voice_alerts = voice_alerts and VOICE_AVAILABLE
        self.email = email
        self.telegram_bot_token = telegram_bot_token
        self.telegram_chat_id = telegram_chat_id
        self.discord_webhook = discord_webhook
        self.custom_handlers = custom_handlers
        if self.voice_alerts:
            self.engine = pyttsx3.init()
        self.alert_conditions = []

    def add_condition(self, name, func):
        self.alert_conditions.append((name, func))

    def check_and_alert(self, data, analysis):
        msgs = []
        for name, func in self.alert_conditions:
            if func(data, analysis):
                msg = f"Alert: {name} triggered"
                msgs.append(msg)
                self._send_alert(msg)
        return msgs

    def _send_alert(self, message):
        if self.voice_alerts:
            self.engine.say(message)
            self.engine.runAndWait()
        if self.telegram_alerts and self.telegram_bot_token and REQUESTS_AVAILABLE:
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
            data = {'chat_id': self.telegram_chat_id, 'text': message}
            try:
                requests.post(url, data=data)
            except: pass
        if self.discord_webhook and REQUESTS_AVAILABLE:
            try:
                requests.post(self.discord_webhook, json={'content': message})
            except: pass

    def speak(self, text):
        if self.voice_alerts:
            self.engine.say(text)
            self.engine.runAndWait()

# =============================================================================
# 6. PORTFOLIO MANAGER – PortfolioManager
# =============================================================================
class PortfolioManager:
    def __init__(self, max_stocks=10, max_crypto=5, max_forex=3, rebalance_frequency='weekly',
                 rebalance_threshold=0.05, tax_loss_harvesting=True, short_term_limit=0.4,
                 max_sector_exposure=0.3, min_assets=5):
        self.max_stocks = max_stocks
        self.max_crypto = max_crypto
        self.max_forex = max_forex
        self.rebalance_frequency = rebalance_frequency
        self.rebalance_threshold = rebalance_threshold
        self.tax_loss_harvesting = tax_loss_harvesting
        self.short_term_limit = short_term_limit
        self.max_sector_exposure = max_sector_exposure
        self.min_assets = min_assets
        self.holdings = {}
        self.last_rebalance = datetime.datetime.now()

    def add_holding(self, symbol, quantity, price, sector='unknown'):
        if symbol in self.holdings:
            old = self.holdings[symbol]
            new_qty = old['quantity'] + quantity
            new_avg = (old['quantity']*old['avg_price'] + quantity*price) / new_qty
            self.holdings[symbol] = {'quantity': new_qty, 'avg_price': new_avg, 'sector': sector}
        else:
            self.holdings[symbol] = {'quantity': quantity, 'avg_price': price, 'sector': sector}

    def remove_holding(self, symbol, quantity):
        if symbol in self.holdings:
            if quantity >= self.holdings[symbol]['quantity']:
                del self.holdings[symbol]
            else:
                self.holdings[symbol]['quantity'] -= quantity

    def get_exposure(self):
        return sum(h['quantity']*h['avg_price'] for h in self.holdings.values())

    def get_sector_exposure(self):
        sec = {}
        for sym, h in self.holdings.items():
            sec[h['sector']] = sec.get(h['sector'],0) + h['quantity']*h['avg_price']
        return sec

# =============================================================================
# 7. VOICE ASSISTANT – VoiceAssistant
# =============================================================================
class VoiceAssistant:
    def __init__(self, language='hi-IN', enabled=True):
        self.enabled = enabled and VOICE_AVAILABLE
        self.language = language
        if self.enabled:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.engine = pyttsx3.init()

    def listen(self):
        if not self.enabled:
            return None
        with self.microphone as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
        try:
            text = self.recognizer.recognize_google(audio, language=self.language)
            return text
        except:
            return None

    def speak(self, text):
        if self.enabled:
            self.engine.say(text)
            self.engine.runAndWait()

    def process_command(self, text):
        text_low = text.lower()
        if 'price' in text_low or 'bhav' in text_low:
            words = text.split()
            for w in words:
                if w.isalpha() and len(w)<=5:
                    return {'intent':'price','symbol':w.upper()}
        elif 'buy' in text_low or 'kharido' in text_low:
            return {'intent':'buy'}
        elif 'sell' in text_low or 'becho' in text_low:
            return {'intent':'sell'}
        return {'intent':'unknown'}

# =============================================================================
# 8. PERFORMANCE OPTIMIZER – PerformanceOptimizer
# =============================================================================
class PerformanceOptimizer:
    def __init__(self, mode='balanced', cache_size=1000, cache_ttl=60,
                 use_multithreading=True, max_workers=4, batch_size=100,
                 max_memory_mb=1024):
        self.mode = mode
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.use_multithreading = use_multithreading
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.cache = {}
        self.cache_timestamps = {}

    def get_cached(self, key):
        if key in self.cache and (time.time() - self.cache_timestamps[key]) < self.cache_ttl:
            return self.cache[key]
        return None

    def set_cache(self, key, value):
        if len(self.cache) >= self.cache_size:
            oldest = min(self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k])
            del self.cache[oldest]
            del self.cache_timestamps[oldest]
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()

# =============================================================================
# 9. CUSTOM FEATURE REGISTRY
# =============================================================================
class CustomFeatureRegistry:
    def __init__(self):
        self.pre_analysis_hooks = []
        self.post_analysis_hooks = []
        self.pre_trade_hooks = []
        self.post_trade_hooks = []
        self.custom_indicators = []
        self.custom_strategies = {}

# =============================================================================
# 10. STYLE CONFIGURATIONS
# =============================================================================
scalping_config = {
    'model': {'type':'cnn','hidden_size':128,'num_layers':2,'dropout':0.1,'use_attention':False},
    'technical': {'sma_periods':[5,10,20],'ema_periods':[3,5,8],'rsi_period':7,
                  'rsi_oversold':20,'rsi_overbought':80,'macd_fast':5,'macd_slow':13,
                  'macd_signal':4,'bb_period':10,'bb_std':1.5,'volume_ma_period':5,'atr_period':7},
    'risk': {'initial_capital':100000,'max_position_pct':0.02,'max_portfolio_risk':0.005,
             'max_daily_loss':0.02,'max_drawdown':0.05,'max_leverage':1.0,
             'stop_loss_type':'atr','stop_loss_atr_multiple':0.5,'stop_loss_percentage':0.002,
             'risk_reward_ratio':1.2,'use_trailing_stop':True,'trailing_activation':0.003,
             'trailing_distance':0.001},
    'performance': {'mode':'fast','cache_size':5000,'cache_ttl':5,'use_multithreading':True,
                    'max_workers':8,'batch_size':10}
}
swing_config = {
    'model': {'type':'lstm','hidden_size':256,'num_layers':3,'dropout':0.2,'use_attention':True},
    'technical': {'sma_periods':[20,50,100],'ema_periods':[9,21,55],'rsi_period':14,
                  'macd_fast':12,'macd_slow':26,'macd_signal':9,'bb_period':20,'bb_std':2,
                  'volume_ma_period':20,'atr_period':14,'adx_period':14},
    'risk': {'max_position_pct':0.05,'max_daily_loss':0.03,'max_weekly_loss':0.08,
             'max_drawdown':0.15,'max_leverage':1.5,'stop_loss_type':'atr',
             'stop_loss_atr_multiple':1.5,'risk_reward_ratio':2.5,
             'use_trailing_stop':True,'trailing_activation':0.05,'trailing_distance':0.03},
    'performance': {'mode':'balanced','cache_ttl':300}
}
positional_config = {
    'model': {'type':'transformer','hidden_size':512,'num_layers':6,'dropout':0.3,'use_attention':True},
    'technical': {'sma_periods':[50,100,200],'ema_periods':[21,55,200],'rsi_period':21,
                  'rsi_oversold':25,'rsi_overbought':75,'bb_period':50,'bb_std':2.5,
                  'volume_ma_period':50,'atr_period':21,'adx_period':21},
    'risk': {'max_position_pct':0.10,'max_portfolio_risk':0.05,'max_daily_loss':0.05,
             'max_weekly_loss':0.10,'max_drawdown':0.25,'max_leverage':1.0,
             'stop_loss_type':'percentage','stop_loss_percentage':0.08,
             'risk_reward_ratio':3.0,'use_trailing_stop':True,'trailing_activation':0.15,
             'trailing_distance':0.07},
    'performance': {'mode':'accurate','cache_ttl':3600,'use_multithreading':False}
}
btst_config = {
    'model': {'type':'lstm','hidden_size':256,'num_layers':3,'dropout':0.2,'use_attention':True},
    'technical': {'sma_periods':[10,20,50],'ema_periods':[9,21],'rsi_period':14,
                  'macd_fast':12,'macd_slow':26,'macd_signal':9,'bb_period':20,'bb_std':2,
                  'volume_ma_period':20,'atr_period':14},
    'risk': {'max_position_pct':0.04,'max_portfolio_risk':0.015,'max_daily_loss':0.03,
             'max_drawdown':0.10,'max_leverage':1.0,'stop_loss_type':'percentage',
             'stop_loss_percentage':0.015,'risk_reward_ratio':1.5},
    'performance': {'mode':'balanced','cache_ttl':3600}
}

# =============================================================================
# 11. STRATEGY LIBRARIES
# =============================================================================
class ScalpingStrategies:
    @staticmethod
    def momentum_scalp(data, params=None):
        if params is None: params={'momentum_period':3}
        if len(data) < 4: return {'buy':False,'sell':False}
        mom = (data['Close'].iloc[-1]/data['Close'].iloc[-3]-1)*100
        rsi = data['RSI'].iloc[-1] if 'RSI' in data else 50
        if mom > 0.5 and rsi < 70: return {'buy':True,'strength':mom}
        if mom < -0.5 and rsi > 30: return {'sell':True,'strength':abs(mom)}
        return {'buy':False,'sell':False}
    @staticmethod
    def order_flow_scalp(data, params=None):
        if params is None: params={'vol_thresh':1.5}
        if len(data)<5: return {'buy':False,'sell':False}
        vol = data['Volume'].iloc[-1]
        avg = data['Volume'].rolling(5).mean().iloc[-1]
        if vol > avg*params['vol_thresh']:
            if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                return {'buy':True,'strength':vol/avg}
            elif data['Close'].iloc[-1] < data['Close'].iloc[-2]:
                return {'sell':True,'strength':vol/avg}
        return {'buy':False,'sell':False}

class SwingStrategies:
    @staticmethod
    def trend_swing(data, params=None):
        if params is None: params={'fast':9,'slow':21,'adx_min':25}
        if len(data) < params['slow']: return {'buy':False,'sell':False}
        f = data[f'EMA_{params["fast"]}'].iloc[-1]
        s = data[f'EMA_{params["slow"]}'].iloc[-1]
        pf = data[f'EMA_{params["fast"]}'].iloc[-2]
        ps = data[f'EMA_{params["slow"]}'].iloc[-2]
        adx = data['ADX'].iloc[-1] if 'ADX' in data else 0
        if pf <= ps and f > s and adx > params['adx_min']:
            return {'buy':True,'strength':adx/50}
        if pf >= ps and f < s and adx > params['adx_min']:
            return {'sell':True,'strength':adx/50}
        return {'buy':False,'sell':False}
    @staticmethod
    def pullback_swing(data, params=None):
        if params is None: params={'ma':20,'pct':0.02}
        if len(data) < params['ma']: return {'buy':False,'sell':False}
        ma = data[f'SMA_{params["ma"]}'].iloc[-1]
        close = data['Close'].iloc[-1]
        rsi = data['RSI'].iloc[-1] if 'RSI' in data else 50
        if close < ma and close > ma*0.98 and rsi > 40:
            return {'buy':True,'strength':(ma-close)/ma*100}
        if close > ma and close < ma*1.02 and rsi < 60:
            return {'sell':True,'strength':(close-ma)/ma*100}
        return {'buy':False,'sell':False}

class PositionalStrategies:
    @staticmethod
    def macro_trend(data, params=None):
        if params is None: params={'long':200,'short':50}
        if len(data) < params['long']: return {'buy':False,'sell':False}
        ma_long = data[f'SMA_{params["long"]}'].iloc[-1]
        ma_short = data[f'SMA_{params["short"]}'].iloc[-1]
        price = data['Close'].iloc[-1]
        if price > ma_long and ma_short > ma_long and ma_short > data[f'SMA_{params["short"]}'].iloc[-5]:
            return {'buy':True,'strength':(price-ma_long)/ma_long}
        if price < ma_long and ma_short < ma_long and ma_short < data[f'SMA_{params["short"]}'].iloc[-5]:
            return {'sell':True,'strength':(ma_long-price)/ma_long}
        return {'buy':False,'sell':False}

class BTSTStrategies:
    @staticmethod
    def gap_prediction(data, params=None):
        if params is None: params={'lookback':20}
        if len(data) < params['lookback']: return {'buy':False,'sell':False}
        close = data['Close'].iloc[-1]
        prev = data['Close'].iloc[-2]
        vol = data['Volume'].iloc[-1]
        avg_vol = data['Volume'].rolling(params['lookback']).mean().iloc[-1]
        gaps = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        avg_gap = gaps.mean()
        std_gap = gaps.std()
        if close > prev*1.01 and vol > avg_vol*1.2 and avg_gap > 0:
            exp = avg_gap + std_gap*0.5
            if exp > 0.005:
                return {'buy':True,'strength':min(exp*100,2)}
        elif close < prev*0.99 and vol > avg_vol*1.2 and avg_gap < 0:
            exp = avg_gap - std_gap*0.5
            if exp < -0.005:
                return {'sell':True,'strength':min(abs(exp)*100,2)}
        return {'buy':False,'sell':False}

# =============================================================================
# 12. SCALPING EXECUTOR
# =============================================================================
class ScalpingExecutor:
    def __init__(self, friday):
        self.friday = friday
    async def watch_ticker(self, symbol, interval='1m'):
        while True:
            try:
                df = self.friday.data_mgr.get_stock_data(symbol, period='1d', interval=interval)
                if df.empty:
                    await asyncio.sleep(1)
                    continue
                signal = ScalpingStrategies.momentum_scalp(df)
                if signal.get('buy') or signal.get('sell'):
                    price = df['Close'].iloc[-1]
                    qty = int(self.friday.risk.capital * self.friday.risk.max_position_pct / price)
                    if qty < 1:
                        continue
                    ok, _ = self.friday.risk.check_trade(symbol, qty, price)
                    if ok:
                        action = 'BUY' if signal.get('buy') else 'SELL'
                        self.friday.execute_trade(symbol, action, qty, price)
                        self.friday.alerts.speak(f"Scalp {action} {symbol}")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Scalping error: {e}")
                await asyncio.sleep(1)

# =============================================================================
# 13. BTST EXECUTOR
# =============================================================================
class BTSTExecutor:
    def __init__(self, friday):
        self.friday = friday
        self.positions = {}
    def scan_opportunities(self, watchlist):
        results = []
        for sym in watchlist:
            df = self.friday.data_mgr.get_stock_data(sym, period='5d', interval='1d')
            if df.empty:
                continue
            sig = BTSTStrategies.gap_prediction(df)
            if sig.get('buy') or sig.get('sell'):
                results.append({
                    'symbol': sym,
                    'action': 'BUY' if sig.get('buy') else 'SELL',
                    'strength': sig.get('strength',0),
                    'price': df['Close'].iloc[-1]
                })
        return sorted(results, key=lambda x: x['strength'], reverse=True)
    def execute_btst(self, symbol, action, quantity, price):
        ok, _ = self.friday.risk.check_trade(symbol, quantity, price)
        if not ok:
            return False
        self.friday.execute_trade(symbol, action, quantity, price)
        self.positions[symbol] = {'entry': price, 'qty': quantity, 'action': action}
        return True
    def exit_positions(self):
        for sym, pos in list(self.positions.items()):
            df = self.friday.data_mgr.get_stock_data(sym, period='1d', interval='1d')
            if not df.empty:
                current = df['Close'].iloc[-1]
                if pos['action'] == 'BUY' and current >= pos['entry']*1.02:  # 2% target
                    self.friday.execute_trade(sym, 'SELL', pos['qty'], current)
                    del self.positions[sym]
                elif pos['action'] == 'SELL' and current <= pos['entry']*0.98:
                    self.friday.execute_trade(sym, 'BUY', pos['qty'], current)
                    del self.positions[sym]

# =============================================================================
# 14. FRIDAY UNIFIED – Main Orchestrator
# =============================================================================
class FridayUnified:
    def __init__(self, trading_style='swing', config_overrides={}):
        self.trading_style = trading_style
        # Load base config
        if trading_style == 'scalping':
            self.config = scalping_config.copy()
        elif trading_style == 'swing':
            self.config = swing_config.copy()
        elif trading_style == 'positional':
            self.config = positional_config.copy()
        elif trading_style == 'btst':
            self.config = btst_config.copy()
        else:
            self.config = swing_config.copy()
        # Apply overrides
        for k,v in config_overrides.items():
            if k in self.config and isinstance(v, dict):
                self.config[k].update(v)
            else:
                self.config[k] = v

        # Initialize components
        self.data_mgr = DataSourceManager()
        self.risk = CustomizableRiskManager(**self.config.get('risk', {}))
        self.alerts = AlertSystem()
        self.portfolio = PortfolioManager()
        self.optimizer = PerformanceOptimizer(**self.config.get('performance', {}))
        self.registry = CustomFeatureRegistry()
        self.voice = VoiceAssistant(enabled=False)  # Disabled by default in Streamlit

        # Initialize AI model (lazy)
        self.model = None
        self.model_loaded = False

        # Executors
        self.scalper = ScalpingExecutor(self)
        self.btst_executor = BTSTExecutor(self)

    def load_model(self, input_dim=20):
        if not self.model_loaded:
            model_cfg = self.config.get('model', {})
            self.model = CustomizableBrain(
                input_dim=input_dim,
                **model_cfg
            )
            self.model_loaded = True

    def prepare_features(self, df):
        # Simple feature engineering: use OHLCV + indicators
        features = []
        for col in ['Open','High','Low','Close','Volume']:
            if col in df.columns:
                features.append(df[col].values[-20:])  # last 20 periods
        # Add some indicators if available
        for ind in ['RSI','MACD','ATR','ADX']:
            if ind in df.columns:
                features.append(df[ind].values[-20:])
        # Pad if necessary
        max_len = max(len(f) for f in features) if features else 20
        padded = []
        for f in features:
            if len(f) < max_len:
                f = np.pad(f, (max_len-len(f),0), constant_values=0)
            padded.append(f[-max_len:])
        if not padded:
            return np.zeros((1,20,5))  # dummy
        feature_array = np.array(padded).T  # shape (time, features)
        # Add batch dimension
        return torch.tensor(feature_array).unsqueeze(0).float()

    def predict_signal(self, df):
        if df.empty or len(df) < 20:
            return {'buy':False,'sell':False,'confidence':0}
        self.load_model(input_dim=len(df.columns))
        features = self.prepare_features(df)
        with torch.no_grad():
            output = self.model(features)
            probs = F.softmax(output, dim=1).numpy()[0]
        # Assume classes: 0=hold, 1=buy, 2=sell
        buy_conf = probs[1]
        sell_conf = probs[2]
        if buy_conf > 0.6 and buy_conf > sell_conf:
            return {'buy':True,'sell':False,'confidence':buy_conf}
        elif sell_conf > 0.6 and sell_conf > buy_conf:
            return {'buy':False,'sell':True,'confidence':sell_conf}
        else:
            return {'buy':False,'sell':False,'confidence':max(buy_conf,sell_conf)}

    def execute_trade(self, symbol, action, quantity, price):
        # Simulate trade execution
        trade = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'pnl': 0,
            'date': datetime.datetime.now().date()
        }
        # Update portfolio
        if action == 'BUY':
            self.portfolio.add_holding(symbol, quantity, price)
            self.risk.positions[symbol] = {'size': quantity, 'price': price}
        elif action == 'SELL':
            # Simplified: assume we sell entire position
            if symbol in self.portfolio.holdings:
                avg = self.portfolio.holdings[symbol]['avg_price']
                trade['pnl'] = (price - avg) * quantity
                self.portfolio.remove_holding(symbol, quantity)
                if symbol in self.risk.positions:
                    del self.risk.positions[symbol]
        self.risk.update_pnl(trade)
        return trade

# =============================================================================
# 15. STREAMLIT APP
# =============================================================================
def main():
    st.set_page_config(page_title="FRIDAY Unified Trading AI", layout="wide")
    st.title("🤖 FRIDAY – AI Trading Assistant")
    st.markdown("Scalping | Swing | Positional | BTST")

    # Sidebar configuration
    st.sidebar.header("Trading Configuration")
    trading_style = st.sidebar.selectbox("Trading Style", ["swing", "scalping", "positional", "btst"], index=0)
    symbol = st.sidebar.text_input("Symbol", "RELIANCE.NS").upper()
    period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=0)
    interval = st.sidebar.selectbox("Interval", ["1d", "1h", "15m", "5m", "1m"], index=0)

        # Advanced options expander
    with st.sidebar.expander("Advanced Options"):
        use_ai_model = st.checkbox("Use AI Model (PyTorch)")
        enable_voice = st.checkbox("Enable Voice Assistant")
        telegram_token = st.sidebar.text_input("Telegram Bot Token")
        telegram_chat = st.sidebar.text_input("Telegram Chat ID")

    # --- AUTO-PILOT SETTINGS ---
    st.sidebar.markdown("---")
    st.sidebar.header("🔄 Auto-Pilot")
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)
    refresh_secs = st.sidebar.slider("Refresh Interval (Secs)", 30, 300, 60)
    
    if 'trade_logs' not in st.session_state:
        st.session_state.trade_logs = []

    # Initialize FRIDAY instance (cached to avoid re-init on each rerun)
    @st.cache_resource
    def init_friday(style):
        return FridayUnified(trading_style=style)

    friday = init_friday(trading_style)

    # Update voice and alert settings
    friday.voice.enabled = enable_voice
    if telegram_token and telegram_chat:
        friday.alerts.telegram_alerts = True
        friday.alerts.telegram_bot_token = telegram_token
        friday.alerts.telegram_chat_id = telegram_chat

    # Main area: data fetching and analysis
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📈 {symbol} - Analysis")
        # Naya Data Fetcher
        @st.cache_data(ttl=600)
        def fetch_data(sym, period, interval):
            try:
                return fetch_stock_data(sym, period, interval)
            except Exception as e:
                st.error(f"Data Error: {str(e)}")
                return pd.DataFrame()

        # Is line se niche tak sab 8 spaces aage hain
        df = fetch_data(symbol, period, interval)
        if df.empty:
            st.warning(f"No data found for symbol {symbol}")
            return

        tech_params = friday.config.get('technical_analysis', {})
        df = CustomizableTechnicalAnalysis.calculate_all(df, tech_params)

        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Market Data'
        )])

        for col in df.columns:
            if col.startswith('SMA_') or col.startswith('EMA_'):
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))

        if 'BB_UPPER' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], name='BB Upper', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], name='BB Lower', line=dict(dash='dash')))

        fig.update_layout(xaxis_rangeslider_visible=False, height=600)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Latest Data"):
            st.dataframe(df.tail(10))

    
    with col2:
        st.subheader("📊 Analysis & Signals")
        
        # Run strategy button
        if st.button("Generate Signal", type="primary"):
            with st.spinner("Analyzing..."):
                # Strategy selection
                if trading_style == 'scalping':
                    signal = ScalpingStrategies.momentum_scalp(df)
                elif trading_style == 'swing':
                    signal = SwingStrategies.trend_swing(df)
                elif trading_style == 'positional':
                    signal = PositionalStrategies.macro_trend(df)
                elif trading_style == 'btst':
                    signal = BTSTStrategies.gap_prediction(df)
                else:
                    signal = {'buy': False, 'sell': False}

                # Signal metadata
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                action = "BUY" if signal.get('buy') else "SELL"

                # Sound alert
                st.markdown('<audio autoplay><source src="https://www.soundjay.com/buttons/beep-01a.mp3"></audio>', unsafe_allow_html=True)

                # Log entry update
                if not st.session_state.trade_logs or st.session_state.trade_logs[0]['Time'] != current_time:
                    st.session_state.trade_logs.insert(0, {"Time": current_time, "Action": action, "Symbol": symbol})

                # Result display
                if signal.get('buy'):
                    st.success(f"🚀 **BUY SIGNAL**")
                elif signal.get('sell'):
                    st.error(f"🔻 **SELL SIGNAL**")
                else:
                    st.info("No clear signal - HOLD")

        # Risk metrics
        metrics = friday.risk.get_risk_metrics()
        st.metric("Capital", f"₹{metrics['capital']:,}")
        st.metric("Drawdown", f"{metrics['drawdown']*100:.2f}%")

# --- ENTRY POINT ---
if __name__ == "__main__":
    # Sidebar logs
    st.sidebar.subheader("📜 Live Logs")
    if st.session_state.trade_logs:
        st.sidebar.table(pd.DataFrame(st.session_state.trade_logs).head(5))
