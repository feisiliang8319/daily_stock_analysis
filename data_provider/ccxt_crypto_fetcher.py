# -*- coding: utf-8 -*-
"""
CCXTCryptoFetcher - 加密货币数据源（基于 CCXT，行业标准）

数据来源：通过 CCXT 直连交易所（默认 Kraken，Sol 已开户）
特点：
- 直连交易所原始数据，质量优于 yfinance（消费级聚合）
- 支持 OHLCV K线、ticker 实时价格
- 100+ 交易所统一接口，可灵活切换

定位：crypto 专用数据源，仅响应 BTC-USD 等加密货币代码，
其他代码（A股/港股/美股）由对应 fetcher 处理。
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .base import BaseFetcher
from .us_index_mapping import is_crypto_code

logger = logging.getLogger(__name__)

# Ticker symbol -> CCXT pair format
# yfinance 用 BTC-USD，CCXT 用 BTC/USD
TICKER_TO_CCXT_PAIR = {
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
    "SOL-USD": "SOL/USD",
    "XRP-USD": "XRP/USD",
    "ADA-USD": "ADA/USD",
    "DOGE-USD": "DOGE/USD",
    "AVAX-USD": "AVAX/USD",
    "DOT-USD": "DOT/USD",
    "LINK-USD": "LINK/USD",
    "MATIC-USD": "MATIC/USD",
    "UNI-USD": "UNI/USD",
    "ATOM-USD": "ATOM/USD",
    "LTC-USD": "LTC/USD",
    "NEAR-USD": "NEAR/USD",
    "APT-USD": "APT/USD",
    "ARB-USD": "ARB/USD",
    "OP-USD": "OP/USD",
    "SUI-USD": "SUI/USD",
    "BNB-USD": "BNB/USD",
}


class CCXTCryptoFetcher(BaseFetcher):
    """
    CCXT 加密货币数据源

    优先级最高（priority=0），仅处理 crypto 代码，其他代码返回空让链路继续。
    """

    name = "CCXTCryptoFetcher"
    priority = int(os.getenv("CCXT_PRIORITY", "0"))

    def __init__(self):
        super().__init__()
        self.exchange_id = os.getenv("CCXT_EXCHANGE", "kraken").lower()
        self._exchange = None

    def _get_exchange(self):
        """Lazy load exchange instance."""
        if self._exchange is None:
            try:
                import ccxt
                exchange_class = getattr(ccxt, self.exchange_id)
                self._exchange = exchange_class({
                    "enableRateLimit": True,
                    "timeout": 15000,
                })
                logger.info(f"[CCXT] Exchange initialized: {self.exchange_id}")
            except ImportError:
                logger.error("ccxt 未安装，pip install ccxt")
                raise
            except Exception as e:
                logger.error(f"[CCXT] Exchange init failed: {e}")
                raise
        return self._exchange

    def get_daily_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        获取加密货币日线数据

        Args:
            stock_code: 加密货币代码（BTC-USD, ETH-USD 等）
            start_date: 开始日期 YYYY-MM-DD（可选）
            end_date: 结束日期 YYYY-MM-DD（可选）
            days: 获取天数（默认 30）

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            返回空 DataFrame 表示该 fetcher 不支持此代码（让链路继续）
        """
        code = (stock_code or "").strip().upper()

        # 仅处理 crypto，其他代码返回空让链路继续
        if not is_crypto_code(code):
            return pd.DataFrame()

        pair = TICKER_TO_CCXT_PAIR.get(code)
        if not pair:
            # 未在映射表中的 crypto，尝试 BTC-USD → BTC/USD 通用转换
            base = code.replace("-USD", "")
            pair = f"{base}/USD"
            logger.warning(f"[CCXT] {code} 不在映射表，尝试通用格式: {pair}")

        try:
            ex = self._get_exchange()
            # 计算 since 时间戳
            if start_date:
                since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
                limit = days + 5
            else:
                since = int((datetime.utcnow() - timedelta(days=days + 5)).timestamp() * 1000)
                limit = days + 5

            ohlcv = ex.fetch_ohlcv(pair, timeframe="1d", since=since, limit=limit)
            if not ohlcv:
                logger.warning(f"[CCXT] {pair} 返回空数据")
                return pd.DataFrame()

            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
            df = df[["date", "open", "high", "low", "close", "volume"]]

            # Volume normalization: ccxt.fetch_ohlcv returns BASE currency
            # volume (e.g. BTC for BTC/USD) from a single exchange, whereas
            # YfinanceFetcher returns USD **notional** volume aggregated
            # across venues. Mixing both in the same ``stock_daily`` table
            # (CCXT primary + YF fallback) breaks downstream
            # volume_ratio_5d and volume_anomaly signals by ~6 orders of
            # magnitude. To keep the unit stable across source switches,
            # convert base volume to a USD-notional proxy.
            #
            # We deliberately scale every row by the **previous day's close**
            # rather than each row's own close. Reasoning:
            #
            # 1. ``data_provider.base.DataFetcherManager._calculate_indicators``
            #    computes ``volume_ratio = today_volume / mean(prev5_volume)``
            #    on the fetcher output. If we used each row's own close, the
            #    ratio would pick up a ``today_close / weighted_mean_prev5_close``
            #    bias factor that drifts ±5-10% on a trending day. Multiplying
            #    every row by the same constant cancels in the ratio and
            #    restores zero drift vs the raw base-volume ratio.
            #
            # 2. We use the previous day's close (``iloc[-2]``) instead of the
            #    last row's close (``iloc[-1]``) so multiple intraday fetches
            #    on the same UTC day all produce identical normalized values
            #    (today's bar is intraday-partial and its close moves; the
            #    previous day's close is closed and stable). DB upserts then
            #    stay deterministic across cron retries.
            #
            # Single-row batches (degenerate) fall back to that row's own
            # close so the normalization stays well-defined.
            if not df.empty:
                if len(df) >= 2:
                    scale = float(df["close"].iloc[-2])
                else:
                    scale = float(df["close"].iloc[-1])
                df["volume"] = (df["volume"].astype(float) * scale).round(2)

            # 过滤日期范围
            if end_date:
                df = df[df["date"] <= end_date]
            if start_date:
                df = df[df["date"] >= start_date]

            df = df.tail(days).reset_index(drop=True)
            logger.info(
                f"[CCXT] {pair} ({self.exchange_id}) 获取成功: {len(df)} 行 (volume normalized to USD notional)"
            )
            return df

        except Exception as e:
            logger.warning(f"[CCXT] {pair} 获取失败: {type(e).__name__}: {e}")
            return pd.DataFrame()

    def get_realtime_data(self, stock_code: str) -> dict:
        """获取实时 ticker 数据"""
        code = (stock_code or "").strip().upper()
        if not is_crypto_code(code):
            return {}

        pair = TICKER_TO_CCXT_PAIR.get(code) or f"{code.replace('-USD', '')}/USD"
        try:
            ex = self._get_exchange()
            ticker = ex.fetch_ticker(pair)
            # Prefer quoteVolume (USD notional) when the exchange reports it
            # so realtime volume stays consistent with the normalized daily
            # data path. Fall back to baseVolume * last.
            quote_volume = ticker.get("quoteVolume")
            if quote_volume is None:
                base_volume = ticker.get("baseVolume")
                last_price = ticker.get("last")
                if base_volume is not None and last_price is not None:
                    quote_volume = float(base_volume) * float(last_price)
            return {
                "code": code,
                "name": code,
                "current": ticker.get("last"),
                "open": ticker.get("open"),
                "high": ticker.get("high"),
                "low": ticker.get("low"),
                "volume": quote_volume,
                "change_pct": ticker.get("percentage"),
                "timestamp": ticker.get("timestamp"),
                "source": f"ccxt:{self.exchange_id}",
            }
        except Exception as e:
            logger.warning(f"[CCXT] {pair} ticker 获取失败: {e}")
            return {}

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        """加密货币名称（直接返回代码）"""
        code = (stock_code or "").strip().upper()
        if is_crypto_code(code):
            return code
        return None

    def _fetch_raw_data(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """BaseFetcher 抽象方法实现（委托给 get_daily_data）"""
        return self.get_daily_data(stock_code, start_date, end_date)

    def _normalize_data(
        self, df: pd.DataFrame, stock_code: str
    ) -> pd.DataFrame:
        """CCXT 数据已经是标准格式（date/open/high/low/close/volume）"""
        return df
