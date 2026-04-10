# -*- coding: utf-8 -*-
"""Unit tests for CCXTCryptoFetcher volume normalization.

CCXT's ``fetch_ohlcv`` returns **base-currency** volume (e.g. BTC for BTC/USD)
from a single exchange, while YfinanceFetcher returns **USD-notional** volume
aggregated across venues. Without normalization, mixing both sources in the
same ``stock_daily`` table breaks downstream volume_ratio_5d and
volume_anomaly signals by ~6 orders of magnitude.

These tests lock in the normalization contract:
- daily path: ``volume = base_volume * close``
- realtime path: prefer ``quoteVolume``, fall back to ``baseVolume * last``
"""

import sys
import unittest
from unittest.mock import MagicMock

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from data_provider.ccxt_crypto_fetcher import CCXTCryptoFetcher


class _FakeExchange:
    """Minimal ccxt exchange double returning deterministic OHLCV."""

    def __init__(self, ohlcv, ticker=None):
        self._ohlcv = ohlcv
        self._ticker = ticker or {}

    def fetch_ohlcv(self, pair, timeframe="1d", since=None, limit=None):
        return list(self._ohlcv)

    def fetch_ticker(self, pair):
        return dict(self._ticker)


class CCXTVolumeNormalizationTest(unittest.TestCase):
    def _fetcher_with_exchange(self, exchange) -> CCXTCryptoFetcher:
        fetcher = CCXTCryptoFetcher()
        fetcher._exchange = exchange  # bypass real ccxt init
        return fetcher

    def test_daily_volume_is_converted_to_usd_notional(self) -> None:
        """volume must equal base_volume * close for every row."""
        # [timestamp_ms, open, high, low, close, base_volume]
        ohlcv = [
            [1712102400000, 69000.0, 70500.0, 68900.0, 70000.0, 1000.0],  # 2024-04-03
            [1712188800000, 70000.0, 70200.0, 69500.0, 69800.0, 1500.0],  # 2024-04-04
            [1712275200000, 69800.0, 69900.0, 68000.0, 68500.0, 2000.0],  # 2024-04-05
        ]
        fetcher = self._fetcher_with_exchange(_FakeExchange(ohlcv))

        df = fetcher.get_daily_data("BTC-USD", days=3)

        self.assertEqual(len(df), 3)
        # Each row's volume must equal round(base * close, 2)
        expected = [1000.0 * 70000.0, 1500.0 * 69800.0, 2000.0 * 68500.0]
        actual = df["volume"].tolist()
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, round(want, 2), places=2)

        # Close prices must be untouched
        self.assertEqual(df["close"].tolist(), [70000.0, 69800.0, 68500.0])

    def test_daily_volume_preserves_rolling_ratio(self) -> None:
        """A constant volume * price product keeps volume_ratio_5d stable."""
        # Same base * close everywhere → normalized volume is constant,
        # so any rolling ratio must be exactly 1.0
        ohlcv = [
            [1712102400000 + i * 86400000, 100.0, 110.0, 90.0, 100.0, 50.0]
            for i in range(5)
        ]
        fetcher = self._fetcher_with_exchange(_FakeExchange(ohlcv))

        df = fetcher.get_daily_data("BTC-USD", days=5)

        self.assertEqual(len(df), 5)
        normalized = df["volume"].tolist()
        self.assertTrue(all(v == normalized[0] for v in normalized))
        # Rolling 5-day average ratio = volume[-1] / mean(volume) = 1.0
        rolling_ratio = normalized[-1] / (sum(normalized) / len(normalized))
        self.assertAlmostEqual(rolling_ratio, 1.0)

    def test_daily_empty_response_returns_empty_df(self) -> None:
        """Empty OHLCV response must not crash the normalization path."""
        fetcher = self._fetcher_with_exchange(_FakeExchange([]))
        df = fetcher.get_daily_data("BTC-USD", days=3)
        self.assertTrue(df.empty)

    def test_daily_non_crypto_code_short_circuits_before_exchange_call(self) -> None:
        """Non-crypto tickers must return an empty df without touching ccxt."""
        fetcher = self._fetcher_with_exchange(_FakeExchange([]))
        df = fetcher.get_daily_data("AAPL", days=3)
        self.assertTrue(df.empty)

    def test_realtime_prefers_quote_volume_when_exchange_reports_it(self) -> None:
        """When the exchange already returns quoteVolume, use it verbatim."""
        ticker = {
            "last": 70000.0,
            "open": 69000.0,
            "high": 70500.0,
            "low": 68900.0,
            "baseVolume": 1000.0,
            "quoteVolume": 12345678.90,  # independent value to prove precedence
            "percentage": 1.44,
            "timestamp": 1712102400000,
        }
        fetcher = self._fetcher_with_exchange(_FakeExchange([], ticker=ticker))

        data = fetcher.get_realtime_data("BTC-USD")

        self.assertEqual(data["volume"], 12345678.90)
        self.assertEqual(data["current"], 70000.0)
        self.assertTrue(data["source"].startswith("ccxt:"))

    def test_realtime_falls_back_to_base_volume_times_last(self) -> None:
        """Without quoteVolume, volume = baseVolume * last."""
        ticker = {
            "last": 70000.0,
            "open": 69000.0,
            "high": 70500.0,
            "low": 68900.0,
            "baseVolume": 1000.0,
            "quoteVolume": None,  # exchange doesn't report it
            "percentage": 1.44,
            "timestamp": 1712102400000,
        }
        fetcher = self._fetcher_with_exchange(_FakeExchange([], ticker=ticker))

        data = fetcher.get_realtime_data("BTC-USD")

        self.assertEqual(data["volume"], 70_000_000.0)  # 1000 * 70000

    def test_realtime_missing_both_volume_fields_returns_none(self) -> None:
        """No baseVolume and no quoteVolume → volume field is None, not crash."""
        ticker = {
            "last": 70000.0,
            "open": 69000.0,
            "high": 70500.0,
            "low": 68900.0,
            "baseVolume": None,
            "quoteVolume": None,
            "percentage": 0.0,
            "timestamp": 1712102400000,
        }
        fetcher = self._fetcher_with_exchange(_FakeExchange([], ticker=ticker))

        data = fetcher.get_realtime_data("BTC-USD")

        self.assertIsNone(data["volume"])
        self.assertEqual(data["current"], 70000.0)


if __name__ == "__main__":
    unittest.main()
