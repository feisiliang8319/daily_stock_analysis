# -*- coding: utf-8 -*-
"""Unit tests for CCXTCryptoFetcher volume normalization.

CCXT's ``fetch_ohlcv`` returns **base-currency** volume (e.g. BTC for BTC/USD)
from a single exchange, while YfinanceFetcher returns **USD-notional** volume
aggregated across venues. Without normalization, mixing both sources in the
same ``stock_daily`` table breaks downstream volume_ratio_5d and
volume_anomaly signals by ~6 orders of magnitude.

These tests lock in the normalization contract:
- daily path: every row scaled by the previous day's close
  (``df.close.iloc[-2]`` when len >= 2, else ``iloc[-1]``).
  This makes the rolling ``volume_ratio`` computed downstream by
  ``DataFetcherManager._calculate_indicators`` mathematically equivalent
  to the raw base-volume ratio (zero drift), and keeps the value stable
  across multiple intraday cron dispatches because the previous day's
  close is closed and immutable.
- realtime path: prefer ``quoteVolume``, fall back to ``baseVolume * last``.
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

    def test_daily_volume_is_converted_to_usd_notional_with_prev_close(self) -> None:
        """Every row's volume must equal base_volume * previous-day close.

        We pick the second-to-last row's close (``iloc[-2]``) so that
        intraday cron retries on the same UTC day produce identical
        normalized values (the last bar is intraday-partial; its close
        moves; the previous day's close is closed and stable).
        """
        # [timestamp_ms, open, high, low, close, base_volume]
        ohlcv = [
            [1712102400000, 69000.0, 70500.0, 68900.0, 70000.0, 1000.0],  # 2024-04-03
            [1712188800000, 70000.0, 70200.0, 69500.0, 69800.0, 1500.0],  # 2024-04-04 (T-1, scale source)
            [1712275200000, 69800.0, 69900.0, 68000.0, 68500.0, 2000.0],  # 2024-04-05 (today, intraday-partial)
        ]
        fetcher = self._fetcher_with_exchange(_FakeExchange(ohlcv))

        df = fetcher.get_daily_data("BTC-USD", days=3)

        self.assertEqual(len(df), 3)
        # All three rows must be scaled by the SAME constant: prev close = 69800.
        scale = 69800.0
        expected = [1000.0 * scale, 1500.0 * scale, 2000.0 * scale]
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

    def test_volume_ratio_drift_is_zero_under_trending_close(self) -> None:
        """Critical regression guard: rolling volume_ratio computed on the
        normalized column must be EXACTLY equal to the rolling ratio computed
        on the raw base-currency column, even when close is trending hard.

        This is the property that justifies the prev-close constant scaling
        choice over per-row close scaling. With per-row scaling, a trending
        close would inject a few-percent bias into volume_ratio downstream;
        with constant scaling that bias is mathematically zero.
        """
        # 6 days, close trending +2% per day, base volume varies independently
        base_volumes = [1000.0, 1200.0, 800.0, 1500.0, 900.0, 1100.0]
        closes = [60000.0, 61200.0, 62424.0, 63672.5, 64946.0, 66244.9]
        ohlcv = [
            [1712102400000 + i * 86400000, c * 0.99, c * 1.01, c * 0.98, c, v]
            for i, (c, v) in enumerate(zip(closes, base_volumes))
        ]
        fetcher = self._fetcher_with_exchange(_FakeExchange(ohlcv))

        df = fetcher.get_daily_data("BTC-USD", days=6)
        self.assertEqual(len(df), 6)

        # Replicate _calculate_indicators rolling ratio on the normalized
        # volume column the fetcher returned.
        normalized = df["volume"].tolist()
        rolling_ratio_normalized = normalized[-1] / (sum(normalized[-6:-1]) / 5)

        # Same rolling ratio computed on the original raw base volumes.
        rolling_ratio_raw = base_volumes[-1] / (sum(base_volumes[-6:-1]) / 5)

        # They MUST be exactly equal — no drift from the close trend.
        self.assertAlmostEqual(rolling_ratio_normalized, rolling_ratio_raw, places=10)

    def test_daily_volume_stable_across_intraday_dispatches(self) -> None:
        """Two fetches on the same UTC day with a moving intraday close on
        the last bar must produce identical normalized volumes for every row.

        This is the property that justifies using df.close.iloc[-2] (closed
        previous day) instead of df.close.iloc[-1] (intraday-partial today).
        """
        # First fetch: today's bar close = 65000
        ohlcv_morning = [
            [1712102400000, 60000.0, 60500.0, 59500.0, 60000.0, 1000.0],
            [1712188800000, 60000.0, 60500.0, 59500.0, 60500.0, 1100.0],  # T-1
            [1712275200000, 60500.0, 65500.0, 60000.0, 65000.0, 1200.0],  # today @ morning
        ]
        # Second fetch: same day, today's close moved to 67000 (intraday)
        ohlcv_afternoon = [
            [1712102400000, 60000.0, 60500.0, 59500.0, 60000.0, 1000.0],
            [1712188800000, 60000.0, 60500.0, 59500.0, 60500.0, 1100.0],  # T-1 (unchanged)
            [1712275200000, 60500.0, 67500.0, 60000.0, 67000.0, 1500.0],  # today @ afternoon
        ]
        fetcher_morning = self._fetcher_with_exchange(_FakeExchange(ohlcv_morning))
        fetcher_afternoon = self._fetcher_with_exchange(_FakeExchange(ohlcv_afternoon))

        df_morning = fetcher_morning.get_daily_data("BTC-USD", days=3)
        df_afternoon = fetcher_afternoon.get_daily_data("BTC-USD", days=3)

        # Historical rows (date < today) must match exactly across the two
        # dispatches because the scale factor (T-1 close = 60500) is the
        # same in both fetches.
        scale = 60500.0
        self.assertAlmostEqual(df_morning["volume"].iloc[0], 1000.0 * scale, places=2)
        self.assertAlmostEqual(df_afternoon["volume"].iloc[0], 1000.0 * scale, places=2)
        self.assertAlmostEqual(df_morning["volume"].iloc[1], 1100.0 * scale, places=2)
        self.assertAlmostEqual(df_afternoon["volume"].iloc[1], 1100.0 * scale, places=2)
        # Today's normalized volume only differs because the underlying base
        # volume increased through the day, NOT because of close drift.
        self.assertAlmostEqual(df_morning["volume"].iloc[2], 1200.0 * scale, places=2)
        self.assertAlmostEqual(df_afternoon["volume"].iloc[2], 1500.0 * scale, places=2)

    def test_daily_volume_single_row_falls_back_to_own_close(self) -> None:
        """Edge case: 1-row batch can't reference iloc[-2], must use iloc[-1]."""
        ohlcv = [
            [1712102400000, 60000.0, 61000.0, 59000.0, 60500.0, 100.0],
        ]
        fetcher = self._fetcher_with_exchange(_FakeExchange(ohlcv))

        df = fetcher.get_daily_data("BTC-USD", days=1)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df["volume"].iloc[0], 100.0 * 60500.0, places=2)

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
