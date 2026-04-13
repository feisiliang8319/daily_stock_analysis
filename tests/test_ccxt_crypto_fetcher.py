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
from data_provider.realtime_types import RealtimeSource, UnifiedRealtimeQuote


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

    def test_daily_data_includes_standard_output_columns(self) -> None:
        """get_daily_data() must return all columns required by BaseFetcher contract.

        ZhuLinsen review blocker (PR #1037): missing amount/pct_chg/ma*/volume_ratio
        caused storage.save_daily_data() upserts to null-out these fields in the DB.
        """
        ohlcv = [
            [1712102400000 + i * 86400000, 60000.0, 61000.0, 59000.0, 60000.0 + i * 100, 1.0]
            for i in range(6)
        ]
        fetcher = self._fetcher_with_exchange(_FakeExchange(ohlcv))
        df = fetcher.get_daily_data("BTC-USD", days=6)

        required = {"amount", "pct_chg", "ma5", "ma10", "ma20", "volume_ratio"}
        missing = required - set(df.columns)
        self.assertEqual(missing, set(), f"Missing standard columns: {missing}")

        # amount == volume (both USD-notional for crypto)
        for a, v in zip(df["amount"].tolist(), df["volume"].tolist()):
            self.assertAlmostEqual(a, v, places=2)

        # pct_chg: first row has no prior close → 0.0
        self.assertAlmostEqual(df["pct_chg"].iloc[0], 0.0, places=4)
        # subsequent rows positive (close is trending up by +100 per step)
        self.assertGreater(df["pct_chg"].iloc[-1], 0.0)

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

    def test_realtime_method_name_matches_manager_router(self) -> None:
        """``DataFetcherManager.get_realtime_quote`` routes fetchers by the
        presence of a ``get_realtime_quote`` attribute (hasattr check). The
        CCXT fetcher MUST expose that exact name or it will be silently
        skipped for every crypto realtime call. Regression guard for the
        PR #1037 review bug where the method was named ``get_realtime_data``.
        """
        fetcher = CCXTCryptoFetcher.__new__(CCXTCryptoFetcher)
        self.assertTrue(hasattr(fetcher, "get_realtime_quote"))
        self.assertTrue(callable(getattr(fetcher, "get_realtime_quote")))

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

        quote = fetcher.get_realtime_quote("BTC-USD")

        self.assertIsInstance(quote, UnifiedRealtimeQuote)
        self.assertEqual(quote.code, "BTC-USD")
        self.assertEqual(quote.source, RealtimeSource.CCXT)
        self.assertEqual(quote.amount, 12345678.90)
        self.assertEqual(quote.price, 70000.0)
        self.assertAlmostEqual(quote.change_amount, 1000.0)  # 70000 - 69000
        self.assertEqual(quote.change_pct, 1.44)

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

        quote = fetcher.get_realtime_quote("BTC-USD")

        self.assertIsInstance(quote, UnifiedRealtimeQuote)
        self.assertEqual(quote.amount, 70_000_000.0)  # 1000 * 70000
        self.assertEqual(quote.volume, 70_000_000)

    def test_realtime_missing_both_volume_fields_returns_none_amount(self) -> None:
        """No baseVolume and no quoteVolume → amount/volume None, not crash."""
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

        quote = fetcher.get_realtime_quote("BTC-USD")

        self.assertIsInstance(quote, UnifiedRealtimeQuote)
        self.assertIsNone(quote.amount)
        self.assertIsNone(quote.volume)
        self.assertEqual(quote.price, 70000.0)

    def test_realtime_non_crypto_code_returns_none(self) -> None:
        """Non-crypto tickers must return None so the manager falls through."""
        fetcher = self._fetcher_with_exchange(_FakeExchange([], ticker={}))
        self.assertIsNone(fetcher.get_realtime_quote("AAPL"))


class TestManagerRealtimeQuoteCryptoRouting(unittest.TestCase):
    """Integration test: DataFetcherManager.get_realtime_quote routes crypto
    tickers to CCXTCryptoFetcher, NOT to A-share sources.

    Regression guard for PR #1037 P1 blocker: prior to the crypto fast path,
    BTC-USD did not match the US-stock regex (contains '-'), fell through to
    the A-share source loop (efinance/akshare/tushare), and silently returned
    None.
    """

    def _build_manager_with_mock_ccxt(self, quote_return):
        """Create a DataFetcherManager with a mocked CCXTCryptoFetcher."""
        from data_provider.base import DataFetcherManager

        mgr = DataFetcherManager.__new__(DataFetcherManager)
        mgr._fetchers = []
        mgr._fetchers_lock = __import__("threading").Lock()

        mock_fetcher = MagicMock()
        mock_fetcher.name = "CCXTCryptoFetcher"
        mock_fetcher.get_realtime_quote = MagicMock(return_value=quote_return)
        mgr._fetchers.append(mock_fetcher)

        # Add a YfinanceFetcher mock as fallback
        yf_mock = MagicMock()
        yf_mock.name = "YfinanceFetcher"
        yf_mock.get_realtime_quote = MagicMock(return_value=None)
        mgr._fetchers.append(yf_mock)

        return mgr, mock_fetcher, yf_mock

    def test_crypto_routes_to_ccxt_not_astock(self):
        """BTC-USD must be routed to CCXTCryptoFetcher via the crypto fast path."""
        expected_quote = UnifiedRealtimeQuote(
            code="BTC-USD", name="Bitcoin", price=70000.0,
            change_amount=1000.0, change_pct=1.44,
            volume=12345678, amount=12345678.0,
            high=70500.0, low=68900.0, open_price=69000.0,
            pre_close=69000.0, source=RealtimeSource.CCXT,
        )
        mgr, ccxt_mock, yf_mock = self._build_manager_with_mock_ccxt(expected_quote)

        result = mgr.get_realtime_quote("BTC-USD")

        self.assertIsNotNone(result, "crypto quote must not be None")
        self.assertEqual(result.source, RealtimeSource.CCXT)
        self.assertEqual(result.code, "BTC-USD")
        ccxt_mock.get_realtime_quote.assert_called_once()
        yf_mock.get_realtime_quote.assert_not_called()

    def test_crypto_falls_back_to_yfinance_when_ccxt_fails(self):
        """When CCXT returns None, manager should try YfinanceFetcher."""
        yf_quote = UnifiedRealtimeQuote(
            code="ETH-USD", name="Ethereum", price=3500.0,
            change_amount=50.0, change_pct=1.4,
            volume=999999, amount=999999.0,
            high=3550.0, low=3450.0, open_price=3450.0,
            pre_close=3450.0, source=RealtimeSource.FALLBACK,
        )
        mgr, ccxt_mock, yf_mock = self._build_manager_with_mock_ccxt(None)
        yf_mock.get_realtime_quote = MagicMock(return_value=yf_quote)

        result = mgr.get_realtime_quote("ETH-USD")

        self.assertIsNotNone(result)
        self.assertEqual(result.source, RealtimeSource.FALLBACK)
        ccxt_mock.get_realtime_quote.assert_called_once()
        yf_mock.get_realtime_quote.assert_called_once()

    def test_crypto_returns_none_when_all_sources_fail(self):
        """When both CCXT and YFinance fail, return None (not crash)."""
        mgr, ccxt_mock, yf_mock = self._build_manager_with_mock_ccxt(None)
        yf_mock.get_realtime_quote = MagicMock(return_value=None)

        result = mgr.get_realtime_quote("SOL-USD")

        self.assertIsNone(result)

    def test_us_stock_does_not_hit_crypto_path(self):
        """AAPL must NOT be routed through the crypto fast path."""
        mgr, ccxt_mock, yf_mock = self._build_manager_with_mock_ccxt(None)
        us_quote = UnifiedRealtimeQuote(
            code="AAPL", name="Apple", price=180.0,
            change_amount=2.0, change_pct=1.1,
            volume=50000000, amount=50000000.0,
            high=181.0, low=178.0, open_price=178.5,
            pre_close=178.0, source=RealtimeSource.FALLBACK,
        )
        yf_mock.get_realtime_quote = MagicMock(return_value=us_quote)

        result = mgr.get_realtime_quote("AAPL")

        # AAPL should go through the US path, not crypto
        ccxt_mock.get_realtime_quote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
