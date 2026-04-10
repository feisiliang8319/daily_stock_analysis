# -*- coding: utf-8 -*-
"""Regression tests for crypto_context_fetcher formatting guards.

CoinGecko's /global and /coins/{id} endpoints occasionally return ``null``
for individual fields (market_cap_change_24h_pct, btc_dominance,
ath_change_percentage, price_change_percentage_7d, etc.) — especially
during early UTC rollover windows or when the upstream aggregator is
degraded. ``build_crypto_context`` formatted these with ``:+.1f`` / ``:.1f``
without None guards, so a single null would raise TypeError, propagate to
the outer except, and drop the entire crypto enrichment (Fear & Greed +
global + coin) from the LLM prompt for that cron tick.

These tests lock in the None-safe contract so a partial CoinGecko
response still produces a usable context string.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from data_provider import crypto_context_fetcher


class BuildCryptoContextNoneGuardTest(unittest.TestCase):
    def _patch(self, fng=None, global_data=None, coin=None):
        return [
            patch.object(
                crypto_context_fetcher, "get_fear_greed_index", return_value=fng
            ),
            patch.object(
                crypto_context_fetcher,
                "get_global_crypto_market",
                return_value=global_data,
            ),
            patch.object(
                crypto_context_fetcher,
                "get_coin_market_data",
                return_value=coin,
            ),
        ]

    def _run(self, ticker="BTC-USD", **kwargs):
        patches = self._patch(**kwargs)
        for p in patches:
            p.start()
        try:
            return crypto_context_fetcher.build_crypto_context(ticker)
        finally:
            for p in patches:
                p.stop()

    def test_global_market_with_null_mc_change_and_btc_dominance(self) -> None:
        """The P2 regression: CoinGecko returns null for market_cap_change_24h_pct
        and btc_dominance. Old code crashed with TypeError on :+.1f / :.1f.
        New code must degrade to ``N/A`` and keep the rest of the line.
        """
        global_data = {
            "total_market_cap_usd": 2.5e12,
            "total_volume_24h_usd": 9.0e10,
            "btc_dominance": None,
            "market_cap_change_24h_pct": None,
        }
        # No coin mapping for an unlisted ticker avoids the coin-specific block.
        out = self._run(ticker="DOES-NOT-EXIST-USD", global_data=global_data)
        self.assertIn("全球加密市场", out)
        self.assertIn("$2.50T", out)
        self.assertIn("N/A", out)  # both btc_dom and mc_change degrade
        self.assertNotIn("None", out)

    def test_coin_data_with_null_percentage_fields(self) -> None:
        """Coin-specific block: ath_change_pct / price_change_7d_pct /
        price_change_30d_pct nullable. Must not crash."""
        coin = {
            "market_cap_rank": 1,
            "ath_usd": 73000.0,
            "ath_change_pct": None,   # unguarded in old code
            "price_change_7d_pct": None,
            "price_change_30d_pct": None,
            "circulating_supply": 19_700_000.0,
            "max_supply": 21_000_000.0,
        }
        out = self._run(ticker="BTC-USD", coin=coin)
        self.assertIn("BTC-USD", out)
        self.assertIn("$73,000", out)  # ath still formatted
        self.assertIn("N/A", out)
        self.assertNotIn("None", out)

    def test_coin_data_with_null_ath_degrades_cleanly(self) -> None:
        """ath_usd itself null → whole ATH field degrades to N/A."""
        coin = {
            "market_cap_rank": 2,
            "ath_usd": None,
            "ath_change_pct": None,
            "price_change_7d_pct": 3.5,
            "price_change_30d_pct": -1.2,
            "circulating_supply": 120_000_000.0,
            "max_supply": None,
        }
        out = self._run(ticker="ETH-USD", coin=coin)
        self.assertIn("ETH-USD", out)
        self.assertIn("ATH N/A", out)
        self.assertIn("+3.5%", out)
        self.assertIn("-1.2%", out)
        self.assertIn("无上限", out)

    def test_all_sources_null_returns_empty_string(self) -> None:
        """Every upstream returning None must yield an empty string, not crash."""
        out = self._run(fng=None, global_data=None, coin=None)
        self.assertEqual(out, "")


if __name__ == "__main__":
    unittest.main()
