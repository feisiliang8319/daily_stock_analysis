# -*- coding: utf-8 -*-
"""
Crypto market context fetcher (MVP).

Data sources:
1. Alternative.me — Fear & Greed Index (free, no API key)
2. CoinGecko — Global market data + coin details (free tier, 30 req/min)

Used by the analysis pipeline to enrich LLM prompts with crypto-specific context.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_TIMEOUT = 10  # seconds


def get_fear_greed_index() -> Optional[dict]:
    """Fetch current Crypto Fear & Greed Index from Alternative.me.

    Returns:
        {"value": "25", "classification": "Extreme Fear", "timestamp": "..."}
        or None on failure.
    """
    try:
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1",
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            return {
                "value": data[0].get("value"),
                "classification": data[0].get("value_classification"),
                "timestamp": data[0].get("timestamp"),
            }
    except Exception as e:
        logger.warning(f"Fear & Greed fetch failed: {e}")
    return None


def get_global_crypto_market() -> Optional[dict]:
    """Fetch global crypto market overview from CoinGecko.

    Returns:
        {
            "total_market_cap_usd": ...,
            "total_volume_24h_usd": ...,
            "btc_dominance": ...,
            "eth_dominance": ...,
            "active_cryptocurrencies": ...,
            "market_cap_change_24h_pct": ...,
        }
        or None on failure.
    """
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/global",
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        return {
            "total_market_cap_usd": data.get("total_market_cap", {}).get("usd"),
            "total_volume_24h_usd": data.get("total_volume", {}).get("usd"),
            "btc_dominance": data.get("market_cap_percentage", {}).get("btc"),
            "eth_dominance": data.get("market_cap_percentage", {}).get("eth"),
            "active_cryptocurrencies": data.get("active_cryptocurrencies"),
            "market_cap_change_24h_pct": data.get("market_cap_change_percentage_24h_usd"),
        }
    except Exception as e:
        logger.warning(f"CoinGecko global fetch failed: {e}")
    return None


def get_coin_market_data(coin_id: str) -> Optional[dict]:
    """Fetch specific coin market data from CoinGecko.

    Args:
        coin_id: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum', 'solana')

    Returns:
        {"market_cap", "market_cap_rank", "ath", "ath_change_pct", "circulating_supply", ...}
        or None on failure.
    """
    try:
        resp = requests.get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}",
            params={
                "localization": "false",
                "tickers": "false",
                "community_data": "false",
                "developer_data": "false",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        md = data.get("market_data", {})
        return {
            "market_cap_usd": md.get("market_cap", {}).get("usd"),
            "market_cap_rank": data.get("market_cap_rank"),
            "ath_usd": md.get("ath", {}).get("usd"),
            "ath_change_pct": md.get("ath_change_percentage", {}).get("usd"),
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply"),
            "price_change_24h_pct": md.get("price_change_percentage_24h"),
            "price_change_7d_pct": md.get("price_change_percentage_7d"),
            "price_change_30d_pct": md.get("price_change_percentage_30d"),
            "total_volume_usd": md.get("total_volume", {}).get("usd"),
        }
    except Exception as e:
        logger.warning(f"CoinGecko coin data fetch failed for {coin_id}: {e}")
    return None


# Ticker symbol -> CoinGecko coin ID mapping
TICKER_TO_COINGECKO = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
    "BNB-USD": "binancecoin",
    "XRP-USD": "ripple",
    "ADA-USD": "cardano",
    "DOGE-USD": "dogecoin",
    "AVAX-USD": "avalanche-2",
    "DOT-USD": "polkadot",
    "LINK-USD": "chainlink",
    "MATIC-USD": "matic-network",
    "UNI-USD": "uniswap",
    "ATOM-USD": "cosmos",
    "LTC-USD": "litecoin",
    "NEAR-USD": "near",
    "APT-USD": "aptos",
    "ARB-USD": "arbitrum",
    "OP-USD": "optimism",
    "SUI-USD": "sui",
}


def build_crypto_context(ticker: str) -> str:
    """Build a crypto-specific context string for LLM prompt enrichment.

    Args:
        ticker: Crypto ticker like 'BTC-USD'

    Returns:
        Multi-line context string with Fear & Greed, global data, and coin data.
        Returns empty string if all fetches fail.
    """
    parts = []

    # Fear & Greed Index
    fng = get_fear_greed_index()
    if fng:
        parts.append(
            f"- **Crypto Fear & Greed Index**: {fng['value']}/100 ({fng['classification']})"
        )

    # Global market
    global_data = get_global_crypto_market()
    if global_data:
        mc = global_data.get("total_market_cap_usd")
        vol = global_data.get("total_volume_24h_usd")
        btc_dom = global_data.get("btc_dominance")
        mc_change = global_data.get("market_cap_change_24h_pct")
        # CoinGecko occasionally returns null for any of these fields; formatting
        # None with :+.1f / :.1f raises TypeError and aborts the whole function,
        # dropping the entire crypto enrichment for that run. Guard each field.
        mc_str = f"${mc/1e12:.2f}T" if mc else "N/A"
        vol_str = f"${vol/1e9:.1f}B" if vol else "N/A"
        mc_change_str = f"{mc_change:+.1f}%" if mc_change is not None else "N/A"
        btc_dom_str = f"{btc_dom:.1f}%" if btc_dom is not None else "N/A"
        parts.append(
            f"- **全球加密市场**: 总市值 {mc_str} (24h {mc_change_str}), "
            f"24h 成交量 {vol_str}, BTC 主导率 {btc_dom_str}"
        )

    # Coin-specific data
    coin_id = TICKER_TO_COINGECKO.get(ticker.upper())
    if coin_id:
        coin = get_coin_market_data(coin_id)
        if coin:
            rank = coin.get("market_cap_rank", "N/A")
            ath = coin.get("ath_usd")
            ath_chg = coin.get("ath_change_pct")
            p7d = coin.get("price_change_7d_pct")
            p30d = coin.get("price_change_30d_pct")
            supply = coin.get("circulating_supply")
            max_sup = coin.get("max_supply")

            # Same defensive guards as above — any null from CoinGecko must not
            # crash the enrichment path.
            if ath is not None:
                ath_chg_str = f"{ath_chg:+.1f}%" if ath_chg is not None else "N/A"
                ath_str = f"${ath:,.0f} ({ath_chg_str})"
            else:
                ath_str = "N/A"
            supply_str = f"{supply/1e6:.1f}M" if supply else "N/A"
            max_str = f"{max_sup/1e6:.1f}M" if max_sup else "无上限"
            p7d_str = f"{p7d:+.1f}%" if p7d is not None else "N/A"
            p30d_str = f"{p30d:+.1f}%" if p30d is not None else "N/A"

            parts.append(
                f"- **{ticker}**: 市值排名 #{rank}, ATH {ath_str}, "
                f"7d {p7d_str}, 30d {p30d_str}, "
                f"流通量 {supply_str} / 最大 {max_str}"
            )

    return "\n".join(parts)
