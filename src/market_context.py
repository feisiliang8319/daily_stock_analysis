# -*- coding: utf-8 -*-
"""
Market context detection for LLM prompts.

Detects the market (A-shares, HK, US) from a stock code and returns
market-specific role descriptions so prompts are not hardcoded to a
single market.

Fixes: https://github.com/ZhuLinsen/daily_stock_analysis/issues/644
"""

import re
from typing import Optional


def detect_market(stock_code: Optional[str]) -> str:
    """Detect market from stock code.

    Returns:
        One of 'cn', 'hk', 'us', 'crypto', or 'us' as fallback.
    """
    if not stock_code:
        return "cn"

    code = stock_code.strip().upper()

    # Crypto: BTC-USD, ETH-USD, SOL-USD etc.
    if re.match(r'^[A-Z]{2,10}-USD$', code):
        return "crypto"

    # HK stocks: HK00700, 00700.HK, or 5-digit pure numbers
    if code.startswith("HK") or code.endswith(".HK"):
        return "hk"
    lower = code.lower()
    if lower.endswith(".hk"):
        return "hk"
    # 5-digit pure numbers are HK (A-shares are 6-digit)
    if code.isdigit() and len(code) == 5:
        return "hk"

    # US stocks: 1-5 uppercase letters (AAPL, TSLA, GOOGL)
    # Also handles suffixed forms like BRK.B
    if re.match(r'^[A-Z]{1,5}(\.[A-Z]{1,2})?$', code):
        return "us"

    # HK stocks: HK00700, 00700.HK, or 5-digit pure numbers
    # (moved below US to deprioritize)

    # A-shares (6-digit numbers like 600519, 000001)
    if code.isdigit() and len(code) == 6:
        return "cn"

    # Default: US (was cn, changed for US+crypto focused deployment)
    return "us"


# -- Market-specific role descriptions --

_MARKET_ROLES = {
    "cn": {
        "zh": " A 股",
        "en": "China A-shares",
    },
    "hk": {
        "zh": "港股",
        "en": "Hong Kong stock",
    },
    "us": {
        "zh": "美股",
        "en": "US stock",
    },
    "crypto": {
        "zh": "加密货币",
        "en": "Cryptocurrency",
    },
}

_MARKET_GUIDELINES = {
    "cn": {
        "zh": (
            "- 本次分析对象为 **A 股**（中国沪深交易所上市股票）。\n"
            "- 请关注 A 股特有的涨跌停机制（±10%/±20%/±30%）、T+1 交易制度及相关政策因素。"
        ),
        "en": (
            "- This analysis covers a **China A-share** (listed on Shanghai/Shenzhen exchanges).\n"
            "- Consider A-share-specific rules: daily price limits (±10%/±20%/±30%), T+1 settlement, and PRC policy factors."
        ),
    },
    "hk": {
        "zh": (
            "- 本次分析对象为 **港股**（香港交易所上市股票）。\n"
            "- 港股无涨跌停限制，支持 T+0 交易，需关注港币汇率、南北向资金流及联交所特有规则。"
        ),
        "en": (
            "- This analysis covers a **Hong Kong stock** (listed on HKEX).\n"
            "- HK stocks have no daily price limits, allow T+0 trading. Consider HKD FX, Southbound/Northbound flows, and HKEX-specific rules."
        ),
    },
    "us": {
        "zh": (
            "- 本次分析对象为 **美股**（美国交易所上市股票）。\n"
            "- 美股无涨跌停限制（但有熔断机制），支持 T+0 交易和盘前盘后交易，需关注美元汇率、美联储政策及 SEC 监管动态。"
        ),
        "en": (
            "- This analysis covers a **US stock** (listed on NYSE/NASDAQ).\n"
            "- US stocks have no daily price limits (but have circuit breakers), allow T+0 and pre/after-market trading. Consider USD FX, Fed policy, and SEC regulations."
        ),
    },
    "crypto": {
        "zh": (
            "- 本次分析对象为 **加密货币**（去中心化数字资产）。\n"
            "- 7×24 小时全球交易，无涨跌停限制，无熔断机制，波动性极高。\n"
            "- 需关注：链上数据、市场情绪（Fear & Greed Index）、BTC 主导率、监管政策、鲸鱼动向、资金费率。\n"
            "- 价格受 FOMO/FUD 情绪驱动显著，技术分析需结合链上指标。"
        ),
        "en": (
            "- This analysis covers a **cryptocurrency** (decentralized digital asset).\n"
            "- Trades 24/7 globally, no price limits, no circuit breakers, extremely volatile.\n"
            "- Consider: on-chain data, Fear & Greed Index, BTC dominance, regulatory news, whale movements, funding rates.\n"
            "- Price is heavily sentiment-driven (FOMO/FUD); combine technical analysis with on-chain metrics."
        ),
    },
}


def get_market_role(stock_code: Optional[str], lang: str = "zh") -> str:
    """Return market-specific role description for LLM prompt.

    Args:
        stock_code: The stock code being analyzed.
        lang: 'zh' or 'en'.

    Returns:
        Role string like 'A 股投资分析' or 'US stock investment analysis'.
    """
    market = detect_market(stock_code)
    lang_key = "en" if lang == "en" else "zh"
    return _MARKET_ROLES.get(market, _MARKET_ROLES["cn"])[lang_key]


def get_market_guidelines(stock_code: Optional[str], lang: str = "zh") -> str:
    """Return market-specific analysis guidelines for LLM prompt.

    Args:
        stock_code: The stock code being analyzed.
        lang: 'zh' or 'en'.

    Returns:
        Multi-line string with market-specific guidelines.
    """
    market = detect_market(stock_code)
    lang_key = "en" if lang == "en" else "zh"
    guidelines = _MARKET_GUIDELINES.get(market, _MARKET_GUIDELINES["cn"])[lang_key]

    # Crypto: append live market context (Fear & Greed, global data, coin data)
    if market == "crypto" and stock_code:
        try:
            from data_provider.crypto_context_fetcher import build_crypto_context
            crypto_ctx = build_crypto_context(stock_code.strip().upper())
            if crypto_ctx:
                guidelines += "\n\n**实时市场数据：**\n" + crypto_ctx
        except Exception:
            pass  # Graceful degradation: missing context won't block analysis

    return guidelines
