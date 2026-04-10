# -*- coding: utf-8 -*-
import unittest
import sys
import os
import tempfile
import threading
from datetime import date
from unittest.mock import patch

import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.sql import func

# Ensure src module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.storage import DatabaseManager, StockDaily

class TestStorage(unittest.TestCase):
    
    def test_parse_sniper_value(self):
        """测试解析狙击点位数值"""
        
        # 1. 正常数值
        self.assertEqual(DatabaseManager._parse_sniper_value(100), 100.0)
        self.assertEqual(DatabaseManager._parse_sniper_value(100.5), 100.5)
        self.assertEqual(DatabaseManager._parse_sniper_value("100"), 100.0)
        self.assertEqual(DatabaseManager._parse_sniper_value("100.5"), 100.5)
        
        # 2. 包含中文描述和"元"
        self.assertEqual(DatabaseManager._parse_sniper_value("建议在 100 元附近买入"), 100.0)
        self.assertEqual(DatabaseManager._parse_sniper_value("价格：100.5元"), 100.5)
        
        # 3. 包含干扰数字（修复的Bug场景）
        # 之前 "MA5" 会被错误提取为 5.0，现在应该提取 "元" 前面的 100
        text_bug = "无法给出。需等待MA5数据恢复，在股价回踩MA5且乖离率<2%时考虑100元"
        self.assertEqual(DatabaseManager._parse_sniper_value(text_bug), 100.0)
        
        # 4. 更多干扰场景
        text_complex = "MA10为20.5，建议在30元买入"
        self.assertEqual(DatabaseManager._parse_sniper_value(text_complex), 30.0)
        
        text_multiple = "支撑位10元，阻力位20元" # 应该提取最后一个"元"前面的数字，即20，或者更复杂的逻辑？
        # 当前逻辑是找最后一个冒号，然后找之后的第一个"元"，提取中间的数字。
        # 测试没有冒号的情况
        self.assertEqual(DatabaseManager._parse_sniper_value("30元"), 30.0)
        
        # 测试多个数字在"元"之前
        self.assertEqual(DatabaseManager._parse_sniper_value("MA5 10 20元"), 20.0)
        
        # 5. Fallback: no "元" character — extracts last non-MA number
        self.assertEqual(DatabaseManager._parse_sniper_value("102.10-103.00（MA5附近）"), 103.0)
        self.assertEqual(DatabaseManager._parse_sniper_value("97.62-98.50（MA10附近）"), 98.5)
        self.assertEqual(DatabaseManager._parse_sniper_value("93.40下方（MA20支撑）"), 93.4)
        self.assertEqual(DatabaseManager._parse_sniper_value("108.00-110.00（前期高点阻力）"), 110.0)

        # 6. 无效输入
        self.assertIsNone(DatabaseManager._parse_sniper_value(None))
        self.assertIsNone(DatabaseManager._parse_sniper_value(""))
        self.assertIsNone(DatabaseManager._parse_sniper_value("没有数字"))
        self.assertIsNone(DatabaseManager._parse_sniper_value("MA5但没有元"))

        # 7. 回归：括号内技术指标数字不应被提取
        self.assertNotEqual(DatabaseManager._parse_sniper_value("1.52-1.53 (回踩MA5/10附近)"), 10.0)
        self.assertNotEqual(DatabaseManager._parse_sniper_value("1.55-1.56(MA5/M20支撑)"), 20.0)
        self.assertNotEqual(DatabaseManager._parse_sniper_value("1.49-1.50(MA60附近企稳)"), 60.0)
        # 验证正确值在区间内
        self.assertIn(DatabaseManager._parse_sniper_value("1.52-1.53 (回踩MA5/10附近)"), [1.52, 1.53])
        self.assertIn(DatabaseManager._parse_sniper_value("1.55-1.56(MA5/M20支撑)"), [1.55, 1.56])
        self.assertIn(DatabaseManager._parse_sniper_value("1.49-1.50(MA60附近企稳)"), [1.49, 1.50])

    def test_get_chat_sessions_prefix_is_scoped_by_colon_boundary(self):
        DatabaseManager.reset_instance()
        db = DatabaseManager(db_url="sqlite:///:memory:")

        db.save_conversation_message("telegram_12345:chat", "user", "first user")
        db.save_conversation_message("telegram_123456:chat", "user", "second user")

        sessions = db.get_chat_sessions(session_prefix="telegram_12345")

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["session_id"], "telegram_12345:chat")

        DatabaseManager.reset_instance()

    def test_get_chat_sessions_can_include_legacy_exact_session_id(self):
        DatabaseManager.reset_instance()
        db = DatabaseManager(db_url="sqlite:///:memory:")

        db.save_conversation_message("feishu_u1", "user", "legacy chat")
        db.save_conversation_message("feishu_u1:ask_600519", "user", "ask session")

        sessions = db.get_chat_sessions(
            session_prefix="feishu_u1:",
            extra_session_ids=["feishu_u1"],
        )

        self.assertEqual({item["session_id"] for item in sessions}, {"feishu_u1", "feishu_u1:ask_600519"})

        DatabaseManager.reset_instance()

    def test_file_sqlite_enables_wal_and_busy_timeout(self):
        temp_dir = tempfile.TemporaryDirectory()
        db_path = os.path.join(temp_dir.name, "sqlite_pragmas.db")
        original_env = {
            "DATABASE_PATH": os.environ.get("DATABASE_PATH"),
            "SQLITE_BUSY_TIMEOUT_MS": os.environ.get("SQLITE_BUSY_TIMEOUT_MS"),
            "SQLITE_WAL_ENABLED": os.environ.get("SQLITE_WAL_ENABLED"),
        }

        try:
            os.environ["DATABASE_PATH"] = db_path
            os.environ["SQLITE_BUSY_TIMEOUT_MS"] = "1234"
            os.environ["SQLITE_WAL_ENABLED"] = "true"
            Config.reset_instance()
            DatabaseManager.reset_instance()

            db = DatabaseManager.get_instance()
            with db.get_session() as session:
                journal_mode = session.connection().exec_driver_sql("PRAGMA journal_mode").scalar()
                busy_timeout = session.connection().exec_driver_sql("PRAGMA busy_timeout").scalar()

            self.assertEqual(str(journal_mode).lower(), "wal")
            self.assertEqual(int(busy_timeout), 1234)
        finally:
            DatabaseManager.reset_instance()
            Config.reset_instance()
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            temp_dir.cleanup()

    def test_sqlite_write_transactions_begin_immediate(self):
        DatabaseManager.reset_instance()
        db = DatabaseManager(db_url="sqlite:///:memory:")
        session = db.get_session()
        connection = session.connection()

        try:
            with patch.object(db, "get_session", return_value=session):
                with patch.object(connection, "exec_driver_sql", wraps=connection.exec_driver_sql) as mock_exec:
                    result = db._run_write_transaction("unit-test", lambda current_session: 7)

            self.assertEqual(result, 7)
            self.assertTrue(
                any(call.args == ("BEGIN IMMEDIATE",) for call in mock_exec.call_args_list)
            )
        finally:
            DatabaseManager.reset_instance()

    def test_save_daily_data_sqlite_concurrent_same_code_date_counts_only_new_rows(self):
        DatabaseManager.reset_instance()
        temp_dir = tempfile.TemporaryDirectory()
        db_path = os.path.join(temp_dir.name, "sqlite_daily_concurrency.db")
        db = DatabaseManager(db_url=f"sqlite:///{db_path}")

        results = []
        results_lock = threading.Lock()
        start_barrier = threading.Barrier(2)

        def worker() -> None:
            start_barrier.wait()
            count = db.save_daily_data(
                pd.DataFrame(
                    [
                        {
                            'date': date(2026, 4, 1),
                            'open': 10,
                            'high': 11,
                            'low': 9,
                            'close': 10.5,
                            'volume': 100,
                            'amount': 1050,
                            'pct_chg': 1.2,
                            'ma5': 10.1,
                            'ma10': 10.2,
                            'ma20': 10.3,
                            'volume_ratio': 1.0,
                        }
                    ]
                ),
                code='600519',
                data_source='test',
            )
            with results_lock:
                results.append(count)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        try:
            self.assertCountEqual(results, [1, 0])

            with db.get_session() as session:
                total = session.execute(
                    select(func.count()).select_from(StockDaily).where(
                        and_(
                            StockDaily.code == '600519',
                            StockDaily.date == date(2026, 4, 1),
                        )
                    )
                ).scalar()

            self.assertEqual(total, 1)
        finally:
            temp_dir.cleanup()
            DatabaseManager.reset_instance()


class TestSourceConsistentTail(unittest.TestCase):
    """
    Guard against cross-source volume_ratio contamination.

    ``CCXTCryptoFetcher`` returns single-exchange (e.g. Kraken) base-currency
    volume — even after the fetcher's own T-1 close normalization this stays
    at the exchange's market share (~0.37% for BTC). ``YfinanceFetcher``
    returns globally aggregated USD notional (~270× larger for the same day).
    If both sources have written rows for the same crypto code and
    ``_analyze_volume`` rolls a window across the boundary, ``volume_ratio_5d``
    jumps ~270×, producing a fake "volume spike" / "volume shrink" signal.

    ``DatabaseManager.get_source_consistent_tail`` defends against this by
    returning only the longest end-of-window slice whose rows share the most
    recent row's ``data_source``.
    """

    def _build_db(self) -> DatabaseManager:
        DatabaseManager.reset_instance()
        return DatabaseManager(db_url="sqlite:///:memory:")

    def _save(self, db: DatabaseManager, code: str, d: date, volume: float, source: str) -> None:
        db.save_daily_data(
            pd.DataFrame(
                [
                    {
                        'date': d,
                        'open': 100.0,
                        'high': 101.0,
                        'low': 99.0,
                        'close': 100.0,
                        'volume': volume,
                        'amount': 100.0 * volume,
                        'pct_chg': 0.0,
                        'ma5': 100.0,
                        'ma10': 100.0,
                        'ma20': 100.0,
                        'volume_ratio': 1.0,
                    }
                ]
            ),
            code=code,
            data_source=source,
        )

    def test_returns_empty_when_no_rows(self) -> None:
        db = self._build_db()
        try:
            result = db.get_source_consistent_tail(
                'BTC-USD', date(2026, 4, 1), date(2026, 4, 10)
            )
            self.assertEqual(result, [])
        finally:
            DatabaseManager.reset_instance()

    def test_returns_full_window_when_all_rows_same_source(self) -> None:
        db = self._build_db()
        try:
            for i in range(6):
                self._save(
                    db, 'BTC-USD', date(2026, 4, 1 + i),
                    volume=1000.0 * (i + 1),
                    source='CCXTCryptoFetcher',
                )
            result = db.get_source_consistent_tail(
                'BTC-USD', date(2026, 4, 1), date(2026, 4, 10)
            )
            self.assertEqual(len(result), 6)
            self.assertTrue(all(bar.data_source == 'CCXTCryptoFetcher' for bar in result))
        finally:
            DatabaseManager.reset_instance()

    def test_trims_leading_rows_when_source_switches_at_tail(self) -> None:
        """Prev source (Yfinance) must be dropped once newer source (CCXT) takes over."""
        db = self._build_db()
        try:
            # Days 1-3: YfinanceFetcher (global aggregate, ~270x bigger)
            for i in range(3):
                self._save(
                    db, 'BTC-USD', date(2026, 4, 1 + i),
                    volume=2.7e10,  # global USD notional
                    source='YfinanceFetcher',
                )
            # Days 4-6: CCXTCryptoFetcher (Kraken single-exchange, ~270x smaller)
            for i in range(3):
                self._save(
                    db, 'BTC-USD', date(2026, 4, 4 + i),
                    volume=1.0e8,  # Kraken USD notional
                    source='CCXTCryptoFetcher',
                )

            result = db.get_source_consistent_tail(
                'BTC-USD', date(2026, 4, 1), date(2026, 4, 10)
            )

            # Only the trailing CCXT tail should be returned (days 4-6)
            self.assertEqual(len(result), 3)
            self.assertEqual(result[0].date, date(2026, 4, 4))
            self.assertEqual(result[-1].date, date(2026, 4, 6))
            self.assertTrue(all(bar.data_source == 'CCXTCryptoFetcher' for bar in result))
        finally:
            DatabaseManager.reset_instance()

    def test_cross_source_volume_ratio_does_not_explode(self) -> None:
        """
        Regression guard: the whole point of the helper.

        With raw get_data_range, a 6-day window split 3 YF + 3 CCXT would
        compute volume_ratio_5d = today (1e8) / mean(prev5 including YF rows
        at 2.7e10) ~ 0.000015, triggering a false "extreme shrink" signal.
        With get_source_consistent_tail, only the CCXT tail is visible, so
        the rolling ratio computed downstream stays ~1.0.
        """
        db = self._build_db()
        try:
            # Mixed source window: 5 YF days then 1 CCXT day
            for i in range(5):
                self._save(
                    db, 'BTC-USD', date(2026, 4, 1 + i),
                    volume=2.7e10,
                    source='YfinanceFetcher',
                )
            self._save(
                db, 'BTC-USD', date(2026, 4, 6),
                volume=1.0e8,
                source='CCXTCryptoFetcher',
            )

            raw = db.get_data_range('BTC-USD', date(2026, 4, 1), date(2026, 4, 10))
            self.assertEqual(len(raw), 6)
            # Naive cross-source ratio explodes: today (1e8) / mean(prev5 at 2.7e10) ≈ 0.0037,
            # which is ~270x below the healthy ~1.0 baseline — a fake "extreme shrink" signal.
            raw_volumes = [b.volume for b in raw]
            naive_ratio = raw_volumes[-1] / (sum(raw_volumes[:-1]) / 5)
            self.assertLess(naive_ratio, 0.01)
            self.assertGreater(
                1.0 / naive_ratio, 100.0,
                f"expected >100x fake spike vs healthy ratio 1.0, got {1.0/naive_ratio:.1f}x",
            )

            consistent = db.get_source_consistent_tail(
                'BTC-USD', date(2026, 4, 1), date(2026, 4, 10)
            )
            # Only the single CCXT day survives (< 5 rows → volume_ratio
            # downstream degrades to default, which is the intended safer
            # signal than a fake 1000× shrink)
            self.assertEqual(len(consistent), 1)
            self.assertEqual(consistent[0].data_source, 'CCXTCryptoFetcher')
        finally:
            DatabaseManager.reset_instance()

    def test_ignores_non_crypto_unrelated_source_noise(self) -> None:
        """
        A stock with a stable single source across the whole window is
        unaffected (no trimming, no warning).
        """
        db = self._build_db()
        try:
            for i in range(10):
                self._save(
                    db, 'AAPL', date(2026, 4, 1 + i),
                    volume=5.0e7,
                    source='YfinanceFetcher',
                )
            result = db.get_source_consistent_tail(
                'AAPL', date(2026, 4, 1), date(2026, 4, 15)
            )
            self.assertEqual(len(result), 10)
        finally:
            DatabaseManager.reset_instance()


if __name__ == '__main__':
    unittest.main()
