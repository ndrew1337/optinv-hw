from pathlib import Path

import pandas as pd
import pytest

import run_tardis
from src.tardis_backtest import L2Config, run_l2_backtest
from src.tardis_data import load_grid


def _book_df(exchange: str, asset: str, rows: list[tuple[int, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts": ts,
                "asks[0].price": ask,
                "asks[0].amount": 10.0,
                "bids[0].price": bid,
                "bids[0].amount": 10.0,
                "exchange": exchange,
                "asset": asset,
            }
            for ts, ask, bid in rows
        ]
    )


def test_l2_latency_executes_at_or_after_target_time():
    panels = {
        ("cheap", "BTC"): _book_df(
            "cheap",
            "BTC",
            [(0, 100.0, 99.0), (100, 100.0, 99.0), (300, 100.0, 99.0)],
        ),
        ("rich", "BTC"): _book_df(
            "rich",
            "BTC",
            [(0, 102.0, 101.0), (100, 102.0, 101.0), (300, 90.0, 89.0)],
        ),
    }

    result = run_l2_backtest(
        panels,
        L2Config(
            fee=0.0,
            max_notional_usdt=100.0,
            min_profit_usdt=0.01,
            latency_ms=150,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=1000,
        ),
    )

    assert len(result.trades) == 2
    assert all(t.pnl < 0 for t in result.trades)
    assert all(t.exec_ts == 300 for t in result.trades)


def test_l2_equity_uses_execution_timestamp():
    panels = {
        ("cheap", "BTC"): _book_df(
            "cheap",
            "BTC",
            [(0, 100.0, 99.0), (100, 100.0, 99.0), (200, 100.0, 99.0)],
        ),
        ("rich", "BTC"): _book_df(
            "rich",
            "BTC",
            [(0, 102.0, 101.0), (100, 102.0, 101.0), (200, 102.0, 101.0)],
        ),
    }

    result = run_l2_backtest(
        panels,
        L2Config(
            fee=0.0,
            max_notional_usdt=100.0,
            min_profit_usdt=0.01,
            latency_ms=150,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=1000,
        ),
    )

    assert len(result.trades) == 1
    assert result.trades[0].ts == 0
    assert result.trades[0].exec_ts == 200
    assert result.equity_curve.iloc[0] == 10_000.0
    assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]
    assert result.summary()["total_return_pct"] == pytest.approx(
        (result.trades[0].pnl / 10_000.0) * 100
    )


def test_l2_drops_stale_forward_filled_quotes():
    panels = {
        ("cheap", "BTC"): _book_df("cheap", "BTC", [(0, 100.0, 99.0)]),
        ("rich", "BTC"): _book_df("rich", "BTC", [(300, 102.0, 101.0)]),
    }

    result = run_l2_backtest(
        panels,
        L2Config(
            fee=0.0,
            max_notional_usdt=100.0,
            min_profit_usdt=0.01,
            latency_ms=0,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=100,
        ),
    )

    assert result.trades == []
    assert result.raw_crosses == 0
    assert result.executable_candidates == 0


def test_l2_min_edge_filter_blocks_tiny_executable_trade():
    panels = {
        ("cheap", "BTC"): _book_df("cheap", "BTC", [(0, 100.0, 99.0)]),
        ("rich", "BTC"): _book_df("rich", "BTC", [(0, 100.2, 100.1)]),
    }

    result = run_l2_backtest(
        panels,
        L2Config(
            fee=0.0,
            max_notional_usdt=100.0,
            min_profit_usdt=0.0,
            min_edge_bps=20.0,
            latency_ms=0,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=100,
        ),
    )

    assert result.raw_crosses == 1
    assert result.executable_candidates == 0
    assert result.trades == []


def test_l2_records_loss_when_signal_passes_but_execution_decays():
    panels = {
        ("cheap", "BTC"): _book_df(
            "cheap",
            "BTC",
            [(0, 100.0, 99.0), (100, 100.0, 99.0)],
        ),
        ("rich", "BTC"): _book_df(
            "rich",
            "BTC",
            [(0, 103.0, 102.0), (100, 95.0, 94.0)],
        ),
    }

    result = run_l2_backtest(
        panels,
        L2Config(
            fee=0.0,
            max_notional_usdt=100.0,
            min_profit_usdt=1.0,
            min_edge_bps=100.0,
            latency_ms=100,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=100,
        ),
    )

    assert result.raw_crosses == 1
    assert result.executable_candidates == 1
    assert len(result.trades) == 1
    assert result.trades[0].signal_expected_pnl > 1.0
    assert result.trades[0].pnl < 0


def test_load_grid_normalizes_legacy_sec_cache(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    cache = grid_dir / "binance_BTC_20260501_d1_g1000ms.parquet"
    pd.DataFrame(
        [
            {
                "sec": 1_777_593_601,
                "asks[0].price": 100.0,
                "asks[0].amount": 1.0,
                "bids[0].price": 99.0,
                "bids[0].amount": 1.0,
                "exchange": "binance",
                "asset": "BTC",
            }
        ]
    ).to_parquet(cache, index=False)

    df = load_grid("binance", "BTC", "2026-05-01", raw_dir, grid_dir, depth=1, grid_ms=1000)

    assert "ts" in df.columns
    assert "sec" not in df.columns
    assert df["ts"].iloc[0] == 1_777_593_601_000


def test_write_trades_csv_overwrites_empty_trade_files(tmp_path: Path):
    stale = tmp_path / "tardis_trades_l2_taker_0p10_lat0.csv"
    stale.write_text("old,data\n1,2\n", encoding="utf-8")

    run_tardis.write_trades_csv([], stale)

    df = pd.read_csv(stale)
    assert list(df.columns) == run_tardis.TRADE_COLUMNS
    assert df.empty
