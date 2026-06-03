from pathlib import Path

import pandas as pd
import pytest

import run_tardis
from src.tardis_backtest import (
    L2Config,
    run_l2_backtest,
    run_l2_combined_backtest,
    run_l2_triangular_backtest,
)
from src.tardis_data import load_grid, normalize_pair, tardis_pair_symbol


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


def _pair_book_df(
    exchange: str,
    base: str,
    quote: str,
    rows: list[tuple[int, float, float]],
) -> pd.DataFrame:
    pair = f"{base}{quote}"
    return pd.DataFrame(
        [
            {
                "ts": ts,
                "asks[0].price": ask,
                "asks[0].amount": 1_000.0,
                "bids[0].price": bid,
                "bids[0].amount": 1_000.0,
                "exchange": exchange,
                "asset": base if quote == "USDT" else pair,
                "pair": pair,
                "base": base,
                "quote": quote,
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


def test_l2_min_profit_pct_blocks_edges_below_percent_of_notional():
    panels = {
        ("cheap", "BTC"): _book_df("cheap", "BTC", [(0, 100.0, 99.0)]),
        ("rich", "BTC"): _book_df("rich", "BTC", [(0, 101.0, 100.4)]),
    }

    result = run_l2_backtest(
        panels,
        L2Config(
            fee=0.0,
            max_notional_usdt=1_000.0,
            min_profit_usdt=0.0,
            min_profit_pct=0.5,
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


def test_l2_spot_fee_reduces_bought_asset_then_quote_proceeds():
    panels = {
        ("cheap", "BTC"): _book_df("cheap", "BTC", [(0, 100.0, 99.0)]),
        ("rich", "BTC"): _book_df("rich", "BTC", [(0, 111.0, 110.0)]),
    }

    result = run_l2_backtest(
        panels,
        L2Config(
            fee=0.001,
            max_notional_usdt=100.0,
            min_profit_usdt=0.0,
            latency_ms=0,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=100,
        ),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.buy_cost == pytest.approx(100.0)
    assert trade.size == pytest.approx(0.999)
    assert trade.sell_proceeds == pytest.approx(0.999 * 110.0 * 0.999)
    assert trade.pnl == pytest.approx(trade.signal_expected_pnl)


def test_l2_inventory_reservation_blocks_overlapping_orders():
    panels = {
        ("cheap", "BTC"): _book_df(
            "cheap",
            "BTC",
            [(0, 100.0, 99.0), (100, 100.0, 99.0), (400, 100.0, 99.0)],
        ),
        ("rich", "BTC"): _book_df(
            "rich",
            "BTC",
            [(0, 130.0, 120.0), (100, 130.0, 120.0), (400, 130.0, 120.0)],
        ),
    }

    result = run_l2_backtest(
        panels,
        L2Config(
            fee=0.0,
            max_notional_usdt=2_000.0,
            min_profit_usdt=1.0,
            latency_ms=300,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=1000,
            start_capital_usdt=10_000.0,
            enforce_inventory=True,
        ),
    )

    assert len(result.trades) == 1
    assert result.inventory_skips >= 1
    assert result.trades[0].ts == 0
    assert result.trades[0].exec_ts == 400


def test_l2_inventory_per_currency_usdt_supports_larger_notional():
    panels = {
        ("cheap", "BTC"): _book_df("cheap", "BTC", [(0, 100.0, 99.0)]),
        ("rich", "BTC"): _book_df("rich", "BTC", [(0, 130.0, 120.0)]),
    }

    result = run_l2_backtest(
        panels,
        L2Config(
            fee=0.0,
            max_notional_usdt=2_000.0,
            min_profit_usdt=1.0,
            latency_ms=0,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=1000,
            inventory_per_currency_usdt=5_000.0,
            enforce_inventory=True,
        ),
    )

    assert len(result.trades) == 1
    assert result.inventory_skips == 0
    assert result.start_capital_usdt == pytest.approx(20_000.0)
    assert result.summary()["final_capital"] == pytest.approx(20_000.0 + result.trades[0].pnl)


def test_l2_stake_fraction_sizes_from_available_inventory():
    panels = {
        ("cheap", "BTC"): _book_df("cheap", "BTC", [(0, 100.0, 99.0)]),
        ("rich", "BTC"): _book_df("rich", "BTC", [(0, 130.0, 120.0)]),
    }

    result = run_l2_backtest(
        panels,
        L2Config(
            fee=0.0,
            stake_fraction=0.2,
            min_profit_usdt=1.0,
            latency_ms=0,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=1000,
            inventory_per_currency_usdt=5_000.0,
            enforce_inventory=True,
        ),
    )

    assert len(result.trades) == 1
    assert result.trades[0].buy_cost == pytest.approx(800.0)


def test_l2_triangular_backtest_executes_profitable_three_leg_cycle():
    panels = {
        ("binance", "BTCUSDT"): _pair_book_df("binance", "BTC", "USDT", [(0, 100.0, 99.0)]),
        ("binance", "ETHBTC"): _pair_book_df("binance", "ETH", "BTC", [(0, 0.5, 0.49)]),
        ("binance", "ETHUSDT"): _pair_book_df("binance", "ETH", "USDT", [(0, 61.0, 60.0)]),
    }

    result = run_l2_triangular_backtest(
        panels,
        L2Config(
            fee=0.0,
            max_notional_usdt=100.0,
            min_profit_usdt=1.0,
            min_edge_bps=1.0,
            latency_ms=0,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=1000,
        ),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.cycle == ["USDT", "BTC", "ETH", "USDT"]
    assert trade.start_amount == pytest.approx(100.0)
    assert trade.end_amount == pytest.approx(120.0)
    assert trade.pnl == pytest.approx(20.0)
    assert result.raw_cycles == 1


def test_l2_triangular_backtest_can_start_from_non_usdt_inventory():
    panels = {
        ("binance", "BTCUSDT"): _pair_book_df("binance", "BTC", "USDT", [(0, 100.0, 99.0)]),
        ("binance", "ETHBTC"): _pair_book_df("binance", "ETH", "BTC", [(0, 0.5, 0.49)]),
        ("binance", "ETHUSDT"): _pair_book_df("binance", "ETH", "USDT", [(0, 61.0, 60.0)]),
    }

    result = run_l2_triangular_backtest(
        panels,
        L2Config(
            fee=0.0,
            max_notional_usdt=100.0,
            min_profit_usdt=1.0,
            latency_ms=0,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=1000,
        ),
        start_currency="ETH",
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.start_currency == "ETH"
    assert trade.cycle == ["ETH", "USDT", "BTC", "ETH"]
    assert trade.start_value_usdt == pytest.approx(100.0)
    assert trade.pnl > 0


def test_l2_triangular_backtest_does_not_lookahead_to_execution_book():
    panels = {
        ("binance", "BTCUSDT"): _pair_book_df(
            "binance",
            "BTC",
            "USDT",
            [(0, 100.0, 99.0), (100, 100.0, 99.0)],
        ),
        ("binance", "ETHBTC"): _pair_book_df(
            "binance",
            "ETH",
            "BTC",
            [(0, 0.5, 0.49), (100, 0.5, 0.49)],
        ),
        ("binance", "ETHUSDT"): _pair_book_df(
            "binance",
            "ETH",
            "USDT",
            [(0, 51.0, 49.0), (100, 61.0, 60.0)],
        ),
    }

    result = run_l2_triangular_backtest(
        panels,
        L2Config(
            fee=0.0,
            max_notional_usdt=100.0,
            min_profit_usdt=1.0,
            min_edge_bps=1.0,
            latency_ms=100,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=1000,
        ),
    )

    assert result.trades == []
    assert result.raw_cycles == 0
    assert result.executable_candidates == 0


def test_l2_combined_backtest_selects_best_direct_or_triangle_signal():
    panels = {
        ("cheap", "BTC"): _book_df("cheap", "BTC", [(0, 100.0, 99.0)]),
        ("rich", "BTC"): _book_df("rich", "BTC", [(0, 130.0, 120.0)]),
    }
    pair_panels = {
        ("cheap", "BTCUSDT"): _pair_book_df("cheap", "BTC", "USDT", [(0, 100.0, 99.0)]),
        ("cheap", "ETHBTC"): _pair_book_df("cheap", "ETH", "BTC", [(0, 0.5, 0.49)]),
        ("cheap", "ETHUSDT"): _pair_book_df("cheap", "ETH", "USDT", [(0, 53.0, 52.0)]),
    }

    result = run_l2_combined_backtest(
        panels,
        pair_panels,
        L2Config(
            fee=0.0,
            stake_fraction=0.2,
            min_profit_pct=0.05,
            latency_ms=0,
            grid_ms=100,
            depth=1,
            max_quote_age_ms=1000,
            inventory_per_currency_usdt=5_000.0,
            enforce_inventory=True,
        ),
    )

    assert result.raw_crosses == 1
    assert result.raw_cycles == 1
    assert len(result.direct_trades) == 1
    assert result.triangular_trades == []
    assert result.summary()["trades_executed"] == 1
    assert result.summary()["by_type"]["direct"]["trades"] == 1


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


def test_build_scenarios_sweeps_fee_percent_and_latency_grid():
    scenarios = run_tardis.build_scenarios(
        "2026-05-01",
        [0.0, 0.01, 0.1],
        [0, 300],
        lambda fee, latency_ms: L2Config(fee=fee, latency_ms=latency_ms),
    )

    assert [s[0] for s in scenarios] == [
        "20260501_fee0pct_lat0",
        "20260501_fee0pct_lat300",
        "20260501_fee0p01pct_lat0",
        "20260501_fee0p01pct_lat300",
        "20260501_fee0p1pct_lat0",
        "20260501_fee0p1pct_lat300",
    ]
    assert [s[1] for s in scenarios] == [
        "Без комиссий, без задержки",
        "Без комиссий, задержка 300мс",
        "Комиссия 0.01%, без задержки",
        "Комиссия 0.01%, задержка 300мс",
        "Комиссия 0.1%, без задержки",
        "Комиссия 0.1%, задержка 300мс",
    ]
    assert [s[3].fee for s in scenarios] == pytest.approx(
        [0.0, 0.0, 0.0001, 0.0001, 0.001, 0.001]
    )


def test_pair_helpers_support_triangle_symbols():
    assert normalize_pair("ETH/BTC") == ("ETH", "BTC")
    assert normalize_pair("ETH-BTC") == ("ETH", "BTC")
    assert normalize_pair("SOL") == ("SOL", "USDT")
    assert tardis_pair_symbol("okx", "ETH", "BTC") == "ETH-BTC"
    assert tardis_pair_symbol("binance", "ETH", "BTC") == "ETHBTC"
    assert run_tardis.triangle_pairs_from_assets(["BTC", "ETH", "SOL"]) == [
        ("BTC", "USDT"),
        ("ETH", "USDT"),
        ("SOL", "USDT"),
        ("ETH", "BTC"),
        ("SOL", "BTC"),
        ("SOL", "ETH"),
    ]
