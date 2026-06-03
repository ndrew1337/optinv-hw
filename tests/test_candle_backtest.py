import pandas as pd

from src.backtest import BacktestConfig, run_backtest
from src.cross_exchange import CrossBacktestConfig, run_cross_exchange_backtest


def test_single_exchange_candles_execute_selected_cycle_on_next_bar():
    signal_t = pd.Timestamp("2026-01-01T00:00:00Z")
    exec_t = pd.Timestamp("2026-01-01T00:05:00Z")
    panel = pd.DataFrame(
        [
            {
                "open_time": t,
                "symbol": symbol,
                "close": close,
                "high": high,
                "low": low,
            }
            for t, eth_low in [(signal_t, 20.0), (exec_t, 5.0)]
            for symbol, close, high, low in [
                ("BTCUSDT", 100.0, 100.0, 100.0),
                ("ETHUSDT", 10.0, 10.0, eth_low),
                ("SOLUSDT", 1.0, 1.0, 1.0),
            ]
        ]
    )

    result = run_backtest(
        panel,
        BacktestConfig(
            fee=0.0,
            slippage_bps=0.0,
            min_gross_multiplier=1.1,
            start_capital_usdt=1_000.0,
            trade_fraction=1.0,
            use_intrabar_hilo=True,
        ),
    )

    assert result.bars_scanned == 1
    assert result.opportunities_found == 1
    assert len(result.trades) == 1

    trade = result.trades[0]
    assert pd.Timestamp(trade.signal_time) == signal_t
    assert pd.Timestamp(trade.time) == exec_t
    assert trade.gross_multiplier > 1.0
    assert trade.realized_gross_multiplier is not None
    assert trade.realized_gross_multiplier < 1.0
    assert trade.pnl_usdt < 0.0


def test_cross_exchange_candles_record_loss_after_next_bar_execution():
    signal_t = pd.Timestamp("2026-01-01T00:00:00Z")
    exec_t = pd.Timestamp("2026-01-01T00:05:00Z")
    panel = pd.DataFrame(
        [
            {
                "open_time": signal_t,
                "exchange": "cheap",
                "symbol": "BTCUSDT",
                "base": "BTC",
                "close": 100.0,
                "high": 100.0,
                "low": 100.0,
            },
            {
                "open_time": signal_t,
                "exchange": "rich",
                "symbol": "BTCUSDT",
                "base": "BTC",
                "close": 103.0,
                "high": 103.0,
                "low": 103.0,
            },
            {
                "open_time": exec_t,
                "exchange": "cheap",
                "symbol": "BTCUSDT",
                "base": "BTC",
                "close": 100.0,
                "high": 100.0,
                "low": 100.0,
            },
            {
                "open_time": exec_t,
                "exchange": "rich",
                "symbol": "BTCUSDT",
                "base": "BTC",
                "close": 90.0,
                "high": 90.0,
                "low": 90.0,
            },
        ]
    )

    result = run_cross_exchange_backtest(
        panel,
        CrossBacktestConfig(
            fee=0.0,
            transfer_fee=0.0,
            half_spread_bps=0.0,
            slippage_bps=0.0,
            min_gross_multiplier=1.01,
            min_net_multiplier=1.0001,
            start_capital_usdt=1_000.0,
            trade_fraction=1.0,
            use_direct_scan=True,
        ),
    )

    assert result.bars_scanned == 1
    assert result.opportunities_found == 1
    assert len(result.trades) == 1

    trade = result.trades[0]
    assert pd.Timestamp(trade.signal_time) == signal_t
    assert pd.Timestamp(trade.time) == exec_t
    assert trade.gross_multiplier > 1.0
    assert trade.realized_gross_multiplier is not None
    assert trade.realized_gross_multiplier < 1.0
    assert trade.pnl_usdt < 0.0
