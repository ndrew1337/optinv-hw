"""HTML report generation for all backtest scenarios."""

from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .cross_exchange import CrossBacktestConfig, run_cross_exchange_backtest


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_equity(curve: pd.Series, title: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if not curve.empty:
        curve.plot(ax=ax, color="#2563eb", lw=1.2)
    ax.set_title(title)
    ax.set_ylabel("USDT")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_b64(fig)


def scenario_summary(name: str, result: BacktestResult, mode: str, with_fees: bool) -> Dict[str, Any]:
    s = result.summary()
    s["scenario"] = name
    s["mode"] = mode
    s["with_fees"] = with_fees
    return s


def build_html_report(
    summaries: List[Dict[str, Any]],
    equity_images: Dict[str, str],
    out_path: Path,
) -> None:
    rows = ""
    for s in summaries:
        ret = s.get("total_return_pct", 0)
        cls = "pos" if ret > 0 else "neg" if ret < 0 else ""
        rows += f"""
        <tr>
          <td>{s['scenario']}</td>
          <td>{s['mode']}</td>
          <td>{'Да' if s['with_fees'] else 'Нет'}</td>
          <td>{s.get('bars_scanned', 0)}</td>
          <td>{s.get('trades_executed', 0)}</td>
          <td class="{cls}">{ret:.2f}%</td>
          <td>{s.get('final_capital', 0):,.2f}</td>
          <td>{s.get('max_drawdown_pct', 0):.2f}%</td>
        </tr>"""

    charts = ""
    for title, b64 in equity_images.items():
        charts += f"""
        <div class="chart-block">
          <h3>{title}</h3>
          <img src="data:image/png;base64,{b64}" alt="{title}"/>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8"/>
  <title>Отчёт: крипто-арбитраж Bellman-Ford</title>
  <style>
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 2rem; color: #1e293b; max-width: 1100px; }}
    h1 {{ color: #0f172a; }}
    h2 {{ margin-top: 2rem; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.3rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 0.5rem 0.75rem; text-align: left; }}
    th {{ background: #f8fafc; }}
    .pos {{ color: #059669; font-weight: 600; }}
    .neg {{ color: #dc2626; }}
    .chart-block {{ margin: 1.5rem 0; }}
    .chart-block img {{ max-width: 100%; border: 1px solid #e2e8f0; border-radius: 8px; }}
    .note {{ background: #fffbeb; border-left: 4px solid #f59e0b; padding: 1rem; margin: 1rem 0; }}
    .method {{ background: #f0f9ff; padding: 1rem; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>Крипто-арбитраж: Форд–Беллман</h1>
  <p>Сформировано: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>

  <div class="method">
    <strong>Метод.</strong> Граф обменов с весами w = −ln(r·(1−fee)).
    Отрицательный цикл ⇒ произведение курсов &gt; 1.
    <ul>
      <li><strong>Одна биржа</strong> — треугольники на Binance (implied cross из USDT-пар).</li>
      <li><strong>Межбиржевой</strong> — вершины asset@exchange (Binance, Bybit, OKX),
          рёбра торговли + перевод монеты между биржами.</li>
    </ul>
  </div>

  <h2>Сводка сценариев</h2>
  <table>
    <tr>
      <th>Сценарий</th><th>Режим</th><th>Комиссии</th><th>Баров</th>
      <th>Сделок</th><th>Доходность</th><th>Капитал</th><th>Max DD</th>
    </tr>
    {rows}
  </table>

  <div class="note">
    <strong>Интерпретация.</strong> Свечные режимы используют закрытый 5m бар
    только для сигнала, а выбранный маршрут исполняют на следующем баре.
    С комиссиями (0.1%/сделка + перевод + спред)
    прибыль ниже, чем без комиссий — оба варианта приведены для сравнения.
  </div>

  <h2>Кривые капитала</h2>
  {charts}

  <h2>Выводы</h2>
  <ul>
    <li>На одной бирже с согласованными mid-котировками арбитраж исчезает — рынок эффективен.</li>
    <li>Межбиржевые свечи без комиссий показывают верхний потолок, но не реалистичную прибыль.</li>
    <li>С комиссиями и исполнением на следующем баре межбиржевой свечной сценарий становится отрицательным.</li>
    <li>Для реальной торговли нужны order book, latency, stale-quote контроль и инвентарная модель.</li>
    <li>Основной реалистичный вывод делается по L2/Tardis, а свечи используются как sanity-check.</li>
  </ul>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def run_all_scenarios(
    single_panel: pd.DataFrame,
    cross_panel: pd.DataFrame,
    results_dir: Path,
) -> List[Dict[str, Any]]:
    results_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, Any]] = []
    equity_images: Dict[str, str] = {}

    scenarios = [
        (
            "single_fees",
            "Одна биржа + комиссии",
            "single",
            True,
            BacktestConfig(
                fee=0.001,
                half_spread_bps=5.0,
                slippage_bps=2.0,
                min_gross_multiplier=1.002,
            ),
            None,
        ),
        (
            "single_no_fees",
            "Одна биржа без комиссий",
            "single",
            False,
            BacktestConfig(
                fee=0.0,
                half_spread_bps=0.0,
                slippage_bps=0.0,
                min_gross_multiplier=1.0000001,
            ),
            None,
        ),
        (
            "cross_fees",
            "Межбиржевой + комиссии (лаг 1 бар)",
            "cross",
            True,
            None,
            CrossBacktestConfig(
                fee=0.001,
                transfer_fee=0.0005,
                half_spread_bps=5.0,
                slippage_bps=2.0,
                min_gross_multiplier=1.0005,
                min_net_multiplier=1.0002,
                use_venue_hilo=False,
                lag_bars=1,  # signal may use a one-bar-old buy quote; execution is next bar
            ),
        ),
        (
            "cross_no_fees",
            "Межбиржевой без комиссий",
            "cross",
            False,
            None,
            CrossBacktestConfig(
                fee=0.0,
                transfer_fee=0.0,
                half_spread_bps=0.0,
                slippage_bps=0.0,
                min_gross_multiplier=1.0000001,
                min_net_multiplier=1.0000001,
                use_venue_hilo=False,
            ),
        ),
    ]

    for key, title, mode, with_fees, single_cfg, cross_cfg in scenarios:
        print(f"Running {title}...")
        if mode == "single":
            res = run_backtest(single_panel, single_cfg)
        else:
            res = run_cross_exchange_backtest(cross_panel, cross_cfg)

        summ = scenario_summary(title, res, mode, with_fees)
        summaries.append(summ)

        res.equity_curve.to_csv(results_dir / f"equity_{key}.csv", header=["equity"])
        if res.trades:
            pd.DataFrame(
                [
                    {
                        "time": t.time,
                        "signal_time": t.signal_time,
                        "cycle": " -> ".join(t.cycle),
                        "gross_mult": t.gross_multiplier,
                        "realized_gross_mult": t.realized_gross_multiplier,
                        "net_mult": t.net_multiplier,
                        "pnl": t.pnl_usdt,
                    }
                    for t in res.trades
                ]
            ).to_csv(results_dir / f"trades_{key}.csv", index=False)

        equity_images[title] = plot_equity(res.equity_curve, title)
        print(f"  return={summ.get('total_return_pct', 0):.2f}% trades={summ.get('trades_executed', 0)}")

    import json

    with open(results_dir / "all_scenarios.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False, default=str)

    build_html_report(summaries, equity_images, results_dir / "REPORT.html")
    return summaries
