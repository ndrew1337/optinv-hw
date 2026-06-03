# Крипто-арбитраж: алгоритм Форда–Беллмана (5m)

Реализация идеи из презентации `crypto_arbitrage_bellman_ford_5min (1).pptx`:

- **Вершины** — криптовалюты (BTC, ETH, USDT, …)
- **Рёбра** — направленный обмен с весом  
  `w(u,v) = -ln(r_uv · (1 - f_uv))`
- **Арбитраж** — отрицательный цикл: `Σ w(u,v) < 0` ⇔ `Π r_uv > 1`
- **Данные** — 5-минутные свечи Binance Spot (close как mid)
- **Бэктест** — комиссия на каждое ребро, опционально slippage (bps)

## Быстрый старт

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
python3 run_report.py --days 14    # все 4 сценария + HTML-отчёт
python3 run_backtest.py --days 14  # только одна биржа
python3 run_analysis.py
python3 run_tardis.py --date 2026-05-01   # реальный L2 order book (Tardis.dev free)
```

## Проверка на реальном стакане (L2)

`run_tardis.py` тестирует межбиржевой арбитраж на **тиковых данных order book**
(не на 5m-close): бесплатный сэмпл Tardis.dev (`book_snapshot_25`, первый день месяца,
без ключа), BTC/ETH на Binance/Bybit/OKX. Проскальзывание — проходом по стакану,
учитываются латентность и инвентарная модель. Вывод: 172k сырых пересечений цен в
сутки, но безубыточность лишь при комиссии ≈0.02–0.04%/ногу — при ретейл-комиссии
прибыли нет. См. `results/tardis_l2_report.json` и раздел в `results/CONCLUSIONS.md`.

**Главный отчёт:** `results/REPORT.html` (одна биржа / межбиржевой × с комиссиями / без).

## Структура

| Файл | Назначение |
|------|------------|
| `src/graph.py` | Граф обменов и веса рёбер |
| `src/bellman_ford.py` | Поиск отрицательных циклов |
| `src/data.py` | Загрузка klines Binance |
| `src/backtest.py` | Симуляция PnL по барам |
| `run_backtest.py` | Основной прогон |
| `run_analysis.py` | Диагностика рынка и fee-sweep |
| `run_report.py` | 4 сценария + `results/REPORT.html` |
| `src/multi_exchange_data.py` | Binance + Bybit + OKX |
| `src/cross_exchange.py` | Межбиржевой граф и бэктест |
| `src/tardis_data.py` | Загрузка бесплатного L2-стакана Tardis.dev |
| `src/tardis_backtest.py` | Event-driven бэктест по стакану (слиппедж, латентность) |
| `run_tardis.py` | Прогон на реальных L2-данных |

## Допущения (можно уточнить)

- Одна биржа (Binance), без cross-exchange
- Цена — close 5m (без order book)
- Комиссия по умолчанию 0.1% на сделку (taker)
- Slippage по умолчанию 5 bps на ногу

## Выводы

См. `results/CONCLUSIONS.md` после прогона.
