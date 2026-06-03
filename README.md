# Крипто-арбитраж: свечи и L2 order book

Реализация идеи из презентации `crypto_arbitrage_bellman_ford_5min (1).pptx`:

- **Вершины** — криптовалюты (BTC, ETH, USDT, …)
- **Рёбра** — направленный обмен с весом  
  `w(u,v) = -ln(r_uv · (1 - f_uv))`
- **Арбитраж** — отрицательный цикл: `Σ w(u,v) < 0` ⇔ `Π r_uv > 1`
- **Данные** — 5-минутные свечи и реальные L2-стаканы Tardis.dev
- **Бэктест** — комиссии, спред/слиппедж, latency, no-lookahead execution

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
учитываются latency, stale quotes и pre-funded inventory.

Текущий сохранённый L2-прогон: `2026-05-01`, notional 2000 USDT, grid 100 ms,
depth 5, max quote age 250 ms. При retail taker `0.10%` сделок нет; при low-fee
`0.01%` и latency `300ms` получается 9 сделок, +11.32 USDT за день. См.
`results/tardis_l2_report.json` и `results/CONCLUSIONS.md`.

**Главный отчёт:** `results/REPORT.html` (одна биржа / межбиржевой × с комиссиями / без).

## Текущие режимы

1. **Свечи, одна биржа** — Binance USDT-пары, implied cross, треугольники/циклы.
   Сигнал строится на закрытом баре `t`, исполнение того же маршрута на `t+1`.
2. **Свечи, несколько бирж** — Binance/Bybit/OKX, прямой cross-exchange маршрут
   `USDT@buy_ex -> asset@buy_ex` и `asset@sell_ex -> USDT@sell_ex`.
   Тоже без same-bar lookahead: сигнал `t`, исполнение `t+1`.
3. **L2/Tardis** — реальные стаканы BTC/ETH, buy по ask и sell по bid на разных
   биржах, исполнение в `t + latency_ms`, проход по глубине стакана.
   Комиссия spot считается как buy fee из купленного asset и sell fee из quote proceeds.

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

## Допущения

- Свечи — грубая модель: нет реальной очереди, L2-глубины и порядка событий внутри бара.
- L2 — pre-funded inventory: активы заранее лежат на биржах; withdrawal/rebalancing fees
  не входят в per-trade PnL.
- Основной realistic вывод делается по L2, свечи нужны как sanity-check.
- Retail taker fee: `0.10%` на ногу; low-fee сценарий `0.01%` — синтетический pro/HFT proxy.

## Выводы

См. `results/CONCLUSIONS.md` после прогона.
