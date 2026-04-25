# Physics-Informed Neural Networks for 2D Heat Equation

Студенты:
Янке Анастасия (tg: @yankeeze) — математическая постановка, архитектура PINN, реализация функции потерь (PDE loss).
Пырлицану Никита (tg: @nikita0607) — реализация численного метода (FDM) для ground truth, генерация синтетических данных, визуализация, структура репозитория.

## Project Goal
Исследование PINN для моделирования температурного поля в 2D-пластине и сравнение с:
- классическим численным решением (FDM, ground truth),
- обычным MLP без физического ограничения.

## Repository Structure
```text
.
├── data
│   ├── processed                 # Сгенерированные сетки и результаты FDM (ground truth)
│   └── raw                       # Параметры физической среды
├── models                        # Сохранённые веса обученных моделей
├── notebooks
│   ├── 01_eda_and_fdm.ipynb      # Генерация сетки и эталонного FDM
│   ├── 02_baseline_mlp.ipynb     # Baseline: MLP без PDE-loss
│   └── 03_experiments_pinn.ipynb # PINN и ablation study
├── presentation
├── report
│   ├── pic
│   ├── main.tex
│   └── report.md
├── src
│   ├── preprocessing.py          # Domain sampling, нормализация, FDM dataset
│   ├── modeling.py               # MLP/PINN, функции потерь, train loop
│   └── utils.py                  # Визуализация тепловых карт и лоссов
├── requirements.txt
└── README.md
```

## Quick Start
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data Pipeline
1. Domain sampling (collocation, BC, IC): `src/preprocessing.py`
2. Normalization to `[0, 1]` for coordinates and temperature
3. FDM solver produces ground truth field in `data/processed/pinn_dataset.npz`

Run data generation:
```bash
python -m src.preprocessing
```

## Experiments
1. `notebooks/01_eda_and_fdm.ipynb` — генерация и проверка датасета
2. `notebooks/02_baseline_mlp.ipynb` — supervised MLP baseline
3. `notebooks/03_experiments_pinn.ipynb` — PINN + ablation (loss weights / depth)

## Report
Основной отчёт: `report/main.tex`
