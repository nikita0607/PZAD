# Streamlit GUI for Inference

Приложение для инференса моделей `PINN` и `MLP` из проекта:
- ввод точки `(x, y, t)`;
- предсказание температуры;
- построение 2D-карты температурного поля.

## Запуск

Из корня проекта:

```bash
pip install -r requirements.txt
streamlit run streamlit/app.py
```

## Что нужно для работы

1. Сгенерированные метаданные нормализации:
```bash
python -m src.preprocessing
```
2. Файл весов модели (`.pt`), например:
- `models/pinn_main.pt`
- `models/mlp_baseline.pt`

Путь к весам задается в боковой панели приложения.
