from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.modeling import MLP, PINN, load_model_weights
from src.preprocessing import denormalize, normalize

DEFAULT_WEIGHTS = {
    "PINN": ROOT_DIR / "models" / "pinn_main.pt",
    "MLP": ROOT_DIR / "models" / "mlp_baseline.pt",
}
META_PATH = ROOT_DIR / "data" / "processed" / "pinn_dataset_meta.json"


def load_meta() -> dict:
    default = {
        "normalization": {
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "t": [0.0, 1.0],
            "temperature": [20.0, 60.0],
        }
    }
    if not META_PATH.exists():
        return default
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def build_model(model_name: str, hidden_dim: int, num_hidden_layers: int) -> torch.nn.Module:
    model_cls = PINN if model_name == "PINN" else MLP
    return model_cls(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)


def infer_arch_from_state_dict(weights_path: str) -> tuple[int, int]:
    state = torch.load(weights_path, map_location="cpu")
    linear_weight_keys = sorted(
        [k for k in state.keys() if k.startswith("net.") and k.endswith(".weight")],
        key=lambda k: int(k.split(".")[1]),
    )
    if len(linear_weight_keys) < 2:
        raise ValueError("Не удалось определить архитектуру по state_dict.")

    first_w = state[linear_weight_keys[0]]
    hidden_dim = int(first_w.shape[0])
    num_hidden_layers = len(linear_weight_keys) - 1
    return hidden_dim, num_hidden_layers


@st.cache_resource
def load_ready_model(
    model_name: str,
    weights_path: str,
    hidden_dim: int,
    num_hidden_layers: int,
) -> torch.nn.Module:
    model = build_model(model_name=model_name, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers)
    return load_model_weights(model, weights_path, map_location="cpu")


def predict_single(
    model: torch.nn.Module,
    x_phys: float,
    y_phys: float,
    t_phys: float,
    mins_maxs: dict,
) -> float:
    x_norm = normalize(np.array([x_phys]), *mins_maxs["x"])[0]
    y_norm = normalize(np.array([y_phys]), *mins_maxs["y"])[0]
    t_norm = normalize(np.array([t_phys]), *mins_maxs["t"])[0]
    xyt = torch.tensor([[x_norm, y_norm, t_norm]], dtype=torch.float32)
    with torch.no_grad():
        temp_norm = model(xyt).cpu().numpy().squeeze()
    return float(denormalize(np.array([temp_norm]), *mins_maxs["temperature"])[0])


def predict_grid(
    model: torch.nn.Module,
    t_phys: float,
    mins_maxs: dict,
    n_points: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_vals = np.linspace(*mins_maxs["x"], n_points)
    y_vals = np.linspace(*mins_maxs["y"], n_points)
    xx, yy = np.meshgrid(x_vals, y_vals)
    tt = np.full_like(xx, t_phys)

    x_norm = normalize(xx.ravel(), *mins_maxs["x"])
    y_norm = normalize(yy.ravel(), *mins_maxs["y"])
    t_norm = normalize(tt.ravel(), *mins_maxs["t"])
    xyt = torch.tensor(np.column_stack([x_norm, y_norm, t_norm]), dtype=torch.float32)

    with torch.no_grad():
        pred_norm = model(xyt).cpu().numpy().reshape(xx.shape)
    pred_temp = denormalize(pred_norm, *mins_maxs["temperature"])
    return xx, yy, pred_temp


st.set_page_config(page_title="PINN/MLP Inference", layout="wide")
st.title("GUI для инференса PINN/MLP")
st.caption("2D heat equation: ввод параметров точки и построение карты температуры.")

meta = load_meta()
mins_maxs = meta["normalization"]

with st.sidebar:
    st.header("Настройки модели")
    model_name = st.selectbox("Тип модели", ["PINN", "MLP"])
    default_path = str(DEFAULT_WEIGHTS[model_name])
    weights_path = st.text_input("Путь к весам (.pt)", value=default_path)

    load_clicked = st.button("Загрузить модель", type="primary")
    if load_clicked:
        st.cache_resource.clear()

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if load_clicked:
    try:
        path_obj = Path(weights_path)
        if not path_obj.exists():
            st.error(f"Файл весов не найден: {path_obj}")
        else:
            used_hidden_dim, used_num_hidden_layers = infer_arch_from_state_dict(str(path_obj))
            st.info(
                f"Архитектура из весов: hidden_dim={used_hidden_dim}, "
                f"num_hidden_layers={used_num_hidden_layers}"
            )
            model = load_ready_model(
                model_name,
                str(path_obj),
                used_hidden_dim,
                used_num_hidden_layers,
            )
            st.session_state.model_loaded = True
            st.session_state.model = model
            st.success("Модель успешно загружена.")
    except Exception as exc:
        st.session_state.model_loaded = False
        st.error(f"Не удалось загрузить модель: {exc}")

if not st.session_state.model_loaded:
    st.info("Нажмите 'Загрузить модель' в боковой панели.")
    st.stop()

model = st.session_state.model

col1, col2 = st.columns(2)

with col1:
    st.subheader("Точечный инференс")
    x_val = st.number_input("x", min_value=float(mins_maxs["x"][0]), max_value=float(mins_maxs["x"][1]), value=0.5)
    y_val = st.number_input("y", min_value=float(mins_maxs["y"][0]), max_value=float(mins_maxs["y"][1]), value=0.5)
    t_val = st.number_input("t", min_value=float(mins_maxs["t"][0]), max_value=float(mins_maxs["t"][1]), value=0.5)

    if st.button("Предсказать температуру"):
        temp_pred = predict_single(model, x_val, y_val, t_val, mins_maxs=mins_maxs)
        st.metric("Предсказанная температура", f"{temp_pred:.3f}")

with col2:
    st.subheader("Карта температуры")
    t_map = st.slider(
        "Время t для карты",
        min_value=float(mins_maxs["t"][0]),
        max_value=float(mins_maxs["t"][1]),
        value=float(mins_maxs["t"][0]),
    )
    grid_size = st.slider("Размер сетки", min_value=20, max_value=150, value=60, step=10)
    if st.button("Построить карту"):
        xx, yy, pred_temp = predict_grid(model, t_phys=t_map, mins_maxs=mins_maxs, n_points=grid_size)
        fig, ax = plt.subplots(figsize=(6, 5))
        contour = ax.contourf(xx, yy, pred_temp, levels=25, cmap="inferno")
        fig.colorbar(contour, ax=ax, label="Temperature")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Predicted Temperature Field (t={t_map:.3f})")
        st.pyplot(fig)
