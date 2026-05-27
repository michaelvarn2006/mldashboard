from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image

from dataset_meta import FIELD_VALUE_LABELS, MODEL_DISPLAY_NAMES, TARGET_COL

ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("models")
DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "data_clf.csv"
FIG_DIR = ARTIFACT_DIR / "figures"

st.set_page_config(page_title="Дашборд для вывода моделей ML", layout="wide")
sns.set_style("whitegrid")


def _safe_image(path: Path):
    try:
        return Image.open(path)
    except Exception:
        return None


def _read_json(path: Path, default: Any):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return _read_csv_flexible(DATA_PATH)


@st.cache_data
def load_metrics() -> pd.DataFrame:
    path = ARTIFACT_DIR / "metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_selected_features() -> list[str]:
    return _read_json(ARTIFACT_DIR / "selected_features.json", [])


@st.cache_data
def load_feature_schema() -> dict[str, dict[str, Any]]:
    return _read_json(ARTIFACT_DIR / "feature_schema.json", {})


@st.cache_data
def load_target_distribution() -> pd.DataFrame:
    path = ARTIFACT_DIR / "target_distribution.csv"
    if path.exists():
        return pd.read_csv(path)
    df = load_dataset()
    return (
        df[TARGET_COL]
        .value_counts()
        .sort_index()
        .rename_axis("target")
        .reset_index(name="count")
    )


@st.cache_data
def load_best_model_name() -> str:
    path = ARTIFACT_DIR / "best_model_name.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()

    metrics = load_metrics()
    if not metrics.empty and "f1" in metrics.columns:
        return str(metrics.sort_values("f1", ascending=False).iloc[0]["name"])
    return "LogisticRegression"


@st.cache_resource
def load_model_bundle(model_name: str):
    candidates = [
        MODEL_DIR / f"{model_name.lower()}.pkl",
        MODEL_DIR / f"{model_name}.pkl",
        MODEL_DIR / "best_model.pkl",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Model not found for {model_name}")

    with open(path, "rb") as f:
        return pickle.load(f)


def _positive_prob(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores
    return model.predict(X).astype(float)


def predict_with_model(model_name: str, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model = load_model_bundle(model_name)
    pred = model.predict(X)
    prob = _positive_prob(model, X)
    return pred, prob


def _get_model_summary_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def _display_name(col: str, schema: dict[str, dict[str, Any]]) -> str:
    label = schema.get(col, {}).get("label", col)
    unit = schema.get(col, {}).get("unit", "")
    if unit:
        return f"{label} ({unit})"
    return label


def _default_numeric_value(meta: dict[str, Any]) -> float:
    for key in ("median", "mean", "min"):
        if key in meta and meta[key] is not None and np.isfinite(meta[key]):
            return float(meta[key])
    return 0.0


def _build_demo_examples(schema: dict[str, dict[str, Any]]):
    typical: dict[str, Any] = {}
    outlier: dict[str, Any] = {}

    for col, meta in schema.items():
        if meta["kind"] == "numeric":
            typical[col] = _default_numeric_value(meta)
            q25 = meta.get("q25", meta.get("min", 0.0))
            q75 = meta.get("q75", meta.get("max", 0.0))
            low_risk_features = any(
                token in col.lower()
                for token in ["approved", "grade", "credited", "evaluations"]
            )
            outlier[col] = float(q25 if low_risk_features else q75)
        else:
            choices = meta.get("choices", [])
            if choices:
                typical[col] = choices[0]
                outlier[col] = choices[-1] if len(choices) > 1 else choices[0]
            else:
                typical[col] = None
                outlier[col] = None

    typical_text = (
        "Корректные данные: значения близки к медианным или наиболее типичным для набора. "
        "Это стандартный профиль студента без явных факторов риска."
    )
    outlier_text = (
        "Данные с выбросами: специально усилены рискованные признаки, например низкая успеваемость, "
        "финансовые проблемы или нетипичные значения по нагрузке. Это сценарий для проверки реакции модели."
    )
    return typical, outlier, typical_text, outlier_text


def _validate_row(row: dict[str, Any], schema: dict[str, dict[str, Any]]) -> pd.DataFrame:
    cleaned: dict[str, Any] = {}
    errors: list[str] = []

    for col, meta in schema.items():
        if col not in row:
            errors.append(f"Missing field: {col}")
            continue

        val = row[col]
        if meta["kind"] == "categorical":
            choices = meta.get("choices", [])
            if val not in choices:
                errors.append(f"{meta.get('label', col)}: invalid category")
            else:
                cleaned[col] = val
        else:
            try:
                num = float(val)
            except Exception:
                errors.append(f"{meta.get('label', col)}: must be numeric")
                continue

            if "min" in meta and num < float(meta["min"]) - 1e-12:
                errors.append(f"{meta.get('label', col)}: below minimum {meta['min']}")
            if "max" in meta and num > float(meta["max"]) + 1e-12:
                errors.append(f"{meta.get('label', col)}: above maximum {meta['max']}")
            cleaned[col] = num

    if errors:
        raise ValueError("\n".join(errors))

    return pd.DataFrame([cleaned])


def _render_input_widget(col: str, meta: dict[str, Any], default_value: Any, key_prefix: str):
    label = _display_name(col, {col: meta})

    if meta["kind"] == "categorical":
        choices = meta.get("choices", [])
        display_map = FIELD_VALUE_LABELS.get(col, {})

        def fmt(v):
            return display_map.get(v, str(v))

        idx = 0
        if default_value in choices:
            idx = choices.index(default_value)
        return st.selectbox(label, choices, index=idx, format_func=fmt, key=f"{key_prefix}_{col}")

    min_v = float(meta.get("min", 0.0))
    max_v = float(meta.get("max", 1.0))
    value = float(default_value) if default_value is not None else _default_numeric_value(meta)
    step = 1.0 if abs(max_v - min_v) > 10 else 0.1
    return st.number_input(
        label,
        value=value,
        min_value=min_v,
        max_value=max_v,
        step=step,
        key=f"{key_prefix}_{col}",
    )


def _predict_label(pred: int) -> str:
    return "Отчисление / риск" if int(pred) == 1 else "Завершит обучение / низкий риск"


def page_1():
    st.title("Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных")

    col1, col2 = st.columns([1, 2])
    with col1:
        img = _safe_image(Path("photo.jpg"))
        if img is not None:
            st.image(img, use_container_width=True)
        else:
            st.info("Поместите photo.jpg рядом с app.py, чтобы показать фото разработчика.")

    with col2:
        st.write("ФИО: Варнавский Михаил Максимович")
        st.write("Группа: ФИТ-242")
        st.write("Стек технологий: Streamlit, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib, Seaborn")


def page_2(df: pd.DataFrame):
    st.title("Набор данных и подготовка")

    schema = load_feature_schema()
    selected_features = load_selected_features()
    target_dist = load_target_distribution()

    st.write("Целевая колонка:", TARGET_COL)
    st.write("Размер набора данных:", df.shape)

    left, right = st.columns([1.1, 0.9])

    with left:
        st.subheader("Описание задачи")
        st.write(
            "В проекте используется датасет по отчислению студентов из UCI для бинарной задачи классификации со следующими классами: отчислен(класс 0) и завершил обучение(класс 1). "
        )
        st.write(
            "Приложение работает с сырыми данными пользователя. Пропущенные значения заполняются, категориальные значения кодируются, "
            "а числовые признаки масштабируются внутри модели."
        )
        st.write(
            "Для интерпретируемости и ручной формы прогнозирования сохраняются только наиболее важные 15 признаков."
        )

    with right:
        st.subheader("Баланс классов")
        st.dataframe(target_dist, use_container_width=True)

    st.subheader("Выбранные признаки")
    if schema and selected_features:
        show_df = pd.DataFrame([schema[c] for c in selected_features if c in schema])
        cols = [c for c in ["label", "kind", "unit", "missing", "min", "median", "max"] if c in show_df.columns]
        st.dataframe(show_df[cols], use_container_width=True)
    else:
        st.warning("Схема не найдена. Сначала запустите train_models.py.")

    st.subheader("Первые строки")
    st.dataframe(df.head(10), use_container_width=True)


def page_3():
    st.title("Визуализации")

    fig_files = [
        "00_model_f1_comparison.png",
        "01_class_distribution.png",
        "cm_BaggingClassifier.png",
        "03_feature_distributions.png",
    ]

    model_names = [
        "DecisionTreeClassifier",
        "AdaBoostClassifier",
        "BaggingClassifier",
        "MLPClassifier",
        "StackingClassifier",
        "XGBClassifier",
        "GradientBoostingClassifier",
    ]

    for name in model_names:
        fig_files.extend([f"roc_{name}.png", f"cm_{name}.png", f"fi_{name}.png"])

    available = [FIG_DIR / fname for fname in fig_files if (FIG_DIR / fname).exists()]
    if not available:
        st.warning("Графики не найдены. Сначала запустите train_models.py.")
        return

    for start in range(0, len(available), 2):
        cols = st.columns(2)
        for idx, path in enumerate(available[start:start + 2]):
            with cols[idx]:
                st.image(str(path), use_container_width=True)


def page_4(df: pd.DataFrame):
    st.title("Прогноз")

    schema = load_feature_schema()
    selected_features = load_selected_features()
    metrics = load_metrics()

    if not schema or not selected_features:
        st.error("Артефакты отсутствуют. Сначала запустите train_models.py.")
        return

    if metrics.empty:
        model_options = [
            "DecisionTreeClassifier",
            "AdaBoostClassifier",
            "BaggingClassifier",
            "MLPClassifier",
            "StackingClassifier",
            "XGBClassifier",
            "GradientBoostingClassifier",
        ]
    else:
        model_options = metrics["name"].astype(str).tolist()

    default_model = load_best_model_name()
    if default_model not in model_options:
        model_options.insert(0, default_model)

    selected_model = st.selectbox(
        "Модель",
        model_options,
        index=model_options.index(default_model),
    )

    typical, outlier, typical_text, outlier_text = _build_demo_examples({c: schema[c] for c in selected_features})
    if "demo_mode" not in st.session_state:
        st.session_state.demo_mode = "typical"

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Загрузить типичный пример"):
            st.session_state.demo_mode = "typical"
            st.rerun()
    with c2:
        if st.button("Загрузить пример с выбросами"):
            st.session_state.demo_mode = "outlier"
            st.rerun()

    preset = typical if st.session_state.demo_mode == "typical" else outlier
    preset_text = typical_text if st.session_state.demo_mode == "typical" else outlier_text

    st.info(preset_text)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Ручной ввод")
        with st.form("manual_predict"):
            user_values: dict[str, Any] = {}
            cols = st.columns(2)
            for i, col in enumerate(selected_features):
                meta = schema[col]
                with cols[i % 2]:
                    _render_input_widget(col, meta, preset.get(col), key_prefix=st.session_state.demo_mode)
                    user_values[col] = st.session_state[f"{st.session_state.demo_mode}_{col}"]

            submit = st.form_submit_button("Прогноз")

        if submit:
            try:
                X_user = _validate_row(user_values, {c: schema[c] for c in selected_features})
                pred, prob = predict_with_model(selected_model, X_user)
                label = _predict_label(int(pred[0]))
                st.success(f"Прогноз: {label}")
                st.metric("Вероятность отчисления", f"{prob[0] * 100:.1f}%")
                st.caption("Сырые данные проверяются в интерфейсе и преобразуются внутри модельного пайплайна.")
            except Exception as exc:
                st.error(str(exc))

    with right:
        st.subheader("Интерпретация")
        st.write(f"Выбранная модель: {_get_model_summary_name(selected_model)}")
        st.write(
            "Пользователь вводит сырые значения в естественной форме. Затем пайплайн применяет заполнение пропусков, "
            "категориальное кодирование, масштабирование и инференс."
        )
        st.write("Вывод представлен как бинарное решение и вероятность в процентах.")
        st.info(f"Текущий набор: {st.session_state.demo_mode}")

    st.divider()
    st.subheader("Пакетный прогноз из CSV")
    uploaded = st.file_uploader("Загрузите CSV", type=["csv"])

    if uploaded is not None:
        incoming = pd.read_csv(uploaded)
        st.write("Предпросмотр")
        st.dataframe(incoming.head(), use_container_width=True)

        if st.button("Выполнить пакетный прогноз"):
            try:
                missing = [c for c in selected_features if c not in incoming.columns]
                if missing:
                    raise ValueError(f"Отсутствуют обязательные столбцы: {', '.join(missing)}")

                X = incoming[selected_features].copy()
                model = load_model_bundle(selected_model)
                preds = model.predict(X)
                probs = _positive_prob(model, X)

                result = incoming.copy()
                result["prediction"] = [_predict_label(int(x)) for x in preds]
                result["dropout_probability"] = probs
                result["dropout_probability_percent"] = (probs * 100).round(2)

                st.success("Прогноз завершён.")
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Скачать predictions.csv",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(str(exc))


def main():
    df = load_dataset()

    if "page" not in st.session_state:
        st.session_state.page = "1. Разработчик"

    pages = ["1. Разработчик", "2. Набор данных", "3. Визуализации", "4. Прогноз"]

    st.sidebar.title("Навигация")
    st.session_state.page = st.sidebar.radio(
        "Раздел",
        pages,
        index=pages.index(st.session_state.page),
    )

    if st.session_state.page == "1. Разработчик":
        page_1()
    elif st.session_state.page == "2. Набор данных":
        page_2(df)
    elif st.session_state.page == "3. Визуализации":
        page_3()
    else:
        page_4(df)


if __name__ == "__main__":
    main()