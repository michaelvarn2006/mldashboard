from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from dataset_meta import (
    BINARY_VARS,
    CATEGORICAL_VARS,
    FIELD_LABELS,
    FIELD_UNITS,
    MODEL_DISPLAY_NAMES,
    NUMERIC_VARS,
    RAW_DATA_CANDIDATES,
    RANDOM_STATE,
    TARGET_COL,
    TOP_K,
)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

warnings.filterwarnings("ignore")

ARTIFACT_DIR = Path("artifacts")
MODEL_DIR = Path("models")
DATA_DIR = Path("data")
FIG_DIR = ARTIFACT_DIR / "figures"


@dataclass
class ModelResult:
    name: str
    f1: float
    accuracy: float
    precision: float
    recall: float
    roc_auc: float


def _find_dataset_path() -> Path:
    for p in RAW_DATA_CANDIDATES:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Dataset not found. Tried: {', '.join(str(Path(p)) for p in RAW_DATA_CANDIDATES)}"
    )


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path)


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def pretty_label(name: str) -> str:
    return FIELD_LABELS.get(name, name)


def infer_kind(col: str, series: pd.Series) -> str:
    if col in NUMERIC_VARS:
        return "numeric"
    if col in CATEGORICAL_VARS:
        return "categorical"
    if series.dtype == "object":
        return "categorical"
    nunique = series.dropna().nunique()
    if nunique <= 2:
        return "categorical"
    if pd.api.types.is_integer_dtype(series) and nunique <= 12:
        return "categorical"
    return "numeric"


def normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found.")

    out = df.copy()
    y = out[TARGET_COL]

    if pd.api.types.is_numeric_dtype(y):
        uniq = sorted(pd.Series(y).dropna().unique().tolist())
        if len(uniq) == 2:
            out[TARGET_COL] = y.astype(int)
            return out

    y_str = y.astype(str).str.strip().str.lower()

    if set(y_str.unique()).issuperset({"dropout", "enrolled", "graduate"}):
        out = out[y_str.isin(["dropout", "graduate"])].copy()
        y_str = out[TARGET_COL].astype(str).str.strip().str.lower()
        out[TARGET_COL] = (y_str == "dropout").astype(int)
        return out

    if set(y_str.unique()).issubset({"0", "1"}):
        out[TARGET_COL] = y_str.astype(int)
        return out

    if set(y_str.unique()) == {"dropout", "graduate"}:
        out[TARGET_COL] = (y_str == "dropout").astype(int)
        return out

    raise ValueError(
        "Unknown target format. Expected binary labels or original dropout/enrolled/graduate labels."
    )


def load_dataset() -> pd.DataFrame:
    path = _find_dataset_path()
    df = _read_csv_flexible(path)
    df = normalize_target(df)
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()
    return X, y


def build_schema(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, Any]]:
    schema: dict[str, dict[str, Any]] = {}
    for col in columns:
        s = df[col]
        kind = infer_kind(col, s)
        meta: dict[str, Any] = {
            "name": col,
            "label": pretty_label(col),
            "kind": kind,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unit": FIELD_UNITS.get(col, ""),
        }
        if kind == "numeric":
            num = pd.to_numeric(s, errors="coerce")
            meta.update(
                {
                    "min": float(num.min()),
                    "max": float(num.max()),
                    "mean": float(num.mean()),
                    "std": float(num.std()),
                    "q25": float(num.quantile(0.25)),
                    "median": float(num.median()),
                    "q75": float(num.quantile(0.75)),
                }
            )
        else:
            vals = pd.Series(s).dropna().unique().tolist()
            try:
                vals = sorted(vals)
            except Exception:
                vals = list(vals)
            meta["choices"] = [int(v) if isinstance(v, (np.integer, bool)) else v for v in vals]
            meta["default"] = meta["choices"][0] if meta["choices"] else None
        schema[col] = meta
    return schema


def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = [c for c in feature_df.columns if infer_kind(c, feature_df[c]) == "numeric"]
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _make_ohe()),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_bagging_classifier():
    tree = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
    try:
        return BaggingClassifier(
            estimator=tree,
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except TypeError:
        return BaggingClassifier(
            base_estimator=tree,
            n_estimators=150,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )


def build_selector_model() -> Any:
    if XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        )
    return GradientBoostingClassifier(random_state=RANDOM_STATE)


def build_logreg_variants() -> dict[str, Any]:
    return {
        "LogisticRegression_L2": LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression_L1": LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression_ElasticNet": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            max_iter=8000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }


def build_models(best_logreg_model: Any) -> dict[str, Any]:
    models: dict[str, Any] = {
        "LogisticRegression": best_logreg_model,
        "AdaBoostClassifier": AdaBoostClassifier(
            n_estimators=250,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
        "BaggingClassifier": make_bagging_classifier(),
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "StackingClassifier": StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
                ("dt", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
                ("ab", AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE)),
            ],
            final_estimator=LogisticRegression(max_iter=3000),
            cv=3,
            n_jobs=-1,
        ),
    }

    if XGBClassifier is not None:
        models["XGBClassifier"] = XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        )
    else:
        models["GradientBoostingClassifier"] = GradientBoostingClassifier(random_state=RANDOM_STATE)

    return models


def fit_pipeline(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    preprocessor = build_preprocessor(X_train)
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def predict_scores(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    if hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores
    return pipe.predict(X).astype(float)


def evaluate(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, name: str) -> ModelResult:
    y_pred = pipe.predict(X_test)
    y_prob = predict_scores(pipe, X_test)
    return ModelResult(
        name=name,
        f1=f1_score(y_test, y_pred),
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        roc_auc=roc_auc_score(y_test, y_prob),
    )


def save_figure(fig, filename: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_eda_plots(df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x=TARGET_COL, data=df, ax=ax)
    ax.set_title("Target distribution (0=graduate, 1=dropout)")
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    save_figure(fig, "01_class_distribution.png")

    numeric_df = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(corr, cmap="viridis", ax=ax)
        ax.set_title("Correlation heatmap")
        save_figure(fig, "02_correlation_heatmap.png")

    show_cols = list(numeric_df.columns[:3]) if numeric_df.shape[1] else []
    if show_cols:
        fig, axes = plt.subplots(1, len(show_cols), figsize=(5 * len(show_cols), 4))
        if len(show_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, show_cols):
            sns.violinplot(x=TARGET_COL, y=df[col], data=df, ax=ax)
            ax.set_title(col)
        save_figure(fig, "03_feature_distributions.png")


def save_metrics_plot(results: list[ModelResult]):
    metrics_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=metrics_df, x="f1", y="name", ax=ax)
    ax.set_title("Model comparison by F1")
    ax.set_xlabel("F1")
    ax.set_ylabel("Model")
    save_figure(fig, "00_model_f1_comparison.png")


def save_model_diagnostics(name: str, pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = pipe.predict(X_test)
    y_prob = predict_scores(pipe, X_test)

    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f"ROC: {name}")
    save_figure(fig, f"roc_{name}.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["graduate", "dropout"],
        ax=ax,
        cmap="Blues",
        values_format="d",
    )
    ax.set_title(f"Confusion matrix: {name}")
    save_figure(fig, f"cm_{name}.png")

    if hasattr(pipe.named_steps["model"], "feature_importances_") or hasattr(pipe.named_steps["model"], "coef_"):
        if hasattr(pipe.named_steps["model"], "feature_importances_"):
            importances = np.asarray(pipe.named_steps["model"].feature_importances_)
        else:
            importances = np.abs(np.asarray(pipe.named_steps["model"].coef_)[0])

        feat_names = list(X_test.columns)
        if len(importances) != len(feat_names):
            perm = permutation_importance(
                pipe,
                X_test,
                y_test,
                scoring="f1",
                n_repeats=7,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            importances = perm.importances_mean

        imp_df = (
            pd.DataFrame({"feature": feat_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(10)
        )
    else:
        perm = permutation_importance(
            pipe,
            X_test,
            y_test,
            scoring="f1",
            n_repeats=7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        imp_df = (
            pd.DataFrame({"feature": list(X_test.columns), "importance": perm.importances_mean})
            .sort_values("importance", ascending=False)
            .head(10)
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax)
    ax.set_title(f"Feature importance: {name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    save_figure(fig, f"fi_{name}.png")


def save_feature_importance(pipe: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, feature_names: list[str], name: str):
    perm = permutation_importance(
        pipe,
        X_val,
        y_val,
        scoring="f1",
        n_repeats=7,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    imp_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": perm.importances_mean,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(ARTIFACT_DIR / f"feature_importance_{name}.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    top = imp_df.head(min(10, len(imp_df)))
    sns.barplot(data=top, x="importance", y="feature", ax=ax)
    ax.set_title(f"Permutation importance: {name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    save_figure(fig, f"fi_{name}.png")
    return imp_df


def save_schema(schema: dict[str, dict[str, Any]]):
    with open(ARTIFACT_DIR / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)


def save_artifacts(
    df: pd.DataFrame,
    selected_features: list[str],
    results: list[ModelResult],
    fitted_models: dict[str, Pipeline],
    best_name: str,
    schema: dict[str, dict[str, Any]],
    best_logreg_variant: str,
):
    ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    DATA_DIR.mkdir(exist_ok=True, parents=True)

    df.to_csv(DATA_DIR / "data_clf.csv", index=False)

    metrics_df = pd.DataFrame([r.__dict__ for r in results]).sort_values("f1", ascending=False)
    metrics_df.to_csv(ARTIFACT_DIR / "metrics.csv", index=False)

    with open(ARTIFACT_DIR / "best_model_name.txt", "w", encoding="utf-8") as f:
        f.write(best_name)

    with open(ARTIFACT_DIR / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump(selected_features, f, ensure_ascii=False, indent=2)

    with open(ARTIFACT_DIR / "best_logreg_variant.txt", "w", encoding="utf-8") as f:
        f.write(best_logreg_variant)

    save_schema(schema)

    for name, pipe in fitted_models.items():
        with open(MODEL_DIR / f"{name.lower()}.pkl", "wb") as f:
            pickle.dump(pipe, f)

    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(fitted_models[best_name], f)

    pd.DataFrame([schema[c] for c in selected_features]).to_csv(
        ARTIFACT_DIR / "selected_feature_profile.csv",
        index=False,
    )


def choose_best_logreg_variant(X_train, y_train, X_val, y_val) -> tuple[Any, str]:
    variants = build_logreg_variants()
    rows = []
    best_name = None
    best_score = -1.0

    for name, model in variants.items():
        pipe = fit_pipeline(model, X_train, y_train)
        result = evaluate(pipe, X_val, y_val, name)
        rows.append(result.__dict__)
        if result.f1 > best_score:
            best_score = result.f1
            best_name = name

    pd.DataFrame(rows).sort_values("f1", ascending=False).to_csv(
        ARTIFACT_DIR / "logreg_variants_validation.csv",
        index=False,
    )
    assert best_name is not None
    return variants[best_name], best_name


def main():
    df = load_dataset()
    X, y = split_features_target(df)

    save_eda_plots(df)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    selector_model = build_selector_model()
    selector_pipe = fit_pipeline(selector_model, X_train, y_train)

    importance_df = save_feature_importance(
        selector_pipe,
        X_val,
        y_val,
        list(X_train.columns),
        "selector",
    )

    selected_features = importance_df.head(min(TOP_K, len(importance_df)))["feature"].tolist()
    selected_features = [c for c in selected_features if c in X.columns]

    schema = build_schema(df, selected_features)

    X_train_sel = X_train_full[selected_features].copy()
    X_test_sel = X_test[selected_features].copy()
    X_logreg_train, X_logreg_val, y_logreg_train, y_logreg_val = train_test_split(
        X_train_sel,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    best_logreg_model, best_logreg_variant = choose_best_logreg_variant(
        X_logreg_train,
        y_logreg_train,
        X_logreg_val,
        y_logreg_val,
    )

    models = build_models(best_logreg_model)

    results: list[ModelResult] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        pipe = fit_pipeline(model, X_train_sel, y_train_full)
        result = evaluate(pipe, X_test_sel, y_test, name)
        results.append(result)
        fitted_models[name] = pipe
        save_model_diagnostics(name, pipe, X_test_sel, y_test)
        print(f"{name}: F1={result.f1:.4f}, ACC={result.accuracy:.4f}, AUC={result.roc_auc:.4f}")

    results_sorted = sorted(results, key=lambda x: x.f1, reverse=True)
    best_name = results_sorted[0].name

    save_metrics_plot(results_sorted)
    save_artifacts(
        df=df,
        selected_features=selected_features,
        results=results_sorted,
        fitted_models=fitted_models,
        best_name=best_name,
        schema=schema,
        best_logreg_variant=best_logreg_variant,
    )

    print("\nSelected features:")
    for c in selected_features:
        print("-", c)

    print(f"\nBest model: {best_name}")
    print(f"Best logreg variant: {best_logreg_variant}")
    print("Artifacts saved to ./artifacts and ./models")


if __name__ == "__main__":
    main()