from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, RationalQuadratic, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="The provided functions are not strictly inverse")

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train_dataset.csv"
TEST_PATH = ROOT / "test_dataset.csv"
TRAINER_NEW_PATH = ROOT / "trainer_new.csv"
MLMONSTERS_PATH = ROOT / "MLMonsters.csv"
PROBLEM_PDF_PATH = ROOT / "Problem Statement.pdf"
PLOTS_DIR = ROOT / "plots"
REACTOR_REFERENCE_PATH = next(iter(sorted(ROOT.glob("*Reactor*.pdf"))), None)

RAW_FEATURES = [
    "flow_rate_L_min",
    "concentration_mol_L",
    "inlet_temperature_K",
    "length_m",
    "jacket_temperature_K",
]
TARGET = "overall_yield"
REMOVED_INDICES = [97, 77, 149, 86, 30, 23, 142, 18, 57, 26]
CV_SEEDS = [42, 101, 2023, 7, 888]
N_SPLITS = 5
ARRHENIUS_GRID = (3000.0, 5000.0, 7000.0)
LOW_MID_HIGH_BINS = (
    ("low_yield_rmse", lambda y: y < 10.0),
    ("mid_yield_rmse", lambda y: (y >= 10.0) & (y <= 70.0)),
    ("high_yield_rmse", lambda y: y > 70.0),
)
EXTREME_BINS = (
    ("yield_lt_10_rmse", lambda y: y < 10.0),
    ("yield_10_30_rmse", lambda y: (y >= 10.0) & (y < 30.0)),
    ("yield_30_70_rmse", lambda y: (y >= 30.0) & (y < 70.0)),
    ("yield_70_90_rmse", lambda y: (y >= 70.0) & (y <= 90.0)),
    ("yield_gt_90_rmse", lambda y: y > 90.0),
)

def log(message: str) -> None:
    print(message, flush=True)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def safe_rmse(y_true, y_pred, mask):
    if int(mask.sum()) == 0:
        return float("nan")
    return rmse(y_true[mask], y_pred[mask])

def clip_yield(y_pred):
    return np.clip(np.asarray(y_pred, dtype=float), 0.0, 100.0)

def yield_to_logit(y):
    return logit(np.clip(np.asarray(y, dtype=float) / 100.0, 1e-4, 1.0 - 1e-4))

def logit_to_yield(z):
    return 100.0 * expit(np.asarray(z, dtype=float))

def format_float(value, digits=4):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "nan"
    return f"{value:.{digits}f}"

def markdown_table(rows, columns):
    widths = {col: max(len(col), max((len(str(row.get(col, ""))) for row in rows), default=0)) for col in columns}
    header = "| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |"
    divider = "| " + " | ".join("-" * widths[col] for col in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])

def analytic_physics_predict(params: np.ndarray, F: np.ndarray, T_in: np.ndarray, L: np.ndarray, T_j: np.ndarray) -> np.ndarray:
    log_alpha, log_A1, E1, log_A2, E2, log_scale = params
    alpha = np.exp(log_alpha)
    A1 = np.exp(log_A1)
    A2 = np.exp(log_A2)
    scale = np.exp(log_scale)

    tau = L / F
    T_eff = T_j - (T_j - T_in) * np.exp(-alpha * tau)
    T_eff = np.clip(T_eff, 100.0, 1000.0)

    k1 = A1 * np.exp(-E1 / T_eff)
    k2 = A2 * np.exp(-E2 / T_eff)
    diff = k2 - k1
    safe_sign = np.where(diff >= 0.0, 1.0, -1.0)
    diff = np.where(np.abs(diff) < 1e-5, 1e-5 * safe_sign, diff)

    pred = scale * (k1 / diff) * (np.exp(-k1 * tau) - np.exp(-k2 * tau)) * 100.0
    return clip_yield(pred)

def fit_analytic_physics(train_df, y_train):
    F = train_df["flow_rate_L_min"].to_numpy()
    T_in = train_df["inlet_temperature_K"].to_numpy()
    L = train_df["length_m"].to_numpy()
    T_j = train_df["jacket_temperature_K"].to_numpy()

    initial_guess = np.array([0.0, 15.0, 5000.0, 18.0, 7000.0, 0.0], dtype=float)
    bounds = [
        (-5.0, 5.0),
        (5.0, 30.0),
        (1000.0, 15000.0),
        (5.0, 35.0),
        (1000.0, 18000.0),
        (-1.0, 1.0),
    ]
    result = minimize(
        lambda params: np.mean((y_train - analytic_physics_predict(params, F, T_in, L, T_j)) ** 2),
        initial_guess,
        method="L-BFGS-B",
        bounds=bounds,
    )
    return result.x

def simulate_pfr(params, F, C0, T_in, L, T_j, n_steps=24):
    log_alpha, log_A1, E1, log_A2, E2, log_scale, order_a, order_b = params
    alpha = np.exp(log_alpha)
    A1 = np.exp(log_A1)
    A2 = np.exp(log_A2)
    scale = np.exp(log_scale)

    CA = C0.astype(float).copy()
    CB = np.zeros_like(CA)
    tau = L / F
    dt = tau / n_steps

    for step in range(n_steps):
        t = step * dt
        T = T_j - (T_j - T_in) * np.exp(-alpha * t)
        T = np.clip(T, 100.0, 1000.0)
        k1 = A1 * np.exp(-E1 / T)
        k2 = A2 * np.exp(-E2 / T)
        r1 = k1 * (CA ** order_a)
        r2 = k2 * (CB ** order_b)

        amount_A = np.minimum(CA, r1 * dt)
        CA = CA - amount_A

        amount_B = np.minimum(CB, r2 * dt)
        CB = np.clip(CB + amount_A - amount_B, 0.0, None)

    return clip_yield(scale * (CB / np.clip(C0, 1e-6, None)) * 100.0)

def fit_pfr_ode(train_df, y_train, n_steps=24):
    F = train_df["flow_rate_L_min"].to_numpy()
    C0 = train_df["concentration_mol_L"].to_numpy()
    T_in = train_df["inlet_temperature_K"].to_numpy()
    L = train_df["length_m"].to_numpy()
    T_j = train_df["jacket_temperature_K"].to_numpy()
    starts = [
        np.array([0.0, 15.0, 5000.0, 18.0, 7000.0, 0.0, 1.0, 1.0], dtype=float),
        np.array([0.5, 14.0, 4500.0, 18.5, 8500.0, -0.1, 1.2, 0.8], dtype=float),
    ]
    bounds = [
        (-5.0, 5.0),
        (5.0, 30.0),
        (1000.0, 15000.0),
        (5.0, 35.0),
        (1000.0, 18000.0),
        (-1.0, 1.0),
        (0.2, 3.0),
        (0.2, 3.0),
    ]
    best_result = None
    for initial_guess in starts:
        result = minimize(
            lambda params: np.mean((y_train - simulate_pfr(params, F, C0, T_in, L, T_j, n_steps=n_steps)) ** 2),
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 250},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    assert best_result is not None
    return best_result.x

@dataclass
class PhysicsFeatureBuilder:
    use_physics: bool
    arrhenius_grid: tuple[float, ...] = ARRHENIUS_GRID
    pfr_steps: int = 24
    physics_params_: np.ndarray | None = None

    def fit(self, X_raw, y=None):
        if self.use_physics:
            if y is None:
                raise ValueError("PhysicsFeatureBuilder requires targets during fit when use_physics=True.")
            self.physics_params_ = fit_pfr_ode(X_raw, np.asarray(y, dtype=float), n_steps=self.pfr_steps)
        return self

    def transform(self, X_raw):
        X = X_raw.copy()
        X["exposure_proxy"] = X["length_m"] / X["flow_rate_L_min"]
        X["temp_diff"] = X["jacket_temperature_K"] - X["inlet_temperature_K"]
        X["temp_mean"] = 0.5 * (X["jacket_temperature_K"] + X["inlet_temperature_K"])
        X["inv_t_in"] = 1.0 / X["inlet_temperature_K"]
        X["inv_t_j"] = 1.0 / X["jacket_temperature_K"]
        X["c_tau"] = X["concentration_mol_L"] * X["exposure_proxy"]
        X["tau_over_c"] = X["exposure_proxy"] / X["concentration_mol_L"]
        X["dt_tau"] = X["temp_diff"] * X["exposure_proxy"]
        X["tin_tau"] = X["inlet_temperature_K"] * X["exposure_proxy"]
        X["tj_tau"] = X["jacket_temperature_K"] * X["exposure_proxy"]
        X["c_dt"] = X["concentration_mol_L"] * X["temp_diff"]
        X["tin_sq_centered"] = ((X["inlet_temperature_K"] - 425.0) / 50.0) ** 2
        X["tj_sq_centered"] = ((X["jacket_temperature_K"] - 445.0) / 60.0) ** 2
        for energy in self.arrhenius_grid:
            arr_name = int(energy)
            X[f"arr_tau_{arr_name}"] = np.exp(-energy / X["inlet_temperature_K"]) * X["exposure_proxy"]
            X[f"arr_mix_tau_{arr_name}"] = (
                0.5
                * (np.exp(-energy / X["inlet_temperature_K"]) + np.exp(-energy / X["jacket_temperature_K"]))
                * X["exposure_proxy"]
            )

        if self.use_physics:
            if self.physics_params_ is None:
                raise ValueError("Physics parameters are not fitted.")
            log_alpha, log_A1, E1, log_A2, E2, _, _, _ = self.physics_params_
            alpha = np.exp(log_alpha)
            A1 = np.exp(log_A1)
            A2 = np.exp(log_A2)
            tau = X["exposure_proxy"].to_numpy(dtype=float)
            t_eff_exit = X["jacket_temperature_K"].to_numpy(dtype=float) + (
                X["inlet_temperature_K"].to_numpy(dtype=float) - X["jacket_temperature_K"].to_numpy(dtype=float)
            ) * np.exp(-alpha * tau)
            t_eff_exit = np.clip(t_eff_exit, 100.0, 1000.0)
            k1_eff = A1 * np.exp(-E1 / t_eff_exit)
            k2_eff = A2 * np.exp(-E2 / t_eff_exit)
            y_phys = simulate_pfr(
                self.physics_params_,
                X["flow_rate_L_min"].to_numpy(),
                X["concentration_mol_L"].to_numpy(),
                X["inlet_temperature_K"].to_numpy(),
                X["length_m"].to_numpy(),
                X["jacket_temperature_K"].to_numpy(),
                n_steps=self.pfr_steps,
            )
            X["t_eff_exit"] = t_eff_exit
            X["k1_eff"] = k1_eff
            X["k2_eff"] = k2_eff
            X["k1_tau"] = k1_eff * tau
            X["k2_tau"] = k2_eff * tau
            X["k1_over_k2"] = k1_eff / np.clip(k2_eff, 1e-12, None)
            X["y_phys"] = y_phys
            X["phys_over_tau"] = y_phys / np.clip(X["exposure_proxy"], 1e-4, None)
            X["phys_over_c_tau"] = y_phys / np.clip(X["c_tau"], 1e-4, None)
            X["phys_minus_mid"] = y_phys - 50.0
        return X

    def fit_transform(self, X_raw, y=None):
        return self.fit(X_raw, y=y).transform(X_raw)

@dataclass
class FoldCache:
    seed: int
    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    engineered_train: pd.DataFrame
    engineered_val: pd.DataFrame
    physics_train: pd.DataFrame
    physics_val: pd.DataFrame

@dataclass
class ModelSpec:
    name: str
    feature_family: str
    estimator_factory: Callable[[], object] | None
    notes: str
    prediction_mode: str = "direct"

@dataclass
class ModelResult:
    name: str
    feature_family: str
    fold_metrics: pd.DataFrame
    seed_oof_predictions: dict[int, np.ndarray]
    averaged_oof_prediction: np.ndarray
    averaged_oof_metrics: dict[str, float]
    notes: str

@dataclass
class FittedBaseModel:
    name: str
    feature_family: str
    prediction_mode: str
    builder: PhysicsFeatureBuilder
    estimator: object

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        X = self.builder.transform(X_raw.copy())
        if self.prediction_mode == "mechanistic":
            y_pred = X["y_phys"].to_numpy(dtype=float)
        elif self.prediction_mode == "residual":
            if self.estimator is None:
                raise ValueError("Residual base model is missing its estimator.")
            y_pred = X["y_phys"].to_numpy(dtype=float) + self.estimator.predict(X)
        else:
            if self.estimator is None:
                raise ValueError("Direct base model is missing its estimator.")
            y_pred = self.estimator.predict(X)
        return clip_yield(y_pred)

@dataclass
class FinalModelBundle:
    model_type: str
    base_models: list[FittedBaseModel]
    weights: np.ndarray
    selection_summary: dict[str, object]

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        preds = np.column_stack([base_model.predict(X_raw) for base_model in self.base_models])
        combined = preds @ self.weights
        return clip_yield(combined)

def create_model_specs():
    return [
        ModelSpec(
            name="Ridge",
            feature_family="engineered",
            estimator_factory=lambda: Pipeline(
                [("scaler", StandardScaler()), ("model", Ridge(alpha=6.0))]
            ),
            notes="Linear baseline on chemistry-aware engineered features.",
        ),
        ModelSpec(
            name="Polynomial Ridge",
            feature_family="engineered",
            estimator_factory=lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("model", Ridge(alpha=20.0)),
                ]
            ),
            notes="Low-order polynomial regression without high-dimensional feature explosion.",
        ),
        ModelSpec(
            name="Random Forest",
            feature_family="engineered",
            estimator_factory=lambda: RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            notes="Tree ensemble on engineered features.",
        ),
        ModelSpec(
            name="ExtraTrees",
            feature_family="engineered",
            estimator_factory=lambda: ExtraTreesRegressor(
                n_estimators=600,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            notes="ExtraTrees on engineered features.",
        ),
        ModelSpec(
            name="HistGradientBoosting",
            feature_family="engineered",
            estimator_factory=lambda: HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.03,
                max_iter=400,
                min_samples_leaf=8,
                l2_regularization=0.5,
                random_state=42,
            ),
            notes="Non-physics HGB baseline.",
        ),
        ModelSpec(
            name="Small MLP",
            feature_family="engineered",
            estimator_factory=lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(24, 12),
                            alpha=0.08,
                            learning_rate_init=0.002,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=60,
                            max_iter=5000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            notes="Small regularized MLP without physics surrogate.",
        ),
        ModelSpec(
            name="Mechanistic PFR",
            feature_family="physics",
            estimator_factory=None,
            notes="Mechanistic non-isothermal PFR surrogate with no data-driven correction.",
            prediction_mode="mechanistic",
        ),
        ModelSpec(
            name="Physics MLP",
            feature_family="physics",
            estimator_factory=lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(50, 50),
                            alpha=0.001,
                            learning_rate_init=0.01,
                            max_iter=3000,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            notes="Legacy-style physics-informed MLP on all 150 rows.",
        ),
        ModelSpec(
            name="PFR + Residual HGB",
            feature_family="physics",
            estimator_factory=lambda: HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=120,
                min_samples_leaf=8,
                random_state=42,
            ),
            notes="Mechanistic PFR prediction plus HGB residual correction.",
            prediction_mode="residual",
        ),
        ModelSpec(
            name="Physics ExtraTrees",
            feature_family="physics",
            estimator_factory=lambda: ExtraTreesRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            notes="Physics-informed ExtraTrees with the ODE surrogate feature.",
        ),
        ModelSpec(
            name="Physics HGB",
            feature_family="physics",
            estimator_factory=lambda: HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=120,
                min_samples_leaf=8,
                random_state=42,
            ),
            notes="Physics-informed HGB without target transform.",
        ),
        ModelSpec(
            name="Physics HGB Logit",
            feature_family="physics",
            estimator_factory=lambda: TransformedTargetRegressor(
                regressor=HistGradientBoostingRegressor(
                    max_depth=3,
                    learning_rate=0.05,
                    max_iter=120,
                    min_samples_leaf=8,
                    random_state=42,
                ),
                func=yield_to_logit,
                inverse_func=logit_to_yield,
                check_inverse=False,
            ),
            notes="Physics-informed HGB with bounded target transform.",
        ),
        ModelSpec(
            name="Gaussian Process",
            feature_family="physics",
            estimator_factory=lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GaussianProcessRegressor(
                            kernel=ConstantKernel(1.0)
                            * (RBF(length_scale=1.0) + RationalQuadratic(length_scale=1.0, alpha=1.0))
                            + WhiteKernel(noise_level=0.1),
                            alpha=1e-6,
                            normalize_y=True,
                            n_restarts_optimizer=0,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            notes="Gaussian process on the physics-informed feature set.",
        ),
    ]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "max_abs_error": float(np.max(np.abs(residual))),
    }
    for metric_name, mask_fn in LOW_MID_HIGH_BINS:
        metrics[metric_name] = safe_rmse(y_true, y_pred, mask_fn(y_true))
    for metric_name, mask_fn in EXTREME_BINS:
        metrics[metric_name] = safe_rmse(y_true, y_pred, mask_fn(y_true))
    return metrics

def build_fold_cache(
    train_df: pd.DataFrame,
    seeds: list[int],
    n_splits: int,
    drop_train_indices: set[int] | None = None,
) -> list[FoldCache]:
    caches: list[FoldCache] = []
    total_folds = len(seeds) * n_splits
    fold_counter = 0
    for seed in seeds:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_id, (train_idx_full, val_idx) in enumerate(cv.split(train_df), start=1):
            fold_counter += 1
            log(f"Building fold cache {fold_counter}/{total_folds} (seed={seed}, fold={fold_id})")
            if drop_train_indices:
                train_idx = np.array([idx for idx in train_idx_full if idx not in drop_train_indices], dtype=int)
            else:
                train_idx = np.asarray(train_idx_full, dtype=int)
            val_idx = np.asarray(val_idx, dtype=int)

            train_raw = train_df.iloc[train_idx][RAW_FEATURES].copy()
            val_raw = train_df.iloc[val_idx][RAW_FEATURES].copy()
            y_train = train_df.iloc[train_idx][TARGET].to_numpy(dtype=float)
            y_val = train_df.iloc[val_idx][TARGET].to_numpy(dtype=float)

            engineered_builder = PhysicsFeatureBuilder(use_physics=False)
            engineered_train = engineered_builder.fit_transform(train_raw, y=None)
            engineered_val = engineered_builder.transform(val_raw)

            physics_builder = PhysicsFeatureBuilder(use_physics=True)
            physics_train = physics_builder.fit_transform(train_raw, y=y_train)
            physics_val = physics_builder.transform(val_raw)

            caches.append(
                FoldCache(
                    seed=seed,
                    fold_id=fold_id,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    y_train=y_train,
                    y_val=y_val,
                    engineered_train=engineered_train,
                    engineered_val=engineered_val,
                    physics_train=physics_train,
                    physics_val=physics_val,
                )
            )
    return caches

def evaluate_model(spec: ModelSpec, fold_caches: list[FoldCache], y_full: np.ndarray) -> ModelResult:
    log(f"Evaluating {spec.name}")
    n_rows = len(y_full)
    per_seed_oof = {seed: np.full(n_rows, np.nan, dtype=float) for seed in CV_SEEDS}
    fold_metric_rows: list[dict[str, float | int]] = []

    for cache in fold_caches:
        X_train = cache.engineered_train if spec.feature_family == "engineered" else cache.physics_train
        X_val = cache.engineered_val if spec.feature_family == "engineered" else cache.physics_val
        if spec.prediction_mode == "mechanistic":
            y_pred = clip_yield(X_val["y_phys"].to_numpy(dtype=float))
        else:
            if spec.estimator_factory is None:
                raise ValueError(f"{spec.name} is missing an estimator factory.")
            estimator = clone(spec.estimator_factory())
            if spec.prediction_mode == "residual":
                residual_target = cache.y_train - X_train["y_phys"].to_numpy(dtype=float)
                estimator.fit(X_train, residual_target)
                y_pred = clip_yield(X_val["y_phys"].to_numpy(dtype=float) + estimator.predict(X_val))
            else:
                estimator.fit(X_train, cache.y_train)
                y_pred = clip_yield(estimator.predict(X_val))
        per_seed_oof[cache.seed][cache.val_idx] = y_pred

        metrics = compute_metrics(cache.y_val, y_pred)
        metrics["seed"] = cache.seed
        metrics["fold_id"] = cache.fold_id
        fold_metric_rows.append(metrics)

    averaged_oof = np.nanmean(np.vstack([per_seed_oof[seed] for seed in CV_SEEDS]), axis=0)
    averaged_metrics = compute_metrics(y_full, averaged_oof)
    return ModelResult(
        name=spec.name,
        feature_family=spec.feature_family,
        fold_metrics=pd.DataFrame(fold_metric_rows),
        seed_oof_predictions=per_seed_oof,
        averaged_oof_prediction=averaged_oof,
        averaged_oof_metrics=averaged_metrics,
        notes=spec.notes,
    )

def optimize_simplex_weights(pred_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n_models = pred_matrix.shape[1]
    x0 = np.full(n_models, 1.0 / n_models, dtype=float)
    bounds = [(0.0, 1.0)] * n_models
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(
        lambda w: rmse(y_true, pred_matrix @ w),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200},
    )
    if not result.success:
        weights = x0
    else:
        weights = np.clip(result.x, 0.0, None)
        weights = weights / weights.sum()
    return weights

def evaluate_seed_heldout_ensemble(
    y_true: np.ndarray,
    candidate_results: dict[str, ModelResult],
    candidate_names: list[str],
) -> dict[str, object]:
    heldout_rows = []
    heldout_predictions = {}
    heldout_weights = {}
    for heldout_seed in CV_SEEDS:
        train_preds = []
        train_targets = []
        for seed in CV_SEEDS:
            matrix = np.column_stack([candidate_results[name].seed_oof_predictions[seed] for name in candidate_names])
            if seed == heldout_seed:
                heldout_matrix = matrix
            else:
                train_preds.append(matrix)
                train_targets.append(y_true)
        inner_X = np.vstack(train_preds)
        inner_y = np.concatenate(train_targets)
        weights = optimize_simplex_weights(inner_X, inner_y)
        heldout_pred = clip_yield(heldout_matrix @ weights)
        heldout_predictions[heldout_seed] = heldout_pred
        heldout_weights[heldout_seed] = weights
        metrics = compute_metrics(y_true, heldout_pred)
        metrics["seed"] = heldout_seed
        heldout_rows.append(metrics)

    optimized_df = pd.DataFrame(heldout_rows)
    simple_rows = []
    best_single_rows = []
    top_single_name = min(
        candidate_names,
        key=lambda name: np.mean(
            [
                rmse(y_true, candidate_results[name].seed_oof_predictions[seed])
                for seed in CV_SEEDS
            ]
        ),
    )
    for seed in CV_SEEDS:
        matrix = np.column_stack([candidate_results[name].seed_oof_predictions[seed] for name in candidate_names])
        simple_pred = clip_yield(matrix.mean(axis=1))
        single_pred = clip_yield(candidate_results[top_single_name].seed_oof_predictions[seed])
        simple_metrics = compute_metrics(y_true, simple_pred)
        simple_metrics["seed"] = seed
        simple_rows.append(simple_metrics)
        single_metrics = compute_metrics(y_true, single_pred)
        single_metrics["seed"] = seed
        best_single_rows.append(single_metrics)

    return {
        "candidate_names": candidate_names,
        "best_single_name": top_single_name,
        "optimized_df": optimized_df,
        "simple_df": pd.DataFrame(simple_rows),
        "single_df": pd.DataFrame(best_single_rows),
        "heldout_predictions": heldout_predictions,
        "heldout_weights": heldout_weights,
        "mean_weights": np.mean(np.vstack([heldout_weights[seed] for seed in CV_SEEDS]), axis=0),
    }

def fit_base_model(spec: ModelSpec, X_raw: pd.DataFrame, y: np.ndarray) -> FittedBaseModel:
    builder = PhysicsFeatureBuilder(use_physics=(spec.feature_family == "physics"))
    X = builder.fit_transform(X_raw.copy(), y=y if spec.feature_family == "physics" else None)
    estimator = None
    if spec.prediction_mode != "mechanistic":
        if spec.estimator_factory is None:
            raise ValueError(f"{spec.name} is missing an estimator factory.")
        estimator = clone(spec.estimator_factory())
        if spec.prediction_mode == "residual":
            estimator.fit(X, y - X["y_phys"].to_numpy(dtype=float))
        else:
            estimator.fit(X, y)
    return FittedBaseModel(
        name=spec.name,
        feature_family=spec.feature_family,
        prediction_mode=spec.prediction_mode,
        builder=builder,
        estimator=estimator,
    )

def compute_permutation_importance(
    final_model: FinalModelBundle,
    X_raw: pd.DataFrame,
    y_true: np.ndarray,
    n_repeats: int = 20,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    transformed = [base_model.builder.transform(X_raw.copy()) for base_model in final_model.base_models]
    feature_names = list(transformed[0].columns)
    baseline_pred = final_model.predict(X_raw)
    baseline_rmse = rmse(y_true, baseline_pred)
    rows = []
    for feature_name in feature_names:
        deltas = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X_raw))
            per_model_preds = []
            for model_idx, base_model in enumerate(final_model.base_models):
                X_perm = transformed[model_idx].copy()
                X_perm[feature_name] = X_perm[feature_name].to_numpy()[perm]
                if base_model.prediction_mode == "mechanistic":
                    per_model_preds.append(clip_yield(X_perm["y_phys"].to_numpy(dtype=float)))
                elif base_model.prediction_mode == "residual":
                    if base_model.estimator is None:
                        raise ValueError("Residual ensemble member is missing its estimator.")
                    per_model_preds.append(
                        clip_yield(X_perm["y_phys"].to_numpy(dtype=float) + base_model.estimator.predict(X_perm))
                    )
                else:
                    if base_model.estimator is None:
                        raise ValueError("Direct ensemble member is missing its estimator.")
                    per_model_preds.append(clip_yield(base_model.estimator.predict(X_perm)))
            stacked = np.column_stack(per_model_preds)
            perm_pred = clip_yield(stacked @ final_model.weights)
            deltas.append(rmse(y_true, perm_pred) - baseline_rmse)
        rows.append(
            {
                "feature": feature_name,
                "rmse_increase_mean": float(np.mean(deltas)),
                "rmse_increase_std": float(np.std(deltas, ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_increase_mean", ascending=False).reset_index(drop=True)

def compute_influence_report(
    X_raw: pd.DataFrame,
    y_true: np.ndarray,
    oof_pred: np.ndarray,
) -> pd.DataFrame:
    builder = PhysicsFeatureBuilder(use_physics=True)
    X_phys = builder.fit_transform(X_raw.copy(), y=y_true)
    leverage_cols = [
        "flow_rate_L_min",
        "concentration_mol_L",
        "inlet_temperature_K",
        "length_m",
        "jacket_temperature_K",
        "exposure_proxy",
        "temp_diff",
        "c_tau",
        "dt_tau",
        "y_phys",
        "phys_over_tau",
        "phys_over_c_tau",
    ]
    X_mat = X_phys[leverage_cols].to_numpy(dtype=float)
    X_scaled = (X_mat - X_mat.mean(axis=0, keepdims=True)) / np.clip(X_mat.std(axis=0, ddof=1, keepdims=True), 1e-6, None)
    X_design = np.column_stack([np.ones(len(X_scaled)), X_scaled])
    xtx_inv = np.linalg.pinv(X_design.T @ X_design)
    leverage = np.einsum("ij,jk,ik->i", X_design, xtx_inv, X_design)
    residual = y_true - oof_pred
    p = X_design.shape[1]
    mse = np.sum(residual ** 2) / max(len(residual) - p, 1)
    sigma = math.sqrt(max(mse, 1e-8))
    standardized_residual = residual / np.clip(sigma * np.sqrt(1.0 - leverage), 1e-8, None)
    cooks_distance = (standardized_residual ** 2 * leverage) / np.clip(p * (1.0 - leverage), 1e-8, None)
    influence_df = pd.DataFrame(
        {
            "row_index": np.arange(len(X_raw)),
            "overall_yield": y_true,
            "oof_prediction": oof_pred,
            "residual": residual,
            "abs_residual": np.abs(residual),
            "standardized_residual": standardized_residual,
            "leverage": leverage,
            "cooks_distance": cooks_distance,
            "was_removed_in_trainer_new": [idx in REMOVED_INDICES for idx in range(len(X_raw))],
        }
    )
    return influence_df.sort_values(["cooks_distance", "abs_residual"], ascending=False).reset_index(drop=True)

def compare_outlier_removal(best_spec: ModelSpec, train_df: pd.DataFrame) -> dict[str, object]:
    log("Rebuilding fold cache for outlier-removal comparison")
    drop_cache = build_fold_cache(train_df, CV_SEEDS, N_SPLITS, drop_train_indices=set(REMOVED_INDICES))
    drop_result = evaluate_model(best_spec, drop_cache, train_df[TARGET].to_numpy(dtype=float))
    return {
        "drop_cache_result": drop_result,
    }

def plot_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, train_df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    exposure = train_df["length_m"] / train_df["flow_rate_L_min"]
    residual = y_true - y_pred

    figures = [
        ("predicted_vs_actual.png", y_true, y_pred, "Actual yield", "Predicted yield"),
        ("residual_vs_actual.png", y_true, residual, "Actual yield", "Residual (actual - predicted)"),
        ("residual_vs_temperature.png", train_df["inlet_temperature_K"], residual, "Inlet temperature (K)", "Residual"),
        ("residual_vs_exposure.png", exposure, residual, "Exposure proxy L/F", "Residual"),
        ("residual_vs_concentration.png", train_df["concentration_mol_L"], residual, "Concentration (mol/L)", "Residual"),
    ]
    for filename, x_values, y_values, xlabel, ylabel in figures:
        plt.figure(figsize=(7, 5))
        plt.scatter(x_values, y_values, alpha=0.8)
        plt.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / filename, dpi=160)
        plt.close()


def summarize_results(results):
    rows = []
    for name, res in results.items():
        row = {
            "model": res.name,
            "feature_family": res.feature_family,
            "cv_rmse_mean": res.fold_metrics["rmse"].mean(),
            "cv_rmse_std": res.fold_metrics["rmse"].std(ddof=0),
            "worst_fold_rmse": res.fold_metrics["rmse"].max(),
            "avg_oof_rmse": res.averaged_oof_metrics["rmse"],
            "avg_oof_mae": res.averaged_oof_metrics["mae"],
            "avg_oof_r2": res.averaged_oof_metrics["r2"],
            "avg_oof_max_abs_error": res.averaged_oof_metrics["max_abs_error"],
            "avg_oof_low_yield_rmse": res.averaged_oof_metrics["low_yield_rmse"],
            "avg_oof_mid_yield_rmse": res.averaged_oof_metrics["mid_yield_rmse"],
            "avg_oof_high_yield_rmse": res.averaged_oof_metrics["high_yield_rmse"],
            "avg_oof_yield_lt_10_rmse": res.averaged_oof_metrics["yield_lt_10_rmse"],
            "avg_oof_yield_10_30_rmse": res.averaged_oof_metrics["yield_10_30_rmse"],
            "avg_oof_yield_30_70_rmse": res.averaged_oof_metrics["yield_30_70_rmse"],
            "avg_oof_yield_70_90_rmse": res.averaged_oof_metrics["yield_70_90_rmse"],
            "avg_oof_yield_gt_90_rmse": res.averaged_oof_metrics["yield_gt_90_rmse"],
            "notes": res.notes
        }
        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    return df.sort_values("cv_rmse_mean").reset_index(drop=True)

def save_oof_predictions(
    y_true: np.ndarray,
    results: dict[str, ModelResult],
    final_oof_prediction: np.ndarray,
) -> None:
    df = pd.DataFrame({"row_index": np.arange(len(y_true)), "actual_overall_yield": y_true})
    for name, result in results.items():
        safe_name = name.lower().replace(" ", "_")
        df[f"{safe_name}_oof"] = result.averaged_oof_prediction
    df["final_model_oof"] = final_oof_prediction
    df.to_csv(ROOT / "oof_predictions.csv", index=False)

def save_predictions_file(predictions: np.ndarray, path: Path) -> None:
    pd.DataFrame({TARGET: predictions}).to_csv(path, index=False)

def print_leaderboard(comparison_df: pd.DataFrame, final_model_name: str, final_rmse: float) -> None:
    log("")
    log("MODEL                         CV RMSE     STD       HIGH-Y RMSE")
    log("----------------------------------------------------------------")
    for _, row in comparison_df.iterrows():
        model_name = str(row["model"])[:28].ljust(28)
        log(
            f"{model_name} {format_float(float(row['cv_rmse_mean']), 4).rjust(9)} "
            f"{format_float(float(row['cv_rmse_std']), 4).rjust(9)} "
            f"{format_float(float(row['avg_oof_high_yield_rmse']), 4).rjust(13)}"
        )
    log("")
    log(f"BEST MODEL: {final_model_name}")
    log(f"BEST ROBUST CV RMSE: {format_float(final_rmse)}")
    log("FINAL SUBMISSION FILE: final_submission.csv")

if __name__ == "__main__":
    log("Loading datasets")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    trainer_new_df = pd.read_csv(TRAINER_NEW_PATH)
    y_train = train_df[TARGET].to_numpy(dtype=float)
        
    log("Auditing data and reproducing legacy baselines")
            
    log("Building leakage-free fold caches")
    full_fold_cache = build_fold_cache(train_df, CV_SEEDS, N_SPLITS)

    log("Benchmarking models")
    model_specs = create_model_specs()
    model_results = {spec.name: evaluate_model(spec, full_fold_cache, y_train) for spec in model_specs}
    comparison_df = summarize_results(model_results)
    comparison_df.to_csv(ROOT / "model_comparison.csv", index=False)

    log("Evaluating seed-held-out constrained ensemble")
    candidate_pool = {
        "Physics HGB Logit",
        "PFR + Residual HGB",
        "Physics ExtraTrees",
        "Gaussian Process",
        "Physics HGB",
        "Physics MLP",
    }
    ensemble_candidates = [
        name for name in comparison_df["model"].tolist() if name in candidate_pool
    ][:3]
    ensemble_summary = evaluate_seed_heldout_ensemble(
        y_true=y_train,
        candidate_results={name: model_results[name] for name in ensemble_candidates},
        candidate_names=ensemble_candidates,
    )

    log("Selecting the final model")
    best_single_name = str(comparison_df.iloc[0]["model"])
    best_single_spec = next(spec for spec in model_specs if spec.name == best_single_name)
    best_single_result = model_results[best_single_name]
    best_single_seed_rmse = float(np.mean([rmse(y_train, best_single_result.seed_oof_predictions[seed]) for seed in CV_SEEDS]))
    ensemble_seed_rmse = float(ensemble_summary["optimized_df"]["rmse"].mean())
    ensemble_seed_std = float(ensemble_summary["optimized_df"]["rmse"].std(ddof=0))
    single_seed_std = float(ensemble_summary["single_df"]["rmse"].std(ddof=0))

    use_ensemble = bool(
        (ensemble_seed_rmse + 0.05 < best_single_seed_rmse)
        and (ensemble_seed_std <= single_seed_std + 0.10)
    )

    if use_ensemble:
        final_model_name = "Best Ensemble"
        final_selection_rmse = ensemble_seed_rmse
        final_oof_prediction = np.mean(
            np.row_stack([ensemble_summary["heldout_predictions"][seed] for seed in CV_SEEDS]),
            axis=0,
        )
    else:
        final_model_name = best_single_name
        final_selection_rmse = float(comparison_df.iloc[0]["cv_rmse_mean"])
        final_oof_prediction = best_single_result.averaged_oof_prediction

    log("Running outlier/influence analysis")
    drop_summary = compare_outlier_removal(best_single_spec, train_df)
    outlier_keep_rmse = float(model_results[best_single_name].fold_metrics["rmse"].mean())
    outlier_drop_rmse = float(drop_summary["drop_cache_result"].fold_metrics["rmse"].mean())
    influence_df = compute_influence_report(train_df[RAW_FEATURES], y_train, best_single_result.averaged_oof_prediction)

    log("Fitting the final full-data model")
    if use_ensemble:
        final_weights = np.asarray(ensemble_summary["mean_weights"], dtype=float)
        final_weights = final_weights / final_weights.sum()
        final_base_models = [
            fit_base_model(next(spec for spec in model_specs if spec.name == name), train_df[RAW_FEATURES], y_train)
            for name in ensemble_candidates
        ]
    else:
        final_weights = np.array([1.0], dtype=float)
        final_base_models = [fit_base_model(best_single_spec, train_df[RAW_FEATURES], y_train)]

    final_model = FinalModelBundle(
        model_type="ensemble" if use_ensemble else "single",
        base_models=final_base_models,
        weights=final_weights,
        selection_summary={
            "use_ensemble": use_ensemble,
            "best_single_name": best_single_name,
            "best_single_seed_rmse": best_single_seed_rmse,
            "ensemble_seed_rmse": ensemble_seed_rmse,
            "ensemble_seed_std": ensemble_seed_std,
            "single_seed_std": single_seed_std,
            "ensemble_candidates": ensemble_candidates,
        },
    )
    test_predictions = final_model.predict(test_df[RAW_FEATURES]).round(6)
    save_predictions_file(test_predictions, ROOT / "final_predictions.csv")
    save_predictions_file(test_predictions, ROOT / "final_submission.csv")
    joblib.dump(final_model, ROOT / "final_model.pkl")

    log("Saving diagnostics and reports")
    feature_importance_df = compute_permutation_importance(final_model, train_df[RAW_FEATURES], y_train)
    feature_importance_df.to_csv(ROOT / "feature_importance.csv", index=False)
    save_oof_predictions(y_train, model_results, final_oof_prediction)
    plot_diagnostics(y_train, final_oof_prediction, train_df)
            
    print_leaderboard(comparison_df, final_model_name, final_selection_rmse)