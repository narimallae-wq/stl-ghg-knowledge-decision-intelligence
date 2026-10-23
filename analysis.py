"""Reproducible St. Louis regional emissions forecasting and decision analysis.

The pipeline downloads public NREL SLOPE scenario data for the 15-county
St. Louis MO-IL MSA, evaluates forecasting baselines against a proposed
knowledge-guided XGBoost model, and produces exactly two paper figures and
two paper tables.

Important scope note: SLOPE reports energy-related CO2 projections rather
than a complete community greenhouse-gas inventory. The 2022 regional GGRP
benchmark (44.5 MMTCO2e) is therefore used as an external scope check, not as
an interchangeable target.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "stl_slope_scenarios.csv"
OUTPUT_DIR = ROOT / "outputs"
SLOPE_API = "https://maps-api.nlr.gov/slope/v2/scenario/data"

RANDOM_SEED = 42
TRAIN_END = 2037
VALID_END = 2042
TEST_START = 2043
GGRP_2022_MMTCO2E = 44.5


# The geographic scope follows the St. Louis MO-IL MSA in the regional PCAP.
COUNTIES = {
    "17005": ("Bond", "IL"),
    "17013": ("Calhoun", "IL"),
    "17027": ("Clinton", "IL"),
    "17083": ("Jersey", "IL"),
    "17117": ("Macoupin", "IL"),
    "17119": ("Madison", "IL"),
    "17133": ("Monroe", "IL"),
    "17163": ("St. Clair", "IL"),
    "29071": ("Franklin", "MO"),
    "29099": ("Jefferson", "MO"),
    "29113": ("Lincoln", "MO"),
    "29183": ("St. Charles", "MO"),
    "29189": ("St. Louis County", "MO"),
    "29219": ("Warren", "MO"),
    "29510": ("St. Louis City", "MO"),
}


SCENARIOS = {
    "reference": {
        "label": "Reference",
        "elec_supply": "Reference",
        "electrification": "Reference",
        "efficiency": "Reference",
        "flexibility": "Reference",
        "grid_level": 0,
        "electrification_level": 0,
        "efficiency_level": 0,
    },
    "grid_2035": {
        "label": "Grid modernization by 2035",
        "elec_supply": "100% by 2035",
        "electrification": "Reference",
        "efficiency": "Reference",
        "flexibility": "Reference",
        "grid_level": 2,
        "electrification_level": 0,
        "efficiency_level": 0,
    },
    "grid_2050": {
        "label": "Grid modernization by 2050",
        "elec_supply": "95% by 2050",
        "electrification": "Reference",
        "efficiency": "Reference",
        "flexibility": "Reference",
        "grid_level": 1,
        "electrification_level": 0,
        "efficiency_level": 0,
    },
    "electrification": {
        "label": "Widespread electrification",
        "elec_supply": "Reference",
        "electrification": "All-High",
        "efficiency": "Reference",
        "flexibility": "Reference",
        "grid_level": 0,
        "electrification_level": 1,
        "efficiency_level": 0,
    },
    "grid_2035_electrification": {
        "label": "2035 grid + electrification",
        "elec_supply": "100% by 2035",
        "electrification": "All-High",
        "efficiency": "Reference",
        "flexibility": "Reference",
        "grid_level": 2,
        "electrification_level": 1,
        "efficiency_level": 0,
    },
    "grid_2050_electrification": {
        "label": "2050 grid + electrification",
        "elec_supply": "95% by 2050",
        "electrification": "All-High",
        "efficiency": "Reference",
        "flexibility": "Reference",
        "grid_level": 1,
        "electrification_level": 1,
        "efficiency_level": 0,
    },
    "efficiency": {
        "label": "Building efficiency",
        "elec_supply": "Reference",
        "electrification": "Reference",
        "efficiency": "Enhanced",
        "flexibility": "Reference",
        "grid_level": 0,
        "electrification_level": 0,
        "efficiency_level": 1,
    },
    "grid_2035_efficiency": {
        "label": "2035 grid + building efficiency",
        "elec_supply": "100% by 2035",
        "electrification": "Reference",
        "efficiency": "Enhanced",
        "flexibility": "Reference",
        "grid_level": 2,
        "electrification_level": 0,
        "efficiency_level": 1,
    },
    "grid_2050_efficiency": {
        "label": "2050 grid + building efficiency",
        "elec_supply": "95% by 2050",
        "electrification": "Reference",
        "efficiency": "Enhanced",
        "flexibility": "Reference",
        "grid_level": 1,
        "electrification_level": 0,
        "efficiency_level": 1,
    },
}


COMPONENT_TO_SECTOR = {
    "TRANSPORTATION_ELECTRIC": "Transportation",
    "TRANSPORTATION_NON-ELECTRIC": "Transportation",
    "INDUSTRIAL_ELECTRIC": "Industrial",
    "INDUSTRIAL_NON-ELECTRIC": "Industrial",
    "RESIDENTIAL_ELECTRIC": "Buildings",
    "RESIDENTIAL_NON-ELECTRIC": "Buildings",
    "COMMERCIAL_ELECTRIC": "Buildings",
    "COMMERCIAL_NON-ELECTRIC": "Buildings",
}


def slope_geography_id(fips: str) -> str:
    """Convert a five-digit county FIPS to SLOPE's county identifier."""
    return f"G{fips[:2]}0{fips[2:]}0"


def request_json(session: requests.Session, payload: dict, retries: int = 4) -> dict:
    """POST to SLOPE with bounded exponential backoff and validation."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.post(SLOPE_API, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            settings = data.get("outputs", {}).get("settings", {})
            if not settings.get("labels") or not settings.get("datasets"):
                raise ValueError(f"SLOPE returned no scenario series for {payload['location']}")
            return data
        except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(2**attempt)
    raise RuntimeError(f"SLOPE request failed after {retries} attempts") from last_error


def download_slope_data(refresh: bool = False) -> pd.DataFrame:
    """Download and cache the compact county-scenario-component panel."""
    if DATA_FILE.exists() and not refresh:
        return pd.read_csv(DATA_FILE, dtype={"county_fips": str})

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "stl-ghg-kbs-research/1.0"})

    for fips, (county, state) in COUNTIES.items():
        for scenario_id, spec in SCENARIOS.items():
            payload = {
                "location": slope_geography_id(fips),
                "res": "county",
                "metric": "co2_emissions",
                "elec_supply": spec["elec_supply"],
                "electrification": spec["electrification"],
                "efficiency": spec["efficiency"],
                "flexibility": spec["flexibility"],
            }
            response = request_json(session, payload)
            settings = response["outputs"]["settings"]
            years = settings["labels"]
            for series in settings["datasets"]:
                component = series["id"]
                values = series["data"]
                if len(values) != len(years):
                    raise ValueError(f"Length mismatch for {fips}/{scenario_id}/{component}")
                for year, value in zip(years, values, strict=True):
                    rows.append(
                        {
                            "year": int(year),
                            "county_fips": fips,
                            "county": county,
                            "state": state,
                            "scenario": scenario_id,
                            "scenario_label": spec["label"],
                            "grid_level": spec["grid_level"],
                            "electrification_level": spec["electrification_level"],
                            "efficiency_level": spec["efficiency_level"],
                            "component": component,
                            "sector": COMPONENT_TO_SECTOR[component],
                            "is_electric": int(component.endswith("_ELECTRIC")),
                            "emissions_mmtco2": float(value),
                        }
                    )

    panel = pd.DataFrame(rows).sort_values(
        ["county_fips", "scenario", "component", "year"]
    )
    expected = len(COUNTIES) * len(SCENARIOS) * 8 * 29
    if len(panel) != expected:
        raise ValueError(f"Expected {expected:,} rows; received {len(panel):,}")
    panel.to_csv(DATA_FILE, index=False)
    return panel


def feature_frame(df: pd.DataFrame, knowledge_guided: bool) -> pd.DataFrame:
    """Create leakage-free scenario features available before the target year."""
    x = df.copy()
    x["year_index"] = x["year"] - 2022
    x["year_index_sq"] = x["year_index"] ** 2
    x["is_building"] = (x["sector"] == "Buildings").astype(int)
    x["is_electrifiable"] = x["sector"].isin(
        ["Buildings", "Transportation"]
    ).astype(int)

    if knowledge_guided:
        # Rules encode which interventions can directly affect each component.
        x["effective_grid"] = x["grid_level"] * x["is_electric"]
        x["effective_efficiency"] = x["efficiency_level"] * x["is_building"]
        x["effective_electrification"] = (
            x["electrification_level"] * x["is_electrifiable"]
        )
        x["grid_time"] = x["effective_grid"] * x["year_index"]
        x["efficiency_time"] = x["effective_efficiency"] * x["year_index"]
        x["electrification_time"] = x["effective_electrification"] * x["year_index"]
    else:
        x["grid_time"] = x["grid_level"] * x["year_index"]
        x["efficiency_time"] = x["efficiency_level"] * x["year_index"]
        x["electrification_time"] = x["electrification_level"] * x["year_index"]
    return x


def build_xgb_pipeline(
    knowledge_guided: bool,
    *,
    max_depth: int = 6,
    n_estimators: int = 650,
    learning_rate: float = 0.035,
) -> Pipeline:
    categorical = ["county_fips", "component"]
    numeric = [
        "year_index",
        "year_index_sq",
        "grid_level",
        "electrification_level",
        "efficiency_level",
        "grid_time",
        "electrification_time",
        "efficiency_time",
    ]
    if knowledge_guided:
        numeric += ["effective_grid", "effective_electrification", "effective_efficiency"]

    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=3,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_alpha=0.03,
        reg_lambda=1.2,
        random_state=RANDOM_SEED,
        n_jobs=4,
    )
    return Pipeline([("features", preprocessor), ("model", model)])


def fit_xgb(
    train: pd.DataFrame,
    knowledge_guided: bool,
    *,
    max_depth: int = 6,
) -> Pipeline:
    x = feature_frame(train, knowledge_guided)
    target = np.log1p(train["emissions_mmtco2"].to_numpy())
    pipeline = build_xgb_pipeline(knowledge_guided, max_depth=max_depth)
    pipeline.fit(x, target)
    return pipeline


def predict_xgb(model: Pipeline, frame: pd.DataFrame, knowledge_guided: bool) -> np.ndarray:
    log_pred = model.predict(feature_frame(frame, knowledge_guided))
    return np.clip(np.expm1(log_pred), 0.0, None)


GROUP_KEYS = ["county_fips", "scenario", "component"]


def persistence_predictions(history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    last = (
        history.sort_values("year")
        .groupby(GROUP_KEYS, observed=True)["emissions_mmtco2"]
        .last()
    )
    keys = pd.MultiIndex.from_frame(future[GROUP_KEYS])
    return last.reindex(keys).to_numpy()


def linear_trend_predictions(
    history: pd.DataFrame, future: pd.DataFrame, window: int = 8
) -> np.ndarray:
    predictions = pd.Series(index=future.index, dtype=float)
    grouped_future = future.groupby(GROUP_KEYS, sort=False, observed=True)
    history_groups = {key: group for key, group in history.groupby(GROUP_KEYS, observed=True)}
    for key, target_group in grouped_future:
        source = history_groups[key].sort_values("year").tail(window)
        x = source["year"].to_numpy(dtype=float)
        y = source["emissions_mmtco2"].to_numpy(dtype=float)
        if len(np.unique(x)) < 2:
            pred = np.repeat(y[-1], len(target_group))
        else:
            slope, intercept = np.polyfit(x, y, deg=1)
            pred = slope * target_group["year"].to_numpy(dtype=float) + intercept
        predictions.loc[target_group.index] = np.clip(pred, 0.0, None)
    return predictions.loc[future.index].to_numpy()


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error, stable for near-zero components."""
    return float(100 * np.abs(y_pred - y_true).sum() / np.abs(y_true).sum())


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "WAPE_pct": wape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def county_bootstrap_rmse(
    test: pd.DataFrame, prediction: np.ndarray, iterations: int = 1000
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    counties = np.array(sorted(test["county_fips"].unique()))
    error_sq = (test["emissions_mmtco2"].to_numpy() - prediction) ** 2
    county_error = {
        county: error_sq[test["county_fips"].to_numpy() == county] for county in counties
    }
    values = []
    for _ in range(iterations):
        sampled = rng.choice(counties, size=len(counties), replace=True)
        errors = np.concatenate([county_error[county] for county in sampled])
        values.append(float(np.sqrt(errors.mean())))
    return tuple(np.percentile(values, [2.5, 97.5]))


def grouped_absolute_error(test: pd.DataFrame, prediction: np.ndarray) -> pd.Series:
    tmp = test[GROUP_KEYS].copy()
    tmp["absolute_error"] = np.abs(test["emissions_mmtco2"].to_numpy() - prediction)
    return tmp.groupby(GROUP_KEYS, observed=True)["absolute_error"].mean().sort_index()


def evaluate_models(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    train = panel[panel["year"] <= TRAIN_END].copy()
    valid = panel[(panel["year"] > TRAIN_END) & (panel["year"] <= VALID_END)].copy()
    development = panel[panel["year"] <= VALID_END].copy()
    test = panel[panel["year"] >= TEST_START].copy()

    # Select model capacity only on the validation interval; the final test
    # years remain untouched until the model specification has been fixed.
    y_valid = valid["emissions_mmtco2"].to_numpy()
    depth_candidates = (4, 6, 8)
    selected_depth = min(
        depth_candidates,
        key=lambda depth: mean_absolute_error(
            y_valid,
            predict_xgb(
                fit_xgb(train, knowledge_guided=True, max_depth=depth),
                valid,
                knowledge_guided=True,
            ),
        ),
    )

    raw_xgb = fit_xgb(development, knowledge_guided=False)
    kg_xgb = fit_xgb(
        development, knowledge_guided=True, max_depth=selected_depth
    )

    predictions = {
        "Persistence": persistence_predictions(development, test),
        "Group linear trend": linear_trend_predictions(development, test),
        "XGBoost": predict_xgb(raw_xgb, test, knowledge_guided=False),
    }
    predictions["Proposed KG-XGB"] = predict_xgb(
        kg_xgb, test, knowledge_guided=True
    )

    y_test = test["emissions_mmtco2"].to_numpy()
    proposed_group_error = grouped_absolute_error(test, predictions["Proposed KG-XGB"])
    rows = []
    for name, prediction in predictions.items():
        metrics = metric_row(y_test, prediction)
        lower, upper = county_bootstrap_rmse(test, prediction)
        if name == "Proposed KG-XGB":
            p_value = np.nan
        else:
            comparison = grouped_absolute_error(test, prediction)
            aligned = comparison.index.intersection(proposed_group_error.index)
            try:
                p_value = wilcoxon(
                    comparison.loc[aligned],
                    proposed_group_error.loc[aligned],
                    alternative="two-sided",
                ).pvalue
            except ValueError:
                p_value = np.nan
        rows.append(
            {
                "Model": name,
                **metrics,
                "RMSE_CI_low": lower,
                "RMSE_CI_high": upper,
                "paired_Wilcoxon_p": p_value,
            }
        )

    performance = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    performance.insert(0, "Rank", np.arange(1, len(performance) + 1))
    performance.to_csv(OUTPUT_DIR / "table_1_model_performance.csv", index=False)

    test_predictions = test[
        ["year", "county_fips", "county", "scenario", "component", "sector", "emissions_mmtco2"]
    ].copy()
    for name, prediction in predictions.items():
        test_predictions[name] = prediction
    return performance, test_predictions, int(selected_depth)


def decision_ranking(panel: pd.DataFrame) -> pd.DataFrame:
    regional = (
        panel.groupby(["scenario", "scenario_label", "year"], observed=True)[
            "emissions_mmtco2"
        ]
        .sum()
        .reset_index()
    )
    pivot = regional.pivot(index="year", columns="scenario", values="emissions_mmtco2")
    reference = pivot["reference"]
    rows = []
    for scenario, spec in SCENARIOS.items():
        if scenario == "reference":
            continue
        avoided = reference - pivot[scenario]
        cumulative = avoided.loc[2026:2050].sum()
        pct_2035 = 100 * avoided.loc[2035] / reference.loc[2035]
        pct_2050 = 100 * avoided.loc[2050] / reference.loc[2050]
        threshold = avoided / reference >= 0.20
        threshold_years = threshold[threshold].index.tolist()
        rows.append(
            {
                "Portfolio": spec["label"],
                "Evidence": "NREL SLOPE scenario",
                "Cumulative_avoided_2026_2050_MMTCO2": cumulative,
                "Reduction_2035_pct": pct_2035,
                "Reduction_2050_pct": pct_2050,
                "First_year_at_least_20pct": threshold_years[0] if threshold_years else "Not reached",
                "Decision_status": (
                    "Not recommended alone"
                    if cumulative <= 0 or pct_2050 <= 0
                    else "Delayed benefit"
                    if pct_2035 < 0
                    else "Admissible"
                ),
            }
        )

    ranking = pd.DataFrame(rows).sort_values(
        "Cumulative_avoided_2026_2050_MMTCO2", ascending=False
    )
    ranking.insert(0, "Rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(OUTPUT_DIR / "table_2_decision_ranking.csv", index=False)
    return ranking


def make_figure_1(panel: pd.DataFrame, predictions: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    colors = {
        "reference": "#4D4D4D",
        "grid_2050_electrification": "#6A3D9A",
        "grid_2035_electrification": "#1B9E77",
        "grid_2035_efficiency": "#D95F02",
    }
    labels = {
        "reference": "Reference",
        "grid_2050_electrification": "2050 grid + electrification",
        "grid_2035_electrification": "2035 grid + electrification",
        "grid_2035_efficiency": "2035 grid + efficiency",
    }

    regional = (
        panel[panel["scenario"].isin(colors)]
        .groupby(["scenario", "year"], observed=True)["emissions_mmtco2"]
        .sum()
        .reset_index()
    )
    actual_test = (
        predictions.groupby(["scenario", "year"], observed=True)["emissions_mmtco2"]
        .sum()
        .reset_index()
    )
    predicted_test = (
        predictions.groupby(["scenario", "year"], observed=True)["Proposed KG-XGB"]
        .sum()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    for scenario, color in colors.items():
        series = regional[regional["scenario"] == scenario]
        axes[0].plot(
            series["year"], series["emissions_mmtco2"], color=color, lw=2.2,
            label=labels[scenario],
        )
    axes[0].axvspan(TEST_START, 2050, color="#D9D9D9", alpha=0.45, label="Held-out years")
    axes[0].set_title("(a) Regional scenario trajectories")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Energy-related CO$_2$ (MMTCO$_2$)")
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    merged = actual_test.merge(
        predicted_test, on=["scenario", "year"], suffixes=("_nrel", "_kgxgb")
    )
    for scenario, color in colors.items():
        points = merged[merged["scenario"] == scenario]
        axes[1].scatter(
            points["emissions_mmtco2"], points["Proposed KG-XGB"],
            s=34, alpha=0.82, color=color, label=labels[scenario], edgecolor="white", linewidth=0.4,
        )
    low = min(merged["emissions_mmtco2"].min(), merged["Proposed KG-XGB"].min())
    high = max(merged["emissions_mmtco2"].max(), merged["Proposed KG-XGB"].max())
    axes[1].plot([low, high], [low, high], ls="--", color="#555555", lw=1.2)
    axes[1].set_title("(b) Held-out regional predictions, 2043-2050")
    axes[1].set_xlabel("NREL SLOPE projection (MMTCO$_2$)")
    axes[1].set_ylabel("KG-XGB prediction (MMTCO$_2$)")

    fig.suptitle("St. Louis regional scenario forecasting and temporal validation", fontsize=13, y=1.03)
    fig.savefig(OUTPUT_DIR / "figure_1_forecast_validation.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def make_figure_2(panel: pd.DataFrame) -> None:
    """Show the two leading integrated portfolios as projection trajectories."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    scenarios = {
        "reference": ("Reference", "#4D4D4D"),
        "grid_2035_electrification": ("2035 grid + electrification", "#1B9E77"),
        "grid_2035_efficiency": ("2035 grid + building efficiency", "#6A3D9A"),
    }
    regional = (
        panel[panel["scenario"].isin(scenarios)]
        .groupby(["scenario", "year"], observed=True)["emissions_mmtco2"]
        .sum()
        .reset_index()
    )
    pivot = regional.pivot(index="year", columns="scenario", values="emissions_mmtco2")
    decision_years = pivot.loc[2026:2050].index

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    for scenario, (label, color) in scenarios.items():
        series = pivot[scenario]
        axes[0].plot(series.index, series, color=color, lw=2.4, label=label)
    axes[0].axvline(2035, color="#777777", lw=1.0, ls="--")
    axes[0].set_title("(a) Regional emissions pathways")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Energy-related CO$_2$ (MMTCO$_2$)")
    axes[0].legend(frameon=False, fontsize=8, loc="best")

    reference = pivot.loc[decision_years, "reference"]
    for scenario in ["grid_2035_electrification", "grid_2035_efficiency"]:
        label, color = scenarios[scenario]
        cumulative = (reference - pivot.loc[decision_years, scenario]).cumsum()
        axes[1].plot(
            cumulative.index,
            cumulative,
            color=color,
            lw=2.4,
            marker="o",
            markevery=[9, -1],
            ms=4.5,
            label=label,
        )
        axes[1].annotate(
            f"{cumulative.iloc[-1]:.1f}",
            xy=(cumulative.index[-1], cumulative.iloc[-1]),
            xytext=(-4, 7),
            textcoords="offset points",
            ha="right",
            color=color,
            fontsize=8,
        )
    axes[1].axhline(0, color="#777777", lw=0.9)
    axes[1].set_title("(b) Cumulative avoided emissions, 2026-2050")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Cumulative avoided CO$_2$ (MMTCO$_2$)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(
        "St. Louis clean-energy portfolio projections and cumulative benefits",
        fontsize=13,
        y=1.03,
    )
    fig.savefig(OUTPUT_DIR / "figure_2_decision_ranking.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def main(refresh_data: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = download_slope_data(refresh=refresh_data)
    performance, predictions, selected_depth = evaluate_models(panel)
    ranking = decision_ranking(panel)
    make_figure_1(panel, predictions)
    make_figure_2(panel)

    baseline_2022 = panel[
        (panel["scenario"] == "reference") & (panel["year"] == 2022)
    ]["emissions_mmtco2"].sum()
    print(f"Rows: {len(panel):,}")
    print(f"NREL SLOPE 2022 energy-related CO2: {baseline_2022:.3f} MMTCO2")
    print(f"Regional GGRP 2022 all-GHG benchmark: {GGRP_2022_MMTCO2E:.1f} MMTCO2e")
    print(f"Validation-selected KG-XGB max depth: {selected_depth}")
    print("\nTable 1 - model performance")
    print(performance.to_string(index=False))
    print("\nTable 2 - decision ranking")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Redownload all public NREL SLOPE scenario series.",
    )
    args = parser.parse_args()
    main(refresh_data=args.refresh_data)
