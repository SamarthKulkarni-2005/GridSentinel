"""GridSentinel end-to-end pipeline orchestrator."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from gridsentinel.config import Config, load_config
from gridsentinel.evaluation.classification import evaluate_classification
from gridsentinel.evaluation.forecasting import evaluate_forecast
from gridsentinel.features.cass_signals import (
    signal_dt_balance_error,
    signal_dtw_divergence,
    signal_entropy,
    signal_night_load_anomaly,
    signal_repeat_anomaly,
    signal_voltage_stability,
    signal_billing_ratio,
)
from gridsentinel.features.feature_store import (
    FEATURE_NAMES,
    FORECAST_FEATURES,
    build_forecast_features,
    build_meter_features,
    compute_thermal_hours,
)
from gridsentinel.features.gsi_signals import (
    signal_calendar,
    signal_ev_load_risk,
    signal_load_quantile,
    signal_power_factor_penalty,
    signal_pv_duck_curve,
    signal_temperature_derating,
    signal_thermal_soak,
    signal_transformer_age,
)
from gridsentinel.features.holidays import is_holiday
from gridsentinel.ingestion.validator import validate_dt_df, validate_meter_df
from gridsentinel.models.demand_forecaster import DemandForecaster
from gridsentinel.models.explainability import (
    compute_shap_values,
    get_top_shap_features,
    plot_shap_summary,
)
from gridsentinel.models.theft_detector import FEATURE_NAMES as TD_FEATURE_NAMES
from gridsentinel.models.theft_detector import TheftDetector
from gridsentinel.scoring.cass import cass_label, compute_cass, compute_g_pv
from gridsentinel.scoring.economic import compute_baseline_cost, compute_cost
from gridsentinel.scoring.gsi import compute_gsi, gsi_label
from gridsentinel.scoring.gss import (
    check_pareto_constraints,
    compute_gss_core,
    compute_gss_final,
    compute_s_delay,
    compute_s_energy,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Container for all pipeline outputs."""
    meter_scores: pd.DataFrame
    dt_scores: pd.DataFrame
    classification_metrics: Dict[str, float]
    forecast_metrics: Dict[str, float]
    economic_cost_inr: float
    gss_core: float
    gss_final: float
    constraints_met: bool
    shap_summary_path: str


class GridSentinelPipeline:
    """End-to-end GridSentinel AI pipeline."""

    def __init__(
        self,
        config: Optional[Config] = None,
        config_path: Optional[Path] = None,
        model_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        """Initialise pipeline with configuration."""
        self.cfg = config or load_config(config_path)
        self.model_dir = Path(model_dir) if model_dir else Path("models")
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.theft_detector: Optional[TheftDetector] = None
        self.demand_forecaster: Optional[DemandForecaster] = None

    def run(self, meter_df: pd.DataFrame, dt_df: pd.DataFrame) -> PipelineResult:
        """Execute the full pipeline and return a PipelineResult."""
        logger.info("=== GridSentinel Pipeline START ===")

        # ── Step 1: Validate schemas ──────────────────────────────────────────
        logger.info("Step 1: Validating schemas")
        meter_df = validate_meter_df(meter_df)
        dt_df = validate_dt_df(dt_df)

        # ── Step 2: Resample and align timestamps ─────────────────────────────
        logger.info("Step 2: Resampling data")
        meter_df, dt_df = self._resample(meter_df, dt_df)

        # ── Determine lookback window ─────────────────────────────────────────
        n_meter_rows = meter_df.groupby("meter_id").size().min()
        lookback_window = min(self.cfg.data.lookback_window, max(1, int(n_meter_rows) // 4))
        if lookback_window < self.cfg.data.lookback_window:
            logger.warning(
                "Reduced lookback_window from %d to %d due to data size",
                self.cfg.data.lookback_window, lookback_window,
            )

        # ── Step 3: Build meter features ──────────────────────────────────────
        logger.info("Step 3: Building meter features")
        feature_df, cluster_labels, cluster_centroids = build_meter_features(
            meter_df=meter_df,
            dt_df=dt_df,
            lookback_window=lookback_window,
            n_clusters=10,
            dtw_norm_denom=self.cfg.cass.dtw_norm_denom,
            voltage_std_norm=self.cfg.cass.voltage_std_norm,
            repeat_window_count=self.cfg.cass.repeat_window_count,
        )

        # ── Train/test split (time-based 80/20) ───────────────────────────────
        all_dates = sorted(meter_df["timestamp"].unique())
        split_idx = int(len(all_dates) * 0.8)
        split_date = all_dates[split_idx] if split_idx < len(all_dates) else all_dates[-1]
        logger.info("Train/test split date: %s", split_date)

        train_meter = meter_df[meter_df["timestamp"] < split_date]
        test_meter = meter_df[meter_df["timestamp"] >= split_date]
        train_dt = dt_df[dt_df["timestamp"] < split_date]
        test_dt = dt_df[dt_df["timestamp"] >= split_date]

        # Build feature matrices for train/test
        train_features, _, _ = build_meter_features(
            train_meter, dt_df,
            lookback_window=lookback_window,
            n_clusters=10,
            dtw_norm_denom=self.cfg.cass.dtw_norm_denom,
            voltage_std_norm=self.cfg.cass.voltage_std_norm,
            repeat_window_count=self.cfg.cass.repeat_window_count,
        )
        test_features, _, _ = build_meter_features(
            test_meter, dt_df,
            lookback_window=lookback_window,
            n_clusters=10,
            dtw_norm_denom=self.cfg.cass.dtw_norm_denom,
            voltage_std_norm=self.cfg.cass.voltage_std_norm,
            repeat_window_count=self.cfg.cass.repeat_window_count,
            cluster_labels=cluster_labels,
            cluster_centroids=cluster_centroids,
        )

        # ── Step 4: Train TheftDetector ───────────────────────────────────────
        logger.info("Step 4: Training TheftDetector")
        X_train = train_features[TD_FEATURE_NAMES].fillna(0).values
        y_train = train_features["is_theft"].values
        X_test = test_features[TD_FEATURE_NAMES].fillna(0).values
        y_test = test_features["is_theft"].values

        # Use 10% of train as validation
        val_idx = max(1, int(len(X_train) * 0.1))
        X_val, y_val = X_train[-val_idx:], y_train[-val_idx:]
        X_tr, y_tr = X_train[:-val_idx], y_train[:-val_idx]

        self.theft_detector = TheftDetector(
            n_estimators=self.cfg.models.xgboost.n_estimators,
            max_depth=self.cfg.models.xgboost.max_depth,
            learning_rate=self.cfg.models.xgboost.learning_rate,
            subsample=self.cfg.models.xgboost.subsample,
            colsample_bytree=self.cfg.models.xgboost.colsample_bytree,
            eval_metric=self.cfg.models.xgboost.eval_metric,
            early_stopping_rounds=self.cfg.models.xgboost.early_stopping_rounds,
            random_state=self.cfg.models.xgboost.random_state,
        )

        if len(X_tr) > 0 and len(np.unique(y_tr)) > 1:
            self.theft_detector.fit(X_tr, y_tr, X_val, y_val)
            td_path = self.model_dir / "theft_detector.json"
            self.theft_detector.save(td_path)
        else:
            logger.warning("Insufficient training data for TheftDetector; using random model")
            self.theft_detector.model = None

        # ── Step 5: CASS scores per meter ─────────────────────────────────────
        logger.info("Step 5: Computing CASS scores")
        cass_weights = self.cfg.cass.weights.model_dump()

        # Get solar irradiance per DT for g_pv calculation
        dt_solar = (
            dt_df.groupby("dt_id")["solar_irradiance"].mean().to_dict()
            if "solar_irradiance" in dt_df.columns
            else {}
        )

        meter_score_records = []
        for _, row in feature_df.iterrows():
            signals = {k: float(row.get(k, 0.0)) for k in cass_weights}
            solar = dt_solar.get(row["dt_id"], 0.0)
            g_pv = compute_g_pv(solar)
            score = compute_cass(
                signals=signals,
                weights=cass_weights,
                sigmoid_shift=self.cfg.cass.sigmoid_shift,
                sigmoid_scale=self.cfg.cass.sigmoid_scale,
                g_pv=g_pv,
            )
            label = cass_label(score)

            # Predict proba if model available
            x_feat = np.array([[float(row.get(f, 0.0)) for f in TD_FEATURE_NAMES]])
            if self.theft_detector is not None and self.theft_detector.model is not None:
                y_proba = float(self.theft_detector.predict_proba(x_feat)[0])
            else:
                y_proba = float(score / 100.0)

            meter_score_records.append({
                "meter_id": row["meter_id"],
                "dt_id": row["dt_id"],
                "cass_score": score,
                "cass_label": label,
                "y_proba": y_proba,
                "is_theft": int(row.get("is_theft", 0)),
            })

        meter_scores = pd.DataFrame(meter_score_records)

        # ── Step 6: GSI signals per DT ────────────────────────────────────────
        logger.info("Step 6: Computing GSI signals")
        dt_df_thermal = compute_thermal_hours(dt_df)

        # ── Step 7: Train DemandForecaster ────────────────────────────────────
        logger.info("Step 7: Training DemandForecaster")
        forecast_features_df = build_forecast_features(meter_df, dt_df)
        feat_cols = [c for c in FORECAST_FEATURES if c in forecast_features_df.columns]

        train_fc = forecast_features_df[forecast_features_df["timestamp"] < split_date]
        test_fc = forecast_features_df[forecast_features_df["timestamp"] >= split_date]

        # Per-DT training
        self.demand_forecaster = DemandForecaster(
            hidden_size=self.cfg.models.lstm.hidden_size,
            num_layers=self.cfg.models.lstm.num_layers,
            dropout=self.cfg.models.lstm.dropout,
            quantiles=list(self.cfg.models.lstm.quantiles),
            epochs=self.cfg.models.lstm.epochs,
            batch_size=self.cfg.models.lstm.batch_size,
            lr=self.cfg.models.lstm.lr,
            patience=self.cfg.models.lstm.patience,
            random_state=self.cfg.models.lstm.random_state,
            lookback_window=lookback_window,
        )

        # Train on first DT's data (shared model for simplicity)
        first_dt = forecast_features_df["dt_id"].iloc[0]
        train_fc_dt = train_fc[train_fc["dt_id"] == first_dt].copy()
        test_fc_dt = test_fc[test_fc["dt_id"] == first_dt].copy()

        if len(train_fc_dt) > lookback_window + 1 and len(test_fc_dt) > lookback_window + 1:
            # Use 10% of train as val
            val_split = max(lookback_window + 1, int(len(train_fc_dt) * 0.9))
            val_fc_dt = train_fc_dt.iloc[val_split:]
            train_fc_dt_actual = train_fc_dt.iloc[:val_split]

            if len(val_fc_dt) > lookback_window:
                self.demand_forecaster.fit(
                    train_data=train_fc_dt_actual,
                    val_data=val_fc_dt,
                    feature_cols=feat_cols,
                    target_col="kwh",
                )
            else:
                self.demand_forecaster.fit(
                    train_data=train_fc_dt,
                    val_data=train_fc_dt,
                    feature_cols=feat_cols,
                    target_col="kwh",
                )
        else:
            logger.warning("Insufficient data for DemandForecaster training")
            self.demand_forecaster.fit(
                train_data=forecast_features_df,
                val_data=forecast_features_df,
                feature_cols=feat_cols,
                target_col="kwh",
            )

        fc_model_path = self.model_dir / "demand_forecaster.pt"
        fc_scaler_path = self.model_dir / "demand_scaler.pkl"
        self.demand_forecaster.save(fc_model_path, fc_scaler_path)

        # ── Step 8: GSI scores per DT ─────────────────────────────────────────
        logger.info("Step 8: Computing GSI scores")
        gsi_weights = self.cfg.gsi.weights.model_dump()

        dt_score_records = []
        all_q5, all_q95, all_y_true_load = [], [], []

        for dt_id in dt_df["dt_id"].unique():
            dt_slice = dt_df_thermal[dt_df_thermal["dt_id"] == dt_id].sort_values("timestamp")
            fc_slice = forecast_features_df[forecast_features_df["dt_id"] == dt_id].copy()

            # Forecast
            if len(fc_slice) > lookback_window:
                _preds = self.demand_forecaster.predict(fc_slice, feature_cols=feat_cols)
                q5_arr, q95_arr = _preds[:, 0], _preds[:, -1]
            else:
                mean_kwh = float(fc_slice["kwh"].mean()) if len(fc_slice) > 0 else 0.0
                q5_arr = np.full(len(fc_slice), mean_kwh * 0.9)
                q95_arr = np.full(len(fc_slice), mean_kwh * 1.1)

            # Get actual kwh for this DT from dt_df
            dt_actual = dt_df[dt_df["dt_id"] == dt_id]["actual_kwh"].values
            fc_len = min(len(q5_arr), len(dt_actual))
            all_q5.extend(q5_arr[:fc_len].tolist())
            all_q95.extend(q95_arr[:fc_len].tolist())
            all_y_true_load.extend(dt_actual[:fc_len].tolist())

            # GSI signals from last snapshot
            last = dt_slice.iloc[-1] if len(dt_slice) > 0 else None
            if last is None:
                continue

            q95_kwh = float(np.percentile(q95_arr, 95)) if len(q95_arr) > 0 else 0.0
            cap_kva = float(last.get("capacity_kva", 500.0))
            pf_mean = float(fc_slice["power_factor"].mean()) if len(fc_slice) > 0 else 0.9
            temp = float(last.get("temperature_c", 28.0))
            ev_dens = float(last.get("ev_density", 0.1))
            solar = float(last.get("solar_irradiance", 0.0))
            age = float(last.get("age_years", 10.0))
            hours_above = float(last.get("hours_above_80pct", 0.0))
            feeder_load = float(last.get("feeder_kwh", 0.0))
            baseline_load = float(dt_slice["feeder_kwh"].mean()) if len(dt_slice) > 0 else feeder_load

            ts_last = pd.Timestamp(last["timestamp"])
            hour = ts_last.hour
            dow = ts_last.dayofweek
            month = ts_last.month
            is_evt = is_holiday(ts_last.date())

            signals = {
                "load_quantile": signal_load_quantile(q95_kwh, cap_kva, pf_mean),
                "temperature_derating": signal_temperature_derating(
                    temp,
                    self.cfg.gsi.temp_threshold,
                    self.cfg.gsi.temp_derate_rate,
                ),
                "power_factor_penalty": signal_power_factor_penalty(pf_mean),
                "thermal_soak": signal_thermal_soak(hours_above, self.cfg.gsi.thermal_tau),
                "ev_load_risk": signal_ev_load_risk(ev_dens, hour),
                "transformer_age": signal_transformer_age(age),
                "calendar_signal": signal_calendar(hour, dow, month, is_evt),
                "pv_duck_curve": signal_pv_duck_curve(solar, feeder_load, baseline_load),
            }

            dt_score_records.append({
                "dt_id": dt_id,
                "gsi_signals": signals,
                "q5": float(np.mean(q5_arr)),
                "q95": float(np.mean(q95_arr)),
                "capacity_kva": cap_kva,
                "feeder_kwh": feeder_load,
            })

        # ── Steps 9–10: Evaluation ────────────────────────────────────────────
        logger.info("Steps 9-10: Evaluating classification and forecast")

        # Classification evaluation
        if self.theft_detector is not None and self.theft_detector.model is not None:
            X_all = feature_df[TD_FEATURE_NAMES].fillna(0).values
            y_all = feature_df["is_theft"].values
            y_proba_all = self.theft_detector.predict_proba(X_all)
            y_pred_all = (y_proba_all >= 0.5).astype(int)
        else:
            y_all = feature_df["is_theft"].values
            y_proba_all = meter_scores["y_proba"].values
            y_pred_all = (y_proba_all >= 0.5).astype(int)

        # Guard: ensure at least 2 classes
        if len(np.unique(y_all)) < 2:
            logger.warning("Only one class in labels; using dummy metrics")
            classification_metrics = {
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "fpr": 0.0, "mcc": 0.0, "roc_auc": 0.5, "pr_auc": 0.0,
                "ece": 0.0, "tp": 0, "fp": 0, "tn": int((y_all == 0).sum()), "fn": 0,
            }
        else:
            classification_metrics = evaluate_classification(y_all, y_pred_all, y_proba_all)

        # Forecast evaluation
        if all_y_true_load:
            forecast_metrics = evaluate_forecast(
                np.array(all_y_true_load),
                np.array(all_q5),
                np.array(all_q95),
            )
        else:
            forecast_metrics = {"mape": 0.0, "rmse": 0.0, "picp": 1.0, "pinaw": 0.0}

        # ── Step 11: Robustness testing ───────────────────────────────────────
        logger.info("Step 11: Robustness testing")
        f1_clean = classification_metrics["f1"]

        if self.theft_detector is not None and self.theft_detector.model is not None:
            rng_robust = np.random.default_rng(self.cfg.robustness.noise_seed)
            X_noisy = X_all + rng_robust.normal(
                0, self.cfg.robustness.noise_std, size=X_all.shape
            )
            y_noisy_proba = self.theft_detector.predict_proba(X_noisy)
            y_noisy_pred = (y_noisy_proba >= 0.5).astype(int)
            if len(np.unique(y_all)) > 1:
                noisy_metrics = evaluate_classification(y_all, y_noisy_pred, y_noisy_proba)
                f1_noisy = noisy_metrics["f1"]
            else:
                f1_noisy = f1_clean
        else:
            f1_noisy = f1_clean

        s_robust = float(np.clip(
            1.0 - abs(f1_clean - f1_noisy) / max(f1_clean, 1e-9),
            0.0, 1.0
        ))

        # ── Step 12: Temporal stability ───────────────────────────────────────
        logger.info("Step 12: Computing temporal stability")
        s_temporal = self._compute_temporal_stability(meter_scores)

        # ── Step 13: Energy consistency ───────────────────────────────────────
        logger.info("Step 13: Computing energy consistency")
        total_meter_kwh = float(meter_df["kwh"].sum())
        total_feeder_kwh = float(dt_df["feeder_kwh"].sum()) if "feeder_kwh" in dt_df.columns else total_meter_kwh
        s_energy = compute_s_energy(total_meter_kwh, total_feeder_kwh)

        # ── Step 14: Detection latency ────────────────────────────────────────
        logger.info("Step 14: Computing detection latency")
        # Default to 0 days latency (ground truth theft start unavailable)
        detection_latency_days = 0.0
        s_delay = compute_s_delay(detection_latency_days)

        # ── Step 15: ECE ──────────────────────────────────────────────────────
        ece = classification_metrics.get("ece", 0.0)
        s_calib = float(np.clip(1.0 - ece, 0.0, 1.0))

        # ── Step 16: Pareto constraints ───────────────────────────────────────
        logger.info("Step 16: Checking Pareto constraints")
        constraints_cfg = self.cfg.constraints.model_dump()
        constraints_met = check_pareto_constraints(
            fpr=classification_metrics["fpr"],
            recall=classification_metrics["recall"],
            mape=forecast_metrics["mape"],
            cfg=constraints_cfg,
        )

        # ── Step 17: Economic cost ────────────────────────────────────────────
        logger.info("Step 17: Computing economic cost")
        fp_count = int(classification_metrics.get("fp", 0))
        fn_count = int(classification_metrics.get("fn", 0))
        tp_count = int(classification_metrics.get("tp", 0))
        total_theft = tp_count + fn_count

        cost = compute_cost(
            fp_count=fp_count,
            fn_count=fn_count,
            fp_cost=self.cfg.economics.fp_cost_inr,
            fn_monthly_cost=self.cfg.economics.fn_cost_per_month_inr,
            theft_months=self.cfg.economics.default_theft_duration_months,
        )
        baseline_cost = compute_baseline_cost(
            total_cases=total_theft,
            fn_monthly_cost=self.cfg.economics.fn_cost_per_month_inr,
            theft_months=self.cfg.economics.default_theft_duration_months,
        )

        # ── Component scores for GSS ──────────────────────────────────────────
        mcc = classification_metrics.get("mcc", 0.0)
        fpr = classification_metrics.get("fpr", 0.0)
        s_cass_score = float(np.clip(mcc * (1.0 - fpr), 0.0, 1.0))

        mape_val = forecast_metrics["mape"]
        picp_val = forecast_metrics["picp"]
        pinaw_val = forecast_metrics["pinaw"]
        s_gsi_score = float(np.clip((1.0 - mape_val) * picp_val * (1.0 - pinaw_val), 0.0, 1.0))

        s_econ_score = float(np.clip(1.0 - cost / max(baseline_cost, 1.0), 0.0, 1.0))

        # ── Step 18: GSS scores ───────────────────────────────────────────────
        logger.info("Step 18: Computing GSS scores")
        core_weights = self.cfg.gss_core.model_dump()
        final_weights = self.cfg.gss_final.model_dump()

        gss_core_score = compute_gss_core(
            s_cass=s_cass_score,
            s_gsi=s_gsi_score,
            s_econ=s_econ_score,
            s_robust=s_robust,
            weights=core_weights,
            constraints_met=constraints_met,
        )

        gss_final_score = compute_gss_final(
            s_cass=s_cass_score,
            s_gsi=s_gsi_score,
            s_econ=s_econ_score,
            s_robust=s_robust,
            s_delay=s_delay,
            s_energy=s_energy,
            s_calib=s_calib,
            weights=final_weights,
            constraints_met=constraints_met,
        )

        # ── GSI score DataFrame ───────────────────────────────────────────────
        u_tconf = float(np.clip(1.0 - pinaw_val, 0.0, 1.0))
        mape_scale = float(np.clip(1.0 - mape_val, 0.0, 1.0))

        dt_rows = []
        for rec in dt_score_records:
            gsi_score = compute_gsi(
                signals=rec["gsi_signals"],
                weights=gsi_weights,
                u_tconf=u_tconf,
                mape_scale=mape_scale,
            )
            label = gsi_label(gsi_score)
            dt_rows.append({
                "dt_id": rec["dt_id"],
                "gsi_score": gsi_score,
                "gsi_label": label,
                "q5": rec["q5"],
                "q95": rec["q95"],
            })
        dt_scores = pd.DataFrame(dt_rows)

        # ── SHAP explainability ───────────────────────────────────────────────
        shap_path = str(self.output_dir / "shap_summary.png")
        if self.theft_detector is not None and self.theft_detector.model is not None:
            try:
                shap_vals = compute_shap_values(
                    self.theft_detector.model,
                    X_all[:500],  # limit for speed
                    feature_names=TD_FEATURE_NAMES,
                )
                shap_path = plot_shap_summary(shap_vals, X_all[:500], TD_FEATURE_NAMES, shap_path)
            except Exception as exc:
                logger.warning("SHAP failed: %s", exc)

        logger.info("=== GridSentinel Pipeline COMPLETE ===")
        logger.info("GSS Core: %.4f | GSS Final: %.4f | Constraints met: %s",
                    gss_core_score, gss_final_score, constraints_met)

        return PipelineResult(
            meter_scores=meter_scores,
            dt_scores=dt_scores,
            classification_metrics=classification_metrics,
            forecast_metrics=forecast_metrics,
            economic_cost_inr=cost,
            gss_core=gss_core_score,
            gss_final=gss_final_score,
            constraints_met=constraints_met,
            shap_summary_path=shap_path,
        )

    def _resample(
        self,
        meter_df: pd.DataFrame,
        dt_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Resample meter and DT data to configured frequency."""
        freq = self.cfg.data.resample_freq
        logger.debug("Resampling to frequency: %s", freq)
        # Data is already at some interval; sort and deduplicate
        meter_df = meter_df.sort_values(["meter_id", "timestamp"]).drop_duplicates(
            subset=["meter_id", "timestamp"]
        )
        dt_df = dt_df.sort_values(["dt_id", "timestamp"]).drop_duplicates(
            subset=["dt_id", "timestamp"]
        )
        return meter_df, dt_df

    def _compute_temporal_stability(self, meter_scores: pd.DataFrame) -> float:
        """Compute temporal stability score across all meters.

        S_temporal = 1 - isolated_spikes / total_anomalies
        An isolated spike: single-interval CASS >= 60 surrounded by CASS < 35.
        """
        if "cass_score" not in meter_scores.columns:
            return 1.0

        scores = meter_scores["cass_score"].values
        if len(scores) < 3:
            return 1.0

        anomaly_mask = scores >= 60.0
        total_anomalies = int(anomaly_mask.sum())
        if total_anomalies == 0:
            return 1.0

        isolated = 0
        for i in range(1, len(scores) - 1):
            if anomaly_mask[i] and scores[i - 1] < 35.0 and scores[i + 1] < 35.0:
                isolated += 1

        return float(np.clip(1.0 - isolated / max(total_anomalies, 1), 0.0, 1.0))
