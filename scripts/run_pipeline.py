"""CLI entry point for the GridSentinel AI pipeline."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gridsentinel.config import load_config, reset_config
from gridsentinel.ingestion.loader import load_from_csv, load_from_parquet
from gridsentinel.pipeline import GridSentinelPipeline, PipelineResult

logger = logging.getLogger(__name__)


def _result_to_dict(result: PipelineResult) -> dict:
    """Serialise PipelineResult to a JSON-compatible dict."""
    return {
        "classification_metrics": {k: float(v) for k, v in result.classification_metrics.items()},
        "forecast_metrics": {k: float(v) for k, v in result.forecast_metrics.items()},
        "economic_cost_inr": float(result.economic_cost_inr),
        "gss_core": float(result.gss_core),
        "gss_final": float(result.gss_final),
        "constraints_met": bool(result.constraints_met),
        "shap_summary_path": result.shap_summary_path,
        "meter_scores_count": len(result.meter_scores),
        "dt_scores_count": len(result.dt_scores),
    }


def _print_summary(result: PipelineResult) -> None:
    """Print a human-readable summary table to stdout."""
    sep = "=" * 60
    print(sep)
    print("  GridSentinel AI Pipeline Summary")
    print(sep)

    cm = result.classification_metrics
    fm = result.forecast_metrics

    print("\n  [Classification Metrics]")
    print(f"    Precision  : {cm.get('precision', 0):.4f}")
    print(f"    Recall     : {cm.get('recall', 0):.4f}")
    print(f"    F1 Score   : {cm.get('f1', 0):.4f}")
    print(f"    FPR        : {cm.get('fpr', 0):.4f}")
    print(f"    MCC        : {cm.get('mcc', 0):.4f}")
    print(f"    ROC-AUC    : {cm.get('roc_auc', 0):.4f}")
    print(f"    PR-AUC     : {cm.get('pr_auc', 0):.4f}")
    print(f"    ECE        : {cm.get('ece', 0):.4f}")

    print("\n  [Forecast Metrics]")
    print(f"    MAPE       : {fm.get('mape', 0):.4f}")
    print(f"    RMSE       : {fm.get('rmse', 0):.4f}")
    print(f"    PICP       : {fm.get('picp', 0):.4f}")
    print(f"    PINAW      : {fm.get('pinaw', 0):.4f}")

    print("\n  [Economic & System Scores]")
    print(f"    Economic Cost  : ₹ {result.economic_cost_inr:,.2f}")
    print(f"    GSS Core       : {result.gss_core:.4f}")
    print(f"    GSS Final      : {result.gss_final:.4f}")
    print(f"    Constraints Met: {result.constraints_met}")

    print("\n  [Meter CASS Distribution]")
    if len(result.meter_scores) > 0:
        for label in ["Normal", "Watch", "Inspect", "Immediate"]:
            count = (result.meter_scores["cass_label"] == label).sum()
            print(f"    {label:10s}: {count}")

    print("\n  [DT GSI Distribution]")
    if len(result.dt_scores) > 0:
        for label in ["Stable", "Caution", "Stressed", "Critical"]:
            count = (result.dt_scores["gsi_label"] == label).sum()
            print(f"    {label:10s}: {count}")

    print(sep)


def main() -> None:
    """Parse CLI args and execute the pipeline."""
    parser = argparse.ArgumentParser(
        description="GridSentinel AI Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--meter", type=str, help="Path to meter readings Parquet file")
    parser.add_argument("--dt", type=str, help="Path to DT readings Parquet file")
    parser.add_argument("--csv", type=str, help="Path to synthetic CSV file (alternative to --meter/--dt)")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--output", type=str, default="results/pipeline_output.json", help="Output JSON path")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for result files")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory for model files")

    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script parent
        config_path = Path(__file__).parent.parent / args.config
    reset_config()
    cfg = load_config(config_path)

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.csv:
        logger.info("Loading from CSV: %s", args.csv)
        meter_df, dt_df = load_from_csv(args.csv)
    elif args.meter and args.dt:
        logger.info("Loading from Parquet: %s, %s", args.meter, args.dt)
        meter_df, dt_df = load_from_parquet(args.meter, args.dt)
    else:
        parser.error("Provide either --csv or both --meter and --dt")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    pipeline = GridSentinelPipeline(
        config=cfg,
        model_dir=model_dir,
        output_dir=results_dir,
    )
    result = pipeline.run(meter_df, dt_df)

    # ── Print summary ─────────────────────────────────────────────────────────
    _print_summary(result)

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(_result_to_dict(result), f, indent=2)
    logger.info("Pipeline output saved to %s", output_path)

    # Save CSV reports
    meter_csv = results_dir / "meter_scores.csv"
    result.meter_scores.to_csv(meter_csv, index=False)

    dt_csv = results_dir / "dt_scores.csv"
    result.dt_scores.to_csv(dt_csv, index=False)

    # Save metrics JSON
    metrics_json = results_dir / "metrics_summary.json"
    with open(metrics_json, "w") as f:
        json.dump(
            {
                "classification": {k: float(v) for k, v in result.classification_metrics.items()},
                "forecast": {k: float(v) for k, v in result.forecast_metrics.items()},
            },
            f, indent=2,
        )

    # Save economic report
    econ_json = results_dir / "economic_report.json"
    cm = result.classification_metrics
    with open(econ_json, "w") as f:
        json.dump(
            {
                "fp_count": int(cm.get("fp", 0)),
                "fn_count": int(cm.get("fn", 0)),
                "economic_cost_inr": float(result.economic_cost_inr),
                "gss_core": float(result.gss_core),
                "gss_final": float(result.gss_final),
                "constraints_met": bool(result.constraints_met),
            },
            f, indent=2,
        )

    print(f"\n  Outputs saved to: {results_dir}")
    print(f"  Pipeline output : {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
