"""Generate GridSentinel AI technical documentation report as a PDF."""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime

OUTPUT = "GridSentinel_Report.pdf"

# ── Colours ──────────────────────────────────────────────────────────────────
NAVY      = colors.HexColor("#003366")
BLUE_MID  = colors.HexColor("#0055A5")
BLUE_LITE = colors.HexColor("#D6E4F7")
BLUE_ALT  = colors.HexColor("#EBF3FB")
WHITE     = colors.white
DARK_GREY = colors.HexColor("#333333")
MID_GREY  = colors.HexColor("#666666")
ROW_ALT   = colors.HexColor("#F4F8FD")
ORANGE    = colors.HexColor("#E07B00")
GREEN     = colors.HexColor("#1A7A3E")

W, H = A4

# ── Styles ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def style(name, **kw):
    return ParagraphStyle(name, **kw)

S = {
    "title":    style("GS_Title",    fontSize=28, textColor=WHITE,     alignment=TA_CENTER, spaceAfter=6,  fontName="Helvetica-Bold"),
    "subtitle": style("GS_Sub",      fontSize=14, textColor=BLUE_LITE, alignment=TA_CENTER, spaceAfter=4,  fontName="Helvetica"),
    "meta":     style("GS_Meta",     fontSize=11, textColor=BLUE_LITE, alignment=TA_CENTER, spaceAfter=2,  fontName="Helvetica"),
    "h1":       style("GS_H1",       fontSize=16, textColor=WHITE,     spaceBefore=2, spaceAfter=6,  fontName="Helvetica-Bold", leftIndent=0),
    "h2":       style("GS_H2",       fontSize=13, textColor=NAVY,      spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold"),
    "h3":       style("GS_H3",       fontSize=11, textColor=BLUE_MID,  spaceBefore=6,  spaceAfter=3, fontName="Helvetica-Bold"),
    "body":     style("GS_Body",     fontSize=9,  textColor=DARK_GREY, spaceAfter=4,  fontName="Helvetica",      leading=14, alignment=TA_JUSTIFY),
    "bullet":   style("GS_Bullet",   fontSize=9,  textColor=DARK_GREY, spaceAfter=2,  fontName="Helvetica",      leftIndent=14, leading=13, bulletIndent=5),
    "mono":     style("GS_Mono",     fontSize=8,  textColor=DARK_GREY, spaceAfter=2,  fontName="Courier",        leading=12, leftIndent=12),
    "toc":      style("GS_TOC",      fontSize=10, textColor=NAVY,      spaceAfter=4,  fontName="Helvetica",      leftIndent=8),
    "toch":     style("GS_TOCH",     fontSize=12, textColor=NAVY,      spaceAfter=6,  fontName="Helvetica-Bold"),
    "caption":  style("GS_Caption",  fontSize=8,  textColor=MID_GREY,  spaceAfter=6,  fontName="Helvetica-Oblique", alignment=TA_CENTER),
    "formula":  style("GS_Formula",  fontSize=9,  textColor=NAVY,      spaceAfter=4,  fontName="Courier-Bold",   leftIndent=20, leading=14),
    "label":    style("GS_Label",    fontSize=8,  textColor=WHITE,     fontName="Helvetica-Bold", alignment=TA_CENTER),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def p(text, sty="body"):   return Paragraph(text, S[sty])
def sp(n=6):               return Spacer(1, n)
def hr():                  return HRFlowable(width="100%", thickness=1, color=BLUE_LITE, spaceAfter=6, spaceBefore=6)

def section_header(num, title):
    """Blue banner heading for each numbered section."""
    t = Table([[Paragraph(f"{num}. {title}", S["h1"])]],
              colWidths=[W - 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return [sp(10), t, sp(8)]

def sub_header(title):
    return [sp(4), p(title, "h2"), hr()]

def table(data, col_widths, header_bg=NAVY, alt=True):
    rows = len(data)
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0),      header_bg),
        ("TEXTCOLOR",     (0, 0), (-1, 0),      WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),      "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),      9),
        ("FONTNAME",      (0, 1), (-1, -1),     "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1),     8),
        ("TEXTCOLOR",     (0, 1), (-1, -1),     DARK_GREY),
        ("ROWBACKGROUND", (0, 1), (-1, -1),     [WHITE, ROW_ALT] if alt else [WHITE]),
        ("GRID",          (0, 0), (-1, -1),     0.4, colors.HexColor("#C5D8EE")),
        ("TOPPADDING",    (0, 0), (-1, -1),     5),
        ("BOTTOMPADDING", (0, 0), (-1, -1),     5),
        ("LEFTPADDING",   (0, 0), (-1, -1),     7),
        ("RIGHTPADDING",  (0, 0), (-1, -1),     7),
        ("VALIGN",        (0, 0), (-1, -1),     "TOP"),
        ("WORDWRAP",      (0, 0), (-1, -1),     True),
    ]
    formatted = []
    for r_idx, row in enumerate(data):
        fmt_row = []
        for c_idx, cell in enumerate(row):
            if r_idx == 0:
                fmt_row.append(Paragraph(str(cell), S["label"]) if cell else Paragraph("", S["label"]))
            else:
                fmt_row.append(Paragraph(str(cell), S["body"]) if cell else Paragraph("", S["body"]))
        formatted.append(fmt_row)
    t = Table(formatted, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle(style_cmds))
    return t

def bullet(text):
    return p(f"<bullet>&bull;</bullet> {text}", "bullet")

def formula(text):
    return p(text, "formula")

# ── Page number footer ────────────────────────────────────────────────────────
def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MID_GREY)
    canvas.drawString(cm, 0.7*cm, "GridSentinel AI v2.0 — BESCOM Technical Report")
    canvas.drawRightString(W - cm, 0.7*cm, f"Page {doc.page}")
    canvas.setStrokeColor(BLUE_LITE)
    canvas.setLineWidth(0.5)
    canvas.line(cm, 0.9*cm, W - cm, 0.9*cm)
    canvas.restoreState()

# ── Build story ───────────────────────────────────────────────────────────────
story = []

# ══════════════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════════════
title_data = [[Paragraph("GridSentinel AI", S["title"])]]
title_tbl = Table(title_data, colWidths=[W - 4*cm])
title_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), NAVY),
    ("TOPPADDING",    (0,0), (-1,-1), 40),
    ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ("LEFTPADDING",   (0,0), (-1,-1), 20),
    ("RIGHTPADDING",  (0,0), (-1,-1), 20),
]))
story.append(sp(60))
story.append(title_tbl)

sub_data = [[Paragraph("v2.0 — Smart Grid Intelligence Platform for BESCOM", S["subtitle"])]]
sub_tbl = Table(sub_data, colWidths=[W - 4*cm])
sub_tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), BLUE_MID),
    ("TOPPADDING",    (0,0), (-1,-1), 10),
    ("BOTTOMPADDING", (0,0), (-1,-1), 14),
    ("LEFTPADDING",   (0,0), (-1,-1), 20),
    ("RIGHTPADDING",  (0,0), (-1,-1), 20),
]))
story.append(sub_tbl)
story.append(sp(30))

info_rows = [
    ["Project", "GridSentinel AI — Electricity Theft Detection & Grid Stress Monitoring"],
    ["Client",  "Bangalore Electricity Supply Company (BESCOM)"],
    ["Version", "2.0"],
    ["Date",    datetime.now().strftime("%d %B %Y")],
    ["Scope",   "Hackathon Technical Documentation"],
]
info_tbl = Table(info_rows, colWidths=[3.5*cm, 12*cm])
info_tbl.setStyle(TableStyle([
    ("FONTNAME",   (0,0), (0,-1),   "Helvetica-Bold"),
    ("FONTNAME",   (1,0), (1,-1),   "Helvetica"),
    ("FONTSIZE",   (0,0), (-1,-1),  10),
    ("TEXTCOLOR",  (0,0), (0,-1),   NAVY),
    ("TEXTCOLOR",  (1,0), (1,-1),   DARK_GREY),
    ("TOPPADDING",    (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LINEBELOW",  (0,0), (-1,-2),  0.3, BLUE_LITE),
]))
story.append(info_tbl)
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS
# ══════════════════════════════════════════════════════════════════
story.append(sp(10))
story.append(p("Table of Contents", "toch"))
hr_toc = HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=10)
story.append(hr_toc)
toc_items = [
    ("1.", "Executive Summary"),
    ("2.", "System Architecture & Workflow"),
    ("3.", "Tools & Technologies"),
    ("4.", "Input Data Schema"),
    ("5.", "The 7 CASS Signals — Theft Detection"),
    ("6.", "The 8 GSI Signals — Grid Stress Index"),
    ("7.", "Machine Learning Models"),
    ("8.", "GSS Composite Scoring"),
    ("9.", "Economic Model"),
    ("10.", "Frontend UI — Tabs & Components"),
    ("11.", "API Endpoints Reference"),
    ("12.", "Classification Metrics Explained"),
    ("13.", "Forecast Metrics Explained"),
]
for num, title in toc_items:
    row_tbl = Table([[Paragraph(f"<b>{num}</b>", S["body"]), Paragraph(title, S["toc"])]],
                    colWidths=[1.2*cm, W - 5.2*cm])
    row_tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LINEBELOW",     (0,0), (-1,-1), 0.3, BLUE_LITE),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(row_tbl)
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 1. EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════
story += section_header("1", "Executive Summary")
story.append(p(
    "GridSentinel AI v2.0 is a production-grade smart grid intelligence platform developed for the "
    "Bangalore Electricity Supply Company (BESCOM). The system addresses two critical operational "
    "challenges in modern power distribution: <b>electricity theft detection</b> at the meter level "
    "and <b>transformer stress monitoring</b> at the distribution transformer (DT) level."
))
story.append(sp(4))
story.append(p(
    "The platform ingests hourly smart meter telemetry for up to 1,000 meters across 21 distribution "
    "transformers, applies a 7-signal anomaly scoring engine (CASS), an 8-signal grid stress index "
    "(GSI), a Bi-LSTM quantile demand forecaster, and an XGBoost theft classifier. Results are "
    "surfaced through a real-time React dashboard with AI-powered natural language querying via "
    "Ollama/LLaMA 3.2."
))
story.append(sp(8))

kpi_data = [
    ["Outcome", "Description", "Scale"],
    ["CASS Score",    "Comparative Anomaly Signal Score per meter — theft suspicion index", "0 to 100"],
    ["GSI Score",     "Grid Stress Index per transformer — operational risk level",           "0 to 100"],
    ["GSS Core",      "Composite system score combining theft, forecast and economic quality", "0.0 to 1.0"],
    ["GSS Final",     "Extended GSS including calibration, energy balance, detection latency", "0.0 to 1.0"],
    ["Economic KPIs", "Revenue protection vs investigation cost in Indian Rupees (INR)",       "INR"],
]
story.append(table(kpi_data, [4*cm, 9.5*cm, 3*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 2. SYSTEM ARCHITECTURE
# ══════════════════════════════════════════════════════════════════
story += section_header("2", "System Architecture & Workflow")

story += sub_header("2.1  High-Level Block Diagram")
story.append(p(
    "The following diagram illustrates the end-to-end data flow from raw CSV ingestion through to "
    "the frontend dashboard."
))
story.append(sp(4))

diagram_text = (
    "  [CSV / Smart Meter Data]  (672,000 rows x 14 columns)\n"
    "           |\n"
    "           v\n"
    "  +--------------------+\n"
    "  |  Ingestion Layer   |  Column mapping, type casting, 3-phase synthesis,\n"
    "  |  (loader.py)       |  billed_kWh derivation, schema validation\n"
    "  +--------------------+\n"
    "           |\n"
    "     +-----+------+\n"
    "     |            |\n"
    "     v            v\n"
    "  meter_df      dt_df\n"
    " (672k rows)  (14k rows)\n"
    "     |            |\n"
    "     v            v\n"
    "  +------------------+       +--------------------+\n"
    "  | Feature Store    |       | Forecast Features  |\n"
    "  | (7 CASS signals  |       | (13 LSTM features  |\n"
    "  |  + 7 engineered) |       |  per DT per hour)  |\n"
    "  +------------------+       +--------------------+\n"
    "           |                          |\n"
    "           v                          v\n"
    "  +------------------+       +--------------------+\n"
    "  | XGBoost Theft    |       | Bi-LSTM Demand     |\n"
    "  | Detector         |       | Forecaster         |\n"
    "  | (14 features in) |       | (Q5, Q95 out)      |\n"
    "  +------------------+       +--------------------+\n"
    "           |                          |\n"
    "           v                          v\n"
    "  +------------------+       +--------------------+\n"
    "  | CASS Scorer      |       | GSI Scorer         |\n"
    "  | (weighted sigmoid|       | (8 signals weighted|\n"
    "  |  per meter)      |       |  per transformer)  |\n"
    "  +------------------+       +--------------------+\n"
    "           |                          |\n"
    "           +----------+  +-----------+\n"
    "                      |  |\n"
    "                      v  v\n"
    "             +-------------------+\n"
    "             |  GSS Compositor   |\n"
    "             |  (7 sub-scores -> |\n"
    "             |   GSS Core/Final) |\n"
    "             +-------------------+\n"
    "                      |\n"
    "                      v\n"
    "             +-------------------+\n"
    "             |  FastAPI REST API  |\n"
    "             |  (SQLite + JWT)   |\n"
    "             +-------------------+\n"
    "                      |\n"
    "                      v\n"
    "             +-------------------+\n"
    "             |  React Dashboard  |\n"
    "             |  Vite + Recharts  |\n"
    "             |  Ollama LLM Chat  |\n"
    "             +-------------------+"
)
box = Table([[Paragraph(diagram_text.replace("\n", "<br/>"), S["mono"])]],
            colWidths=[W - 4*cm])
box.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F0F6FF")),
    ("BOX",        (0,0), (-1,-1), 0.8, BLUE_MID),
    ("TOPPADDING",    (0,0), (-1,-1), 10),
    ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ("LEFTPADDING",   (0,0), (-1,-1), 10),
    ("RIGHTPADDING",  (0,0), (-1,-1), 10),
]))
story.append(box)
story.append(sp(6))
story.append(p("Figure 1: GridSentinel AI end-to-end data flow", "caption"))

story += sub_header("2.2  Pipeline Steps (18 Steps)")
story.append(p("The <b>GridSentinelPipeline.run()</b> method executes the following ordered steps:"))
story.append(sp(4))

pipeline_data = [
    ["Step", "Name", "Description"],
    ["1",  "Schema Validation",       "Validates column presence, types and value ranges for meter_df and dt_df using validator.py"],
    ["2",  "Resample & Align",        "Sorts by meter_id/dt_id + timestamp, removes duplicate (meter, timestamp) pairs"],
    ["3",  "Meter Feature Building",  "Computes DTW clusters via tslearn KMeans on hourly profiles, then calculates all 7 CASS signals and 7 engineered features for each meter"],
    ["4",  "Train/Test Split",        "Time-based 80/20 split on sorted unique timestamps — preserves temporal ordering"],
    ["5",  "TheftDetector Training",  "Trains XGBoost classifier on training-set feature matrix with scale_pos_weight for class imbalance; saves to models/theft_detector.json"],
    ["6",  "CASS Score Computation",  "Applies weighted sigmoid formula to 7 signals per meter using trained model probabilities where available"],
    ["7",  "GSI Signal Computation",  "Calls compute_thermal_hours() to derive hours_above_80pct rolling sums per DT"],
    ["8",  "DemandForecaster Training","Builds 13-feature forecast matrix per DT, trains Bi-LSTM with quantile loss on the first DT as representative; saves model + scaler"],
    ["9",  "GSI Score per DT",        "For each transformer: runs demand forecast, extracts Q95, computes 8 GSI signals, derives GSI score"],
    ["10", "Classification Evaluation","Computes precision, recall, F1, FPR, MCC, ROC-AUC, PR-AUC, ECE on full feature matrix"],
    ["11", "Robustness Testing",       "Injects Gaussian noise (std=0.05) into features, measures F1 drop; S_Robust = 1 - |F1_clean - F1_noisy| / F1_clean"],
    ["12", "Temporal Stability",       "Counts isolated CASS spikes (score >= 60 surrounded by <35) as fraction of anomalies"],
    ["13", "Energy Consistency",       "S_Energy = 1 - |sum_meter_kWh - sum_feeder_kWh| / sum_feeder_kWh"],
    ["14", "Detection Latency",        "S_Delay = exp(-latency_days / 7); defaults to 1.0 when theft start timestamps unavailable"],
    ["15", "Calibration Score",        "S_Calib = 1 - ECE (Expected Calibration Error from 10-bin reliability diagram)"],
    ["16", "Pareto Constraints",       "Checks: FPR <= 0.02, Recall >= 0.85, MAPE <= 0.07; constraints_met flag penalises GSS by 50%"],
    ["17", "Economic Cost",            "cost = FP_count x 8500 + FN_count x 5500 x 3; S_Econ = 1 - cost / baseline_cost"],
    ["18", "GSS Computation",          "GSS Core and GSS Final via weighted sums of sub-scores; SHAP summary plot saved to results/shap_summary.png"],
]
story.append(table(pipeline_data, [1*cm, 4.5*cm, 11*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 3. TOOLS & TECHNOLOGIES
# ══════════════════════════════════════════════════════════════════
story += section_header("3", "Tools & Technologies")

tech_data = [
    ["Tool / Library", "Version", "Role in GridSentinel"],
    ["Python",              "3.11+",     "Primary runtime language for all backend components"],
    ["FastAPI",             "0.110+",    "Async REST API framework — handles all HTTP endpoints"],
    ["Uvicorn",             "0.29+",     "ASGI server hosting the FastAPI application"],
    ["SQLAlchemy",          "2.0+",      "ORM for SQLite — User and AuditLog tables"],
    ["SQLite",              "Built-in",  "Embedded database for users and audit logs"],
    ["XGBoost",             "2.0+",      "Gradient-boosted tree classifier for electricity theft detection"],
    ["PyTorch",             "2.2+",      "Deep learning framework powering the Bi-LSTM demand forecaster"],
    ["tslearn",             "0.6+",      "Time-series DTW K-Means clustering for meter load profile grouping"],
    ["SHAP",                "0.45+",     "SHapley Additive exPlanations for XGBoost model interpretability"],
    ["LIME",                "0.2+",      "Local Interpretable Model-agnostic Explanations (available)"],
    ["scikit-learn",        "1.4+",      "StandardScaler, KMeans fallback for clustering, metrics"],
    ["scipy",               "1.12+",     "Shannon entropy calculation for the entropy CASS signal"],
    ["pandas",              "2.1+",      "Tabular data manipulation, groupby, resampling, merges"],
    ["numpy",               "1.26+",     "Array operations, DTW norm, sigmoid, polyfit for trend slope"],
    ["pyarrow",             "15.0+",     "Parquet file I/O for alternative data loading path"],
    ["React",               "19+",       "Frontend UI framework — component-based single-page application"],
    ["Vite",                "latest",    "Frontend build tool with dev-server proxy to FastAPI"],
    ["Recharts",            "3.8+",      "Chart library for DTW divergence and demand forecast visualisations"],
    ["Axios",               "1.16+",     "HTTP client used in App.jsx for authenticated API calls"],
    ["Lucide React",        "latest",    "Icon set used throughout the dashboard UI"],
    ["python-jose",         "3.3+",      "JWT encoding/decoding for HS256 bearer token authentication"],
    ["passlib + bcrypt",    "1.7+ / 4+", "Secure password hashing for the user credential store"],
    ["fpdf2",               "2.7+",      "PDF report generation for the /api/report/summary endpoint"],
    ["ollama",              "0.4+",      "Python client for local LLaMA 3.2 inference via Ollama daemon"],
    ["LLaMA 3.2",           "3.2",       "Local large language model for the AI chat assistant panel"],
    ["matplotlib",          "3.7+",      "SHAP summary plot generation saved to results/shap_summary.png"],
]
story.append(table(tech_data, [4*cm, 2.5*cm, 10*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 4. INPUT DATA SCHEMA
# ══════════════════════════════════════════════════════════════════
story += section_header("4", "Input Data Schema")
story.append(p(
    "The system accepts a single flat CSV file containing smart meter readings. Each row represents "
    "one meter reading at one timestamp. The loader maps CSV column names to internal schema "
    "names and derives additional columns automatically."
))
story.append(sp(6))

story += sub_header("4.1  Required CSV Columns")
col_data = [
    ["CSV Column Name",          "Internal Name",        "Type",    "Description"],
    ["Meter_ID",                 "meter_id",             "String",  "Unique identifier for each smart meter (e.g. MTR_000001)"],
    ["Transformer_ID",           "dt_id",                "String",  "Distribution transformer this meter is connected to (e.g. DT_1)"],
    ["Timestamp",                "timestamp",            "Datetime","UTC datetime of the reading. Parsed with timezone awareness."],
    ["Active_Power (kWh)",       "kwh",                  "Float",   "Energy consumed during this interval in kilowatt-hours"],
    ["Voltage",                  "voltage_r",            "Float",   "Primary phase (R-phase) voltage in volts"],
    ["Current",                  "current",              "Float",   "Current draw in amperes"],
    ["Power_Factor",             "power_factor",         "Float",   "Ratio of real to apparent power. Range: 0.0 to 1.0"],
    ["Is_Theft (0/1)",           "is_theft",             "Int",     "Ground truth theft label (1 = theft, 0 = normal). Used for model training and evaluation only."],
    ["Imputation_Flag",          "imputation_flag",      "Int",     "1 if this reading was imputed/estimated, 0 if measured directly"],
    ["Imputation_Confidence",    "imputation_confidence","Float",   "Confidence score of the imputation model (0.0 to 1.0)"],
]
story.append(table(col_data, [4.2*cm, 3.5*cm, 1.8*cm, 7*cm]))
story.append(sp(8))

story += sub_header("4.2  Optional CSV Columns")
opt_data = [
    ["CSV Column Name",    "Internal Name",       "Description"],
    ["Peer_Group_ID",      "peer_group_id",       "Pre-assigned peer group for comparative analysis. Used as passthrough metadata."],
    ["Theft_Type",         "theft_type",          "Category of theft (e.g. bypass, tamper). Metadata only — not used in model features."],
    ["GSI_Event_Type",     "gsi_event_type",      "Type of grid stress event associated with this reading. Metadata passthrough."],
    ["Topology_Confidence","topology_confidence", "Confidence score that the reported DT-meter topology is correct (0.0 to 1.0)."],
]
story.append(table(opt_data, [4.2*cm, 4*cm, 8.3*cm]))
story.append(sp(8))

story += sub_header("4.3  Derived Columns (Auto-generated by Loader)")
der_data = [
    ["Derived Column",   "Source",              "Description"],
    ["voltage_y",        "voltage_r + noise",   "Y-phase voltage synthesised by adding uniform noise U(-2, +2) V to R-phase"],
    ["voltage_b",        "voltage_r + noise",   "B-phase voltage synthesised by adding uniform noise U(-2, +2) V to R-phase"],
    ["billed_kwh",       "Monthly kWh sum",     "Sum of kwh per meter per calendar month — represents the billed amount for billing ratio calculation"],
    ["feeder_kwh",       "DT aggregate",        "Sum of all meter kwh under the same DT at the same timestamp — represents the feeder measurement"],
    ["actual_kwh",       "feeder_kwh copy",     "Alias of feeder_kwh used as the LSTM forecast target"],
    ["temperature_c",    "Diurnal formula",     "Synthetic: 28 + 4 x sin(2*pi*hour/24). Ranges 24 to 32 degrees C."],
    ["solar_irradiance", "Hour formula",        "Synthetic: max(0, 800 x sin(pi*(hour-6)/12)) during daylight (6-18h), 0 otherwise"],
    ["ev_density",       "Constant",            "Default 0.1 — EV penetration density proxy"],
    ["capacity_kva",     "Constant per DT",     "Fixed at 500 kVA per transformer"],
    ["age_years",        "DT index formula",    "Cycles through 5, 10, 15, 20 years based on transformer sort order"],
]
story.append(table(der_data, [3.8*cm, 4*cm, 8.7*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 5. CASS SIGNALS
# ══════════════════════════════════════════════════════════════════
story += section_header("5", "The 7 CASS Signals — Comparative Anomaly Signal Scoring")
story.append(p(
    "CASS (Comparative Anomaly Signal Scoring) is GridSentinel's primary theft detection framework. "
    "It computes seven complementary signals per meter, weights them, and applies a sigmoid "
    "transformation to produce a theft suspicion score from 0 to 100. Each signal is normalised "
    "to [0, 1] before weighting."
))
story.append(sp(6))

story += sub_header("5.1  Signal Definitions")
sig_data = [
    ["Signal", "Weight", "Formula / Logic", "What It Detects"],
    ["DTW Divergence",     "0.25",
     "norm(meter_hourly_profile - cluster_centroid) / dtw_norm_denom (default 3.0), clipped to [0,1]",
     "Abnormal load shape relative to peer meters in the same DTW cluster. A thief often manipulates bypass circuits creating an atypical daily profile."],
    ["Voltage Stability",  "0.10",
     "std(voltage_R, voltage_Y, voltage_B) / 0.15, clipped to [0,1]",
     "Voltage irregularities across the three phases indicating meter tampering, loose connections, or bypass wiring that introduces noise."],
    ["Billing Ratio",      "0.20",
     "|1 - billed_kWh / actual_kWh|, clipped to [0,1]",
     "Discrepancy between billed consumption and measured consumption. Under-billing relative to actual usage is a strong theft indicator."],
    ["Entropy",            "0.10",
     "1 - (Shannon entropy of 20-bin histogram / log(20))",
     "Unusually repetitive or predictable consumption patterns. Low entropy means the meter reports nearly the same value repeatedly — a sign of manipulation."],
    ["Night Load Anomaly", "0.15",
     "clip((night_avg / day_avg - 1) / 2, 0, 1)\nNight = 22:00-05:59, Day = 06:00-21:59",
     "Disproportionately high consumption at night relative to daytime. Theft often peaks at night to avoid detection. Signal is 0 when night <= day."],
    ["DT Balance Error",   "0.15",
     "|sum_all_meters_kWh - feeder_kWh| / feeder_kWh, clipped to [0,1]",
     "Energy loss at the transformer level. If the sum of all meters is significantly less than the feeder reading, energy is being taken without metering."],
    ["Repeat Anomaly",     "0.05",
     "Fraction of 96-interval windows where any Z-score > 3.0, divided by window_count (5)",
     "Systematic repeating anomalous consumption windows. Tampered meters often produce regular spikes or flat segments that appear as statistical outliers repeatedly."],
]
story.append(table(sig_data, [3.2*cm, 1.5*cm, 5.5*cm, 6.3*cm]))
story.append(sp(8))

story += sub_header("5.2  CASS Scoring Formula")
story.append(p("The final CASS score is computed as follows:"))
story.append(sp(4))
story.append(formula("raw_score = sum( weight[k] x signal[k] )  for k in {7 signals}"))
story.append(formula("CASS = sigmoid( (raw_score - 0.35) x 8.0 ) x (1 - g_pv) x 100"))
story.append(formula("g_pv  = clip( solar_irradiance / 1000 x 0.3,  0.0,  0.3 )"))
story.append(sp(4))
story.append(p(
    "The sigmoid shift (0.35) and scale (8.0) are calibrated so that a raw score below 0.35 maps to "
    "less than 50 CASS. The PV correction factor <b>g_pv</b> reduces the CASS score by up to 30% "
    "during high solar irradiance periods, preventing false alarms caused by grid-tied solar "
    "installations altering apparent consumption profiles."
))
story.append(sp(6))

story += sub_header("5.3  CASS Label Thresholds")
label_data = [
    ["CASS Range", "Label",      "Recommended Action"],
    ["0 – 34",     "Normal",     "No action. Meter operating within expected parameters."],
    ["35 – 59",    "Watch",      "Monitor for trend. Flag for next billing cycle review."],
    ["60 – 79",    "Inspect",    "Schedule field inspection. Cross-check billing records."],
    ["80 – 100",   "Immediate",  "Immediate field intervention. Potential active theft in progress."],
]
story.append(table(label_data, [3*cm, 3*cm, 10.5*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 6. GSI SIGNALS
# ══════════════════════════════════════════════════════════════════
story += section_header("6", "The 8 GSI Signals — Grid Stress Index")
story.append(p(
    "The GSI (Grid Stress Index) quantifies operational risk at the distribution transformer level. "
    "It combines eight physical and temporal signals to produce a stress score from 0 to 100. "
    "Unlike CASS (which is per meter), GSI is computed once per distribution transformer per "
    "time step using both measured telemetry and Bi-LSTM demand forecasts."
))
story.append(sp(6))

gsi_data = [
    ["Signal", "Weight", "Formula / Logic", "What It Measures"],
    ["Load Quantile",         "0.30",
     "q95_forecast_kWh / (capacity_kVA x power_factor), clipped to [0,1]",
     "The Q95 forecast load as a fraction of rated capacity. Values near 1.0 mean the transformer is near or above rated load — highest risk signal."],
    ["Temperature Derating",  "0.15",
     "clip( (temperature_C - 32) x 0.05,  0,  1 )",
     "Thermal derating of transformer capacity above 32 degrees C. Every degree above 32 reduces effective capacity by 5%, increasing stress risk."],
    ["Power Factor Penalty",  "0.15",
     "clip( (0.9 - power_factor) / 0.9,  0,  1 )",
     "Reactive power burden. Low power factor means the transformer carries more apparent current for the same real power — increasing copper losses and heating."],
    ["Thermal Soak",          "0.10",
     "Exponential moving sum of hours above 80% load with time constant 4 hours",
     "Accumulated thermal stress from sustained heavy loading. A transformer that has been above 80% for several hours retains heat even after load drops."],
    ["EV Load Risk",          "0.10",
     "ev_density x evening_peak_multiplier(hour)",
     "Risk from EV charging demand concentrating in the 18:00-22:00 evening peak. Higher EV density in the area increases this signal during evening hours."],
    ["Transformer Age",       "0.10",
     "log(1 + age_years) / log(1 + 25)",
     "Aging degradation factor. Older transformers have reduced insulation integrity and higher failure probability under load. Normalised to a 25-year reference."],
    ["Calendar Signal",       "0.05",
     "Peak hour factor x seasonal factor x event multiplier",
     "Composite time-of-use factor capturing peak hours (morning/evening), high-consumption months, and holiday/event periods when demand spikes are expected."],
    ["PV Duck Curve",         "0.05",
     "clip( (feeder_load - baseline_load) / baseline_load,  0,  1 ) x solar_factor",
     "Ramp-up stress from solar generation drop-off at sunset. The 'duck curve' describes the rapid increase in conventional generation needed as solar drops."],
]
story.append(table(gsi_data, [3.2*cm, 1.5*cm, 5.5*cm, 6.3*cm]))
story.append(sp(8))

story += sub_header("6.1  GSI Scoring Formula")
story.append(formula("GSI = sum( weight[k] x signal[k] ) x u_tconf x mape_scale x 100"))
story.append(formula("u_tconf   = clip( 1 - PINAW,  0,  1 )   [forecast uncertainty correction]"))
story.append(formula("mape_scale = clip( 1 - MAPE,   0,  1 )   [forecast accuracy correction]"))
story.append(sp(4))
story.append(p(
    "The two correction factors adjust GSI downward when the demand forecast is uncertain "
    "(wide prediction intervals) or inaccurate (high MAPE), reflecting reduced confidence in the "
    "load quantile signal."
))
story.append(sp(6))

story += sub_header("6.2  GSI Label Thresholds")
gsi_label = [
    ["GSI Range", "Label",    "Recommended Action"],
    ["0 – 34",    "Stable",   "Normal operation. No intervention needed."],
    ["35 – 54",   "Caution",  "Monitor load trend. Check for upcoming events or peak periods."],
    ["55 – 74",   "Stressed", "Prepare load-shedding contingency. Alert grid operations team."],
    ["75 – 100",  "Critical", "Immediate grid operator notification. Risk of transformer failure."],
]
story.append(table(gsi_label, [3*cm, 3*cm, 10.5*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 7. ML MODELS
# ══════════════════════════════════════════════════════════════════
story += section_header("7", "Machine Learning Models")

story += sub_header("7.1  XGBoost Theft Detector")
story.append(p(
    "The theft detector is an XGBoost gradient-boosted tree binary classifier trained to distinguish "
    "theft meters (label=1) from honest meters (label=0). Because electricity theft is rare in "
    "practice, the dataset is highly imbalanced; the model compensates with a dynamically computed "
    "<b>scale_pos_weight</b>."
))
story.append(sp(4))

xgb_data = [
    ["Parameter",           "Value",     "Purpose"],
    ["n_estimators",        "500",       "Maximum number of boosting rounds"],
    ["max_depth",           "6",         "Maximum tree depth — controls model complexity"],
    ["learning_rate",       "0.05",      "Step size shrinkage to prevent overfitting"],
    ["subsample",           "0.8",       "Fraction of rows sampled per tree — reduces variance"],
    ["colsample_bytree",    "0.8",       "Fraction of features sampled per tree"],
    ["scale_pos_weight",    "~19x",      "Computed as count(negative) / count(positive) — corrects class imbalance"],
    ["eval_metric",         "aucpr",     "Optimises PR-AUC — more appropriate than ROC-AUC for imbalanced data"],
    ["early_stopping",      "30 rounds", "Stops training when validation PR-AUC stops improving"],
    ["random_state",        "42",        "Reproducibility seed"],
]
story.append(table(xgb_data, [4.5*cm, 3*cm, 9*cm]))
story.append(sp(6))

story += sub_header("7.1.1  Input Features (14 total)")
feat_data = [
    ["#", "Feature Name",        "Source",               "Description"],
    ["1",  "dtw_divergence",     "DTW clustering",       "Euclidean distance of meter profile from cluster centroid"],
    ["2",  "voltage_stability",  "Measured voltage",     "Std dev of 3-phase voltages normalised by 0.15"],
    ["3",  "billing_ratio",      "Billing data",         "|1 - billed/actual| consumption ratio"],
    ["4",  "entropy",            "kWh series",           "1 - normalised Shannon entropy of 20-bin histogram"],
    ["5",  "night_load_anomaly", "kWh + timestamps",     "Night-to-day consumption ratio anomaly"],
    ["6",  "dt_balance_error",   "DT aggregates",        "Feeder vs. meter sum energy balance error"],
    ["7",  "repeat_anomaly",     "kWh rolling windows",  "Fraction of windows with Z-score outlier > 3.0"],
    ["8",  "power_factor_mean",  "Measured",             "Mean power factor across all readings for this meter"],
    ["9",  "kwh_mean_7d",        "Last 672 readings",    "Mean kWh over the most recent 7-day window"],
    ["10", "kwh_std_7d",         "Last 672 readings",    "Std deviation of kWh over the most recent 7-day window"],
    ["11", "kwh_trend_slope",    "Linear regression",    "Slope of linear fit to the full kWh time series (polyfit)"],
    ["12", "hour_of_day",        "Last timestamp",       "Hour (0-23) of the final reading for temporal context"],
    ["13", "day_of_week",        "Last timestamp",       "Day of week (0=Monday, 6=Sunday)"],
    ["14", "month",              "Last timestamp",       "Calendar month (1-12) for seasonal context"],
]
story.append(table(feat_data, [0.8*cm, 3.8*cm, 3.5*cm, 8.4*cm]))
story.append(sp(6))

story += sub_header("7.2  Bi-LSTM Demand Forecaster")
story.append(p(
    "The demand forecaster is a bidirectional LSTM (Long Short-Term Memory) neural network that "
    "predicts the distribution of future kWh demand at the transformer level. It outputs <b>two "
    "quantiles</b> — Q5 and Q95 — forming a prediction interval used by the GSI load quantile signal "
    "and displayed in the frontend demand chart."
))
story.append(sp(4))

lstm_data = [
    ["Parameter",          "Value",              "Purpose"],
    ["Architecture",       "Bidirectional LSTM", "Captures both forward and backward temporal dependencies"],
    ["hidden_size",        "128",                "LSTM hidden state dimension per direction (256 total)"],
    ["num_layers",         "2",                  "Stacked LSTM layers with dropout between them"],
    ["dropout",            "0.2",                "20% dropout between layers to reduce overfitting"],
    ["quantiles",          "[0.05, 0.95]",        "Q5 and Q95 output heads — defines the prediction interval"],
    ["lookback_window",    "24 steps",            "Hours of history fed into each sequence (24-hour lookback)"],
    ["epochs",             "100",                "Maximum training epochs"],
    ["batch_size",         "64",                 "Mini-batch size for stochastic gradient descent"],
    ["learning_rate",      "0.001",              "Adam optimiser initial learning rate"],
    ["patience",           "10",                 "Early stopping: halt if validation loss doesn't improve for 10 epochs"],
    ["Loss function",      "Quantile loss",      "Pinball loss computed separately for Q5 and Q95 then summed"],
    ["Scheduler",          "ReduceLROnPlateau",  "Halves learning rate when validation loss plateaus (patience=5)"],
    ["Normalisation",      "StandardScaler",     "All 13 input features scaled to zero mean, unit variance"],
    ["Saved artefacts",    "2 files",            "models/demand_forecaster.pt (weights) + models/demand_scaler.pkl (scaler)"],
]
story.append(table(lstm_data, [4*cm, 4*cm, 8.5*cm]))
story.append(sp(6))

story += sub_header("7.2.1  LSTM Input Features (13 total)")
lstm_feat = [
    ["Feature",          "Description"],
    ["kwh",              "Aggregated kWh across all meters under this DT (target variable for training)"],
    ["temperature_c",    "Ambient temperature in Celsius — diurnal 24-32 degree C sinusoidal pattern"],
    ["power_factor",     "Mean power factor of all meters on this DT"],
    ["hour_sin",         "sin(2*pi*hour/24) — cyclical encoding of hour of day"],
    ["hour_cos",         "cos(2*pi*hour/24) — cyclical encoding of hour of day"],
    ["day_sin",          "sin(2*pi*dayofweek/7) — cyclical encoding of day of week"],
    ["day_cos",          "cos(2*pi*dayofweek/7) — cyclical encoding of day of week"],
    ["month_sin",        "sin(2*pi*month/12) — cyclical encoding of calendar month"],
    ["month_cos",        "cos(2*pi*month/12) — cyclical encoding of calendar month"],
    ["is_weekend",       "1.0 if Saturday or Sunday, else 0.0"],
    ["is_holiday",       "1.0 if the date is a recognised Indian public holiday, else 0.0"],
    ["ev_density",       "Electric vehicle penetration proxy — currently 0.1 (configurable)"],
    ["solar_irradiance", "Solar irradiance W/m^2 — diurnal sinusoidal profile peaking at noon"],
]
story.append(table(lstm_feat, [4*cm, 12.5*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 8. GSS SCORING
# ══════════════════════════════════════════════════════════════════
story += section_header("8", "GSS Composite Scoring — Grid Sentinel Score")
story.append(p(
    "The GSS (Grid Sentinel Score) is a single composite metric that captures overall system "
    "performance across theft detection quality, forecast accuracy, economic efficiency, robustness, "
    "calibration, energy balance, and detection speed. Two variants are computed: <b>GSS Core</b> "
    "(4 components) and <b>GSS Final</b> (all 7 components)."
))
story.append(sp(6))

story += sub_header("8.1  Component Sub-scores")
comp_data = [
    ["Sub-score",  "Formula",                                         "Measures"],
    ["S_CASS",     "MCC x (1 - FPR)",                                "Theft detection quality: balanced accuracy adjusted for false alarm rate"],
    ["S_GSI",      "(1 - MAPE) x PICP x (1 - PINAW)",               "Forecast quality: accuracy x interval coverage x interval sharpness"],
    ["S_Econ",     "1 - cost / baseline_cost",                       "Economic efficiency: savings vs. a baseline of no detection"],
    ["S_Robust",   "1 - |F1_clean - F1_noisy| / F1_clean",          "Stability under Gaussian noise injection (std=0.05)"],
    ["S_Delay",    "exp(-latency_days / 7)",                          "Detection speed: exponential decay from theft start to detection"],
    ["S_Energy",   "1 - |meter_kWh_total - feeder_kWh_total| / feeder_kWh_total", "Energy accounting consistency across meters and feeders"],
    ["S_Calib",    "1 - ECE",                                        "Probability calibration: reliability of predicted theft probabilities"],
]
story.append(table(comp_data, [2.5*cm, 6.5*cm, 7.5*cm]))
story.append(sp(8))

story += sub_header("8.2  Aggregation Formulas")
story.append(formula("GSS_Core  = w_CASS(0.35) x S_CASS  +  w_GSI(0.20) x S_GSI"))
story.append(formula("          + w_Econ(0.30) x S_Econ  +  w_Robust(0.15) x S_Robust"))
story.append(sp(4))
story.append(formula("GSS_Final = w_CASS(0.30) x S_CASS  +  w_GSI(0.15) x S_GSI"))
story.append(formula("          + w_Econ(0.25) x S_Econ  +  w_Robust(0.10) x S_Robust"))
story.append(formula("          + w_Delay(0.08) x S_Delay +  w_Energy(0.07) x S_Energy"))
story.append(formula("          + w_Calib(0.05) x S_Calib"))
story.append(sp(4))
story.append(p(
    "<b>Pareto Constraint Penalty:</b> If any of FPR > 0.02, Recall < 0.85, or MAPE > 0.07 "
    "is violated, both GSS Core and GSS Final are multiplied by 0.5. This ensures the composite "
    "score cannot be high when fundamental operational constraints are breached."
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 9. ECONOMIC MODEL
# ══════════════════════════════════════════════════════════════════
story += section_header("9", "Economic Model")
story.append(p(
    "GridSentinel quantifies the economic impact of its detection decisions in Indian Rupees. "
    "The model distinguishes between the cost of false investigations (false positives) and the "
    "cost of missed thefts (false negatives)."
))
story.append(sp(6))

econ_data = [
    ["Parameter",                          "Value (INR)", "Description"],
    ["FP investigation cost",              "Rs. 8,500",   "Cost per field team dispatch to investigate a flagged meter that turns out to be honest"],
    ["FN monthly theft loss",              "Rs. 5,500",   "Monthly revenue loss per undetected theft case (stolen energy + administrative cost)"],
    ["Default theft duration",             "3 months",    "Assumed duration a theft goes undetected if flagged as FN"],
    ["Total FN cost per case",             "Rs. 16,500",  "5,500 x 3 months — total loss for one missed theft"],
    ["Baseline cost (no detection)",       "Variable",    "Cost assuming ALL theft cases remain undetected (FN_count + TP_count) x 16,500"],
    ["System cost",                        "Variable",    "FP_count x 8,500 + FN_count x 16,500"],
    ["Economic benefit (S_Econ)",          "0.0 to 1.0",  "1 - system_cost / baseline_cost — fraction of potential loss avoided"],
]
story.append(table(econ_data, [5*cm, 3*cm, 8.5*cm]))
story.append(sp(6))

story += sub_header("9.1  Frontend Economic Summary Panel")
for txt in [
    "<b>Critical Meters</b> (CASS > 80): Count of meters flagged for immediate intervention.",
    "<b>Watch List</b> (CASS 50–80): Count of meters under monitoring.",
    "<b>Estimated Revenue Protection</b>: critical_count x 5,500 x 3 — revenue recoverable if all critical meters are confirmed theft.",
    "<b>Investigation Cost</b>: watch_count x 8,500 — estimated cost of dispatching teams to all watch-list meters.",
    "<b>Net Benefit</b>: Revenue protection minus investigation cost.",
]:
    story.append(bullet(txt))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 10. FRONTEND UI
# ══════════════════════════════════════════════════════════════════
story += section_header("10", "Frontend UI — Tabs & Components")
story.append(p(
    "The GridSentinel frontend is a React 19 single-page application served by Vite on "
    "http://localhost:5173 during development. All API calls use the /api prefix proxied by Vite "
    "to the FastAPI backend on port 8000. Authentication uses JWT bearer tokens stored in "
    "localStorage."
))
story.append(sp(6))

story += sub_header("10.1  Login Page")
for txt in [
    "<b>Username field</b>: Accepts the BESCOM operator username.",
    "<b>Password field</b>: Masked input for the bcrypt-hashed password.",
    "<b>Default credentials</b>: username = <i>admin</i>, password = <i>bescom2026</i> (seeded at startup).",
    "On successful login, a JWT bearer token (valid 60 minutes) is stored in localStorage as <i>gs_token</i>.",
    "401 responses from any subsequent API call trigger automatic logout and redirect to login.",
]:
    story.append(bullet(txt))
story.append(sp(6))

story += sub_header("10.2  Main Dashboard — Left Panel")
story.append(p("<b>Live Score Feed</b> (Server-Sent Events, updates every 10 seconds):"))
sse_data = [
    ["Element",     "Description"],
    ["meter_id",    "Meter identifier string (e.g. MTR_000042)"],
    ["risk score",  "CASS score 0-100 with small random noise (+/- 2) for live animation"],
    ["status badge","Colour-coded: normal (green), moderate (yellow), high (orange), critical (red)"],
]
story.append(table(sse_data, [4*cm, 12.5*cm]))
story.append(sp(6))

story.append(p("<b>Anomalous Cluster Feed Table</b> (top-8 meters by CASS score):"))
clust_data = [
    ["Column",      "Description"],
    ["Feeder",      "Meter ID — click to load DTW chart and SHAP breakdown for this meter"],
    ["Risk Score",  "CASS score (0-100). Colour coded by severity."],
    ["Logic",       "'DTW + CASS' when risk >= 65, else 'CASS' only"],
    ["Status",      "Text label: normal / moderate / high / critical"],
]
story.append(table(clust_data, [4*cm, 12.5*cm]))
story.append(sp(6))

story += sub_header("10.3  DTW Divergence Chart (Centre Panel)")
for txt in [
    "<b>Triggered by</b>: clicking any row in the Anomalous Cluster Feed, or querying a meter ID in the AI chat.",
    "<b>X-axis</b>: Hour of day, 0 to 23.",
    "<b>Y-axis</b>: Average kWh consumption.",
    "<b>Peer Average line</b>: Mean hourly consumption of all other meters connected to the same distribution transformer.",
    "<b>Target line</b>: Hourly profile of the selected meter.",
    "<b>Interpretation</b>: Large divergence between Target and Peer Average at specific hours indicates suspicious behaviour. A thief using a bypass shows lower consumption than peers at high-usage times.",
]:
    story.append(bullet(txt))
story.append(sp(6))

story += sub_header("10.4  SHAP Feature Importance Panel")
for txt in [
    "<b>Triggered by</b>: same selection as DTW chart.",
    "Displays a horizontal bar chart of all 7 CASS signal contributions: <i>value x weight</i>.",
    "Bars are sorted by absolute contribution — the most influential signal appears at the top.",
    "Contribution values are computed using the actual meter's signals from the stored meter_df.",
    "<b>DTW Divergence</b> and <b>DT Balance Error</b>: computed against population mean and full DT aggregate respectively.",
    "<b>Purpose</b>: Explains to the operator exactly which signals drove the suspicion score, enabling targeted field investigation.",
]:
    story.append(bullet(txt))
story.append(sp(6))

story += sub_header("10.5  Prediction / Demand Forecast Tab")
for txt in [
    "<b>Transformer selector</b>: Dropdown listing all DT IDs loaded from the CSV (DT_1 through DT_N).",
    "<b>GSI Score badge</b>: Large coloured number showing the current Grid Stress Index (0-100) for the selected transformer.",
    "<b>Recommended Action</b>: Text derived from GSI label — e.g. 'Immediate Grid Operator Notification' for Critical.",
    "<b>Stress Factors table</b>: Lists each contributing GSI signal with its computed value.",
]:
    story.append(bullet(txt))
story.append(sp(4))
story.append(p("<b>Demand Chart</b> — 4 lines per hour (0-23):"))
chart_data = [
    ["Line",       "Colour",  "Description"],
    ["Actual",     "Blue",    "Measured feeder kWh averaged by hour from the stored dt_df"],
    ["Predicted",  "Green",   "Q50 median forecast from Bi-LSTM (middle quantile if 3 quantiles, Q95 if 2)"],
    ["Q5 Lower",   "Orange",  "5th percentile forecast — lower bound of the prediction interval"],
    ["Q95 Upper",  "Red",     "95th percentile forecast — upper bound of the prediction interval"],
]
story.append(table(chart_data, [3.5*cm, 2.5*cm, 10.5*cm]))
story.append(sp(6))

story += sub_header("10.6  AI Chat Panel")
for txt in [
    "<b>Input</b>: Natural language text field — accepts free-form queries in English.",
    "<b>Backend</b>: POST /api/chat — tries Ollama (llama3.2) first; falls back to keyword matching if Ollama unavailable.",
    "<b>System prompt</b>: GridSentinel is briefed as a BESCOM grid expert; responses are limited to 2-3 sentences with a recommended action.",
    "<b>Meter query</b> (e.g. 'MTR_000001 anomaly'): Returns CASS score + automatically triggers DTW chart for that meter.",
    "<b>Transformer query</b> (e.g. 'DT_3 stress'): Returns GSI commentary + triggers demand forecast panel for that DT.",
    "<b>Theft/anomaly keywords</b>: Returns top-3 highest CASS meters from live store.",
    "<b>Grid/GSI keywords</b>: Explains the 8 GSI signal composition.",
]:
    story.append(bullet(txt))
story.append(sp(6))

story += sub_header("10.7  Audit Log Tab")
audit_data = [
    ["Column",      "Description"],
    ["Timestamp",   "UTC datetime of the action"],
    ["Action",      "Action type: anomaly_detail, demand_detail, etc."],
    ["Target ID",   "The meter or transformer ID that was queried"],
    ["Target Type", "'meter' or 'transformer'"],
    ["Details",     "Human-readable description of the operation"],
]
story.append(table(audit_data, [3.5*cm, 13*cm]))
story.append(p(
    "The audit log auto-refreshes every 30 seconds. It records every call to /api/anomaly/{id} "
    "and /api/demand/{id}, providing a full operator activity trail. Requires authentication."
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 11. API ENDPOINTS
# ══════════════════════════════════════════════════════════════════
story += section_header("11", "API Endpoints Reference")
story.append(p(
    "The FastAPI backend exposes the following endpoints on port 8000. All /api/* endpoints "
    "use relative paths proxied by Vite during development. CORS is pre-configured for "
    "localhost:5173."
))
story.append(sp(6))

api_data = [
    ["Method", "Endpoint",                 "Auth", "Description"],
    ["GET",    "/health",                  "No",   "Returns API version and status string"],
    ["POST",   "/api/auth/login",          "No",   "Verifies credentials, returns JWT bearer token + username"],
    ["POST",   "/api/auth/register",       "No",   "Creates a new user account with bcrypt-hashed password"],
    ["POST",   "/api/auth/token",          "No",   "OAuth2 alias for /api/auth/login"],
    ["GET",    "/api/anomaly/clusters",    "No",   "Returns top-N meters ranked by CASS score. Default N=8."],
    ["GET",    "/api/anomaly/{meter_id}",  "No",   "Returns hourly DTW series (peer vs target) and CASS score for one meter"],
    ["GET",    "/api/shap/{meter_id}",     "No",   "Returns 7 CASS signal values and weighted contributions for one meter"],
    ["GET",    "/api/demand/{dt_id}",      "No",   "Returns hourly actual vs forecast demand, GSI score, action and stress factors for one DT"],
    ["GET",    "/api/economic/summary",    "No",   "Returns critical/watch counts and INR economic KPIs"],
    ["GET",    "/api/map/zones",           "No",   "Returns per-DT risk summary for the zone map view"],
    ["GET",    "/api/stream/scores",       "No",   "Server-Sent Events stream: top-3 meter scores + GSI every 10 seconds"],
    ["POST",   "/api/chat",               "No",   "Natural language query — Ollama LLM with keyword fallback"],
    ["GET",    "/api/audit/log",           "YES",  "Returns last 50 audit log entries. JWT required."],
    ["GET",    "/api/report/summary",      "No",   "Generates and streams a PDF intelligence report"],
    ["POST",   "/score/meter",             "No",   "Scores a single meter from supplied records via CASS formula"],
    ["POST",   "/score/transformer",       "No",   "Scores a single transformer from supplied DT records via GSI formula"],
    ["POST",   "/evaluate",               "No",   "Runs the full 18-step pipeline on supplied meter and DT records"],
]
story.append(table(api_data, [1.5*cm, 5.5*cm, 1.5*cm, 8*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 12. CLASSIFICATION METRICS
# ══════════════════════════════════════════════════════════════════
story += section_header("12", "Classification Metrics Explained")
story.append(p(
    "The following metrics are computed by evaluate_classification() after the TheftDetector "
    "predicts on the full feature matrix. They appear in the pipeline summary output and are "
    "used to compute S_CASS for the GSS score."
))
story.append(sp(6))

clf_data = [
    ["Metric",    "Full Name",                          "Formula",                                    "Interpretation"],
    ["Precision", "Precision",                           "TP / (TP + FP)",                             "Of all meters flagged as theft, what fraction truly are. High precision = few wasted field investigations."],
    ["Recall",    "Recall / Sensitivity",               "TP / (TP + FN)",                             "Of all actual theft meters, what fraction were caught. High recall = fewer missed thefts. Constraint: >= 0.85"],
    ["F1",        "F1 Score",                            "2 x Precision x Recall / (Precision+Recall)","Harmonic mean — balances precision and recall. Primary metric for imbalanced datasets."],
    ["FPR",       "False Positive Rate",                "FP / (FP + TN)",                             "Fraction of honest meters incorrectly flagged. Constraint: <= 0.02 (at most 2% false alarms)."],
    ["MCC",       "Matthews Correlation Coefficient",   "(TP*TN - FP*FN) / sqrt(...)",                "Balanced metric for imbalanced classes. Ranges -1 to +1. Only metric that uses all 4 confusion matrix cells equally."],
    ["ROC-AUC",   "Area Under ROC Curve",               "Integral of TPR vs FPR curve",               "Probability that the model ranks a random theft meter above a random honest meter. 0.5 = random, 1.0 = perfect."],
    ["PR-AUC",    "Area Under PR Curve",                "Integral of Precision vs Recall curve",       "More informative than ROC-AUC for heavily imbalanced data (rare theft). Used as XGBoost eval_metric."],
    ["ECE",       "Expected Calibration Error",         "Mean |accuracy - confidence| per bin",        "Measures how well predicted probabilities match true frequencies in 10 reliability bins. Used to compute S_Calib."],
]
story.append(table(clf_data, [2*cm, 4.5*cm, 3.5*cm, 6.5*cm]))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# 13. FORECAST METRICS
# ══════════════════════════════════════════════════════════════════
story += section_header("13", "Forecast Metrics Explained")
story.append(p(
    "Forecast metrics are computed by evaluate_forecast() comparing Bi-LSTM Q5/Q95 predictions "
    "against actual feeder kWh values. They feed into S_GSI for the GSS score and are reported "
    "in the pipeline summary. Constraint: MAPE <= 0.07."
))
story.append(sp(6))

fc_data = [
    ["Metric", "Full Name",                                           "Formula",                                   "Interpretation"],
    ["MAPE",   "Mean Absolute Percentage Error",                      "mean(|actual - pred| / actual) x 100%",     "Average percentage forecast error. 7% MAPE means forecasts are off by 7% on average. Constraint: <= 7%."],
    ["RMSE",   "Root Mean Squared Error",                             "sqrt(mean((actual - pred)^2))",             "Penalises large errors more than MAPE. Measured in kWh. Useful for detecting infrequent large forecast failures."],
    ["PICP",   "Prediction Interval Coverage Probability",            "mean(Q5 <= actual <= Q95)",                 "Fraction of actual values that fall within the Q5-Q95 prediction band. Ideal value: 0.90 (90% coverage)."],
    ["PINAW",  "Prediction Interval Normalised Average Width",        "mean(Q95 - Q5) / range(actual)",            "Width of the prediction interval relative to the data range. Lower = sharper (more confident) intervals. Trade-off vs PICP."],
]
story.append(table(fc_data, [2*cm, 5.5*cm, 4.5*cm, 4.5*cm]))
story.append(sp(10))

story += sub_header("13.1  Pareto Constraint Summary")
pareto_data = [
    ["Constraint", "Threshold", "Metric",  "Consequence if Violated"],
    ["Max FPR",    "<= 0.02",   "FPR",     "GSS Core and Final multiplied by 0.5 — system penalised for excessive false alarms"],
    ["Min Recall", ">= 0.85",   "Recall",  "GSS Core and Final multiplied by 0.5 — system penalised for missing too many thefts"],
    ["Max MAPE",   "<= 0.07",   "MAPE",    "GSS Core and Final multiplied by 0.5 — system penalised for poor demand forecasting"],
]
story.append(table(pareto_data, [3*cm, 3*cm, 2.5*cm, 8*cm]))
story.append(sp(12))
story.append(hr())
story.append(sp(6))
story.append(p(
    "This document was automatically generated by GridSentinel AI v2.0 on "
    f"{datetime.now().strftime('%d %B %Y at %H:%M')}. "
    "All technical specifications reflect the implementation in the GridSentinel codebase "
    "as reviewed and documented during the BESCOM Hackathon.",
    "caption"
))

# ── Build PDF ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2*cm,
    rightMargin=2*cm,
    topMargin=2*cm,
    bottomMargin=1.8*cm,
    title="GridSentinel AI v2.0 — Technical Documentation",
    author="GridSentinel AI",
    subject="BESCOM Smart Grid Intelligence Platform",
)
doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
print(f"Report saved to {OUTPUT}")
