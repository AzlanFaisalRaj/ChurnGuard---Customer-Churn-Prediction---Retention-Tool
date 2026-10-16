"""ChurnGuard - Gradio app: single customer, CSV upload, business report. Run: python app.py"""
import html
import os
import tempfile
import traceback

import gradio as gr
import numpy as np
import pandas as pd

from src import config as cfg
from src import model_store
from src.business import customer_value, expected_impact, load_test_impact
from src.pipeline import FEATURES
from src.predict import predict_dataframe

MAX_ROWS = 5000
CAT = cfg.ARTIFACTS["categorical_values"]
COLORS = {"Low": "#16a34a", "Medium": "#d97706", "High": "#dc2626"}

ACTIONS = {
    "Low": "No offer needed. Keep monitoring.",
    "Medium": "Send a retention offer: the expected saving is higher than the offer cost.",
    "High": "Priority customer: contact them first with a retention offer.",
}

# 19 inputs, in a fixed order (the examples below follow the same order)
ORDER = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
         "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
         "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"]
HIGH_EXAMPLE = ["Female", "No", "No", "No", 2, "Yes", "No", "Fiber optic", "No", "No", "No", "No", "No", "No",
                "Month-to-month", "Yes", "Electronic check", 85.0, None]
LOW_EXAMPLE = ["Male", "No", "Yes", "Yes", 60, "Yes", "Yes", "DSL", "Yes", "Yes", "Yes", "Yes", "No", "No",
               "Two year", "No", "Bank transfer (automatic)", 70.0, None]


def money(x):
    return f"{x:,.0f}"


def render_card(p, level, reasons, offer, loss, warnings):
    color = COLORS[level]
    heading = ("Main factors behind this score (top 3)" if level == "Low"
               else "Why is this customer at risk? Top 3 reasons")
    items = "".join(
        f"<li style='margin:6px 0'><b>{html.escape(r['arrow'])} {html.escape(r['text'])}</b> "
        f"<span style='opacity:.65'>({r['direction']}, impact {r['impact']:+.2f})</span></li>" for r in reasons)
    offer_html = (f"<p><b>Suggested offer:</b> {html.escape(offer)}</p>" if offer != "-" else "")
    warn_html = "".join(f"<p style='color:#b45309'>Note: {html.escape(w)}</p>" for w in warnings)
    return f"""
<div style="border:2px solid {color};border-radius:14px;padding:18px;line-height:1.5">
  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
    <div style="font-size:44px;font-weight:800;color:{color}">{p:.1%}</div>
    <div>
      <div style="display:inline-block;background:{color};color:#fff;padding:3px 12px;border-radius:999px;font-weight:700">{level} risk</div>
      <div style="opacity:.75;margin-top:4px">probability that this customer churns</div>
    </div>
  </div>
  <p style="margin-top:14px"><b>Decision:</b> {ACTIONS[level]}</p>
  <p><b>{heading}</b></p>
  <ul style="margin-top:0;padding-left:20px">{items}</ul>
  {offer_html}
  <p><b>Expected revenue at risk:</b> {money(loss)} (probability x {cfg.VALUE_MONTHS} months of charges)</p>
  <p style="font-size:12px;opacity:.65">Offer threshold = {cfg.THRESHOLD:.2f} (chosen from business cost). ↑ raises the risk, ↓ lowers it.
  Cost assumptions: offer cost {cfg.OFFER_COST}, offer success {cfg.OFFER_SUCCESS:.0%}.</p>
  {warn_html}
</div>"""


def predict_single(*vals):
    row = dict(zip(ORDER, vals))
    if row["MonthlyCharges"] is None or row["MonthlyCharges"] < 0:
        return "<p style='color:#dc2626'><b>Please enter a valid monthly charge.</b></p>"
    row["tenure"] = 0 if row["tenure"] is None else row["tenure"]
    row["TotalCharges"] = np.nan if row["TotalCharges"] in (None, "") else row["TotalCharges"]
    try:
        res, clean, reasons, warns = predict_dataframe(pd.DataFrame([row]))
    except ValueError as e:
        return f"<p style='color:#dc2626'><b>{html.escape(str(e))}</b></p>"
    except Exception as e:      # never hide the real error: print it in the terminal and show the type in the app
        traceback.print_exc()
        return (f"<p style='color:#dc2626'><b>Prediction failed: {html.escape(type(e).__name__)}: "
                f"{html.escape(str(e)[:300])}</b></p>")
    r = res.iloc[0]
    loss = float(r["expected_revenue_at_risk"])
    return render_card(float(r["churn_probability"]), r["risk_level"], reasons[0], r["suggested_offer"], loss, warns)


def process_df(df_raw):
    notes = []
    if len(df_raw) > MAX_ROWS:
        df_raw = df_raw.head(MAX_ROWS)
        notes.append(f"Only the first {MAX_ROWS} rows were scored (demo limit).")
    res, clean, reasons, warns = predict_dataframe(df_raw)
    res = res.sort_values("churn_probability", ascending=False)
    n = len(res)
    lv = res["risk_level"].value_counts()
    flagged = int((res["send_offer"] == "Yes").sum())
    value = customer_value(clean.loc[res.index, "MonthlyCharges"])
    impact = expected_impact(res["churn_probability"].values, value)
    summary = (
        f"### Results for {n} customers\n"
        f"- **High risk:** {int(lv.get('High', 0))} | **Medium:** {int(lv.get('Medium', 0))} | **Low:** {int(lv.get('Low', 0))}\n"
        f"- **Customers to target (probability >= {cfg.THRESHOLD:.2f}):** {flagged}\n"
        f"- **Expected revenue at risk (all customers):** {money(float(res['expected_revenue_at_risk'].sum()))}\n"
        + "".join(f"- Note: {w}\n" for w in warns + notes))
    path = os.path.join(tempfile.mkdtemp(), "churnguard_predictions.csv")
    res.to_csv(path, index=False)
    return summary, res.head(200), impact, path


def run_csv(file):
    if file is None:
        return "Please upload a CSV file first.", None, None, None
    try:
        return process_df(pd.read_csv(file))
    except ValueError as e:
        return f"**Error:** {e}", None, None, None
    except Exception as e:  # unreadable file
        return f"**Could not read the file:** {e}", None, None, None


def run_sample():
    return process_df(pd.read_csv(cfg.SAMPLE_CSV))


def model_info_md():
    m = cfg.ARTIFACTS.get("test_metrics", {})
    return f"""
### About the model
- **Model source:** {model_store.STATUS}
- **Model:** {cfg.ARTIFACTS.get('model', 'XGBoost')} inside a scikit-learn pipeline (ColumnTransformer + classifier), trained on the IBM Telco Customer Churn data (7,043 customers, 26.5% churn).
- **Main metric:** PR-AUC (not accuracy: a model that says "nobody churns" is already 73.5% accurate). Held-out test set: **PR-AUC {m.get('pr_auc', 0):.3f}**, ROC-AUC {m.get('roc_auc', 0):.3f}.
- **Decision threshold {cfg.THRESHOLD:.2f}** (not 0.5): chosen by minimising business cost. At this threshold: precision {m.get('precision', 0):.2f}, recall {m.get('recall', 0):.2f}.
- **Risk levels:** Low < {cfg.THRESHOLD:.2f} <= Medium < {cfg.HIGH_FROM:.2f} <= High.
- **Reasons:** SHAP values, summed per original column.

### Business assumptions (assumptions, not facts from the data)
Customer value = monthly charges x {cfg.VALUE_MONTHS}; one retention offer costs {cfg.OFFER_COST}; {cfg.OFFER_SUCCESS:.0%} of real churners stay after an offer.
Changing these numbers changes the best threshold a lot, so treat the money figures as estimates.

### Validated business impact (held-out test set, real churn labels)
"""


def build_demo():
    with gr.Blocks(title="ChurnGuard") as demo:
        gr.Markdown("# 🛡️ ChurnGuard\nPredict which telecom customers are likely to churn, **why**, and what to do about it.")
        with gr.Tab("Single customer"):
            with gr.Row():
                with gr.Column(scale=3):
                    comps = {}
                    with gr.Row():
                        comps["gender"] = gr.Dropdown(CAT["gender"], value="Female", label="Gender")
                        comps["SeniorCitizen"] = gr.Dropdown(["No", "Yes"], value="No", label="Senior citizen")
                        comps["Partner"] = gr.Dropdown(CAT["Partner"], value="No", label="Has partner")
                        comps["Dependents"] = gr.Dropdown(CAT["Dependents"], value="No", label="Has dependents")
                    with gr.Row():
                        comps["tenure"] = gr.Slider(0, 72, value=2, step=1, label="Tenure (months)")
                        comps["MonthlyCharges"] = gr.Number(value=85.0, minimum=0, label="Monthly charges")
                        comps["TotalCharges"] = gr.Number(value=None, minimum=0, label="Total charges (empty = estimate)")
                    with gr.Row():
                        comps["Contract"] = gr.Dropdown(CAT["Contract"], value="Month-to-month", label="Contract")
                        comps["PaymentMethod"] = gr.Dropdown(CAT["PaymentMethod"], value="Electronic check", label="Payment method")
                        comps["PaperlessBilling"] = gr.Dropdown(CAT["PaperlessBilling"], value="Yes", label="Paperless billing")
                    with gr.Row():
                        comps["PhoneService"] = gr.Dropdown(CAT["PhoneService"], value="Yes", label="Phone service")
                        comps["MultipleLines"] = gr.Dropdown(CAT["MultipleLines"], value="No", label="Multiple lines")
                        comps["InternetService"] = gr.Dropdown(CAT["InternetService"], value="Fiber optic", label="Internet service")
                    with gr.Row():
                        for c, lab in [("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
                                       ("DeviceProtection", "Device protection")]:
                            comps[c] = gr.Dropdown(CAT[c], value="No", label=lab)
                    with gr.Row():
                        for c, lab in [("TechSupport", "Tech support"), ("StreamingTV", "Streaming TV"),
                                       ("StreamingMovies", "Streaming movies")]:
                            comps[c] = gr.Dropdown(CAT[c], value="No", label=lab)
                    inputs = [comps[c] for c in ORDER]
                    with gr.Row():
                        btn = gr.Button("Predict churn risk", variant="primary")
                        ex_hi = gr.Button("Load high-risk example")
                        ex_lo = gr.Button("Load low-risk example")
                with gr.Column(scale=2):
                    card = gr.HTML("<p style='opacity:.7'>Fill in the customer details and press <b>Predict churn risk</b>.</p>")
            btn.click(predict_single, inputs, card)
            ex_hi.click(lambda: HIGH_EXAMPLE, None, inputs)
            ex_lo.click(lambda: LOW_EXAMPLE, None, inputs)

        with gr.Tab("Upload CSV"):
            gr.Markdown("Upload a CSV with the 19 Telco columns (a `customerID` column is optional, a `Churn` column is ignored). "
                        "Every row gets a probability, risk level, top 3 reasons and a suggested offer.")
            with gr.Row():
                up = gr.File(label="Customers CSV", file_types=[".csv"], type="filepath")
                with gr.Column():
                    go = gr.Button("Score customers", variant="primary")
                    sample = gr.Button("Try with the sample file (25 customers)")
            summary = gr.Markdown()
            table = gr.Dataframe(label="Predictions (highest risk first, first 200 rows shown)", interactive=False, wrap=True)
            impact = gr.Dataframe(label="Expected business impact (from predicted probabilities, using the assumptions above)", interactive=False)
            dl = gr.File(label="Download all predictions (CSV)")
            outs = [summary, table, impact, dl]
            go.click(run_csv, up, outs)
            sample.click(run_sample, None, outs)

        with gr.Tab("Model & business report"):
            gr.Markdown(model_info_md())
            gr.Dataframe(value=load_test_impact().round(2), interactive=False)
            gr.Markdown("Net benefit = revenue saved - offer cost. 'If random' = same number of customers picked at random. "
                        "Top 20% by risk beats random targeting by a wide margin; the exact money values depend on the assumptions above.")
    return demo


model_store.get_model()          # load (or retrain) once at startup so problems show up immediately
demo = build_demo()

if __name__ == "__main__":
    demo.launch()
