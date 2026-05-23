"""Page Streamlit de monitoring des prédictions et du run de drift."""

import json
import logging
import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Monitoring", page_icon="📊", layout="wide")


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def fetch_json(path: str) -> Any:
    """Récupère un endpoint JSON de l'API et retourne la réponse décodée."""
    url = f"{API_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        logger.error("Erreur API %s : %s", url, exc)
        st.error(f"❌ Impossible de joindre l'API : {url}")
        return None


def load_monitoring_data() -> dict[str, Any]:
    """Charge les prédictions et le résultat du run de drift."""
    predictions = fetch_json("/predictions")
    drift_report = fetch_json("/drift")

    return {
        "predictions": predictions or [],
        "drift_report": drift_report,
    }


def serialize_metric_value(value: Any) -> str:
    """Convertit les valeurs de métrique en texte lisible et Arrow-compatible."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return "null"
    return str(value)


def build_metrics_dataframe(metrics: list[dict[str, Any]]) -> pd.DataFrame:
    """Construit un DataFrame exploitable pour l'affichage des métriques de drift."""
    normalized_metrics = [
        {
            "metric_name": metric.get("metric_name", ""),
            "value": serialize_metric_value(metric.get("value")),
        }
        for metric in metrics
    ]
    return pd.DataFrame(normalized_metrics)


if "monitoring_data" not in st.session_state:
    st.session_state.monitoring_data = load_monitoring_data()


st.title("📈 Monitoring des prédictions")
st.markdown(
    "Cette page affiche les prédictions les plus récentes, ainsi que le résultat du dernier run de drift."
)

refresh_clicked = st.button("🔄 Actualiser le monitoring", type="primary")
if refresh_clicked:
    st.session_state.monitoring_data = load_monitoring_data()

monitoring_data = st.session_state.monitoring_data
predictions = monitoring_data.get("predictions", [])

drft_report = monitoring_data.get("drift_report")

if not predictions:
    st.warning("⚠️ Aucune prédiction n’est disponible pour le moment.")
else:
    predictions_df = pd.DataFrame(predictions)
    predictions_df["created_at"] = pd.to_datetime(
        predictions_df["created_at"], errors="coerce"
    )
    predictions_df = predictions_df.sort_values("created_at", ascending=False)
    predictions_df["created_at"] = predictions_df["created_at"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    st.subheader("🗂️ Prédictions")
    st.dataframe(predictions_df, width="stretch", hide_index=True)

if drft_report is None:
    st.warning("⚠️ Le rapport de drift n’est pas disponible pour le moment.")
else:
    st.subheader("📉 Résultat du run de drift")

    metrics = drft_report.get("metrics", [])
    tests = drft_report.get("tests", [])

    drift_metrics = {
        metric.get("metric_name"): metric.get("value") for metric in metrics
    }

    drift_summary = {
        "DriftedColumnsCount": drift_metrics.get(
            "DriftedColumnsCount(drift_share=0.5)", {}
        ),
        "Failed tests": sum(1 for test in tests if test.get("status") == "FAIL"),
        "Total tests": len(tests),
    }

    cols = st.columns(3)
    with cols[0]:
        st.metric(
            "Colonnes driftées",
            drift_summary["DriftedColumnsCount"].get("count", 0)
            if isinstance(drift_summary["DriftedColumnsCount"], dict)
            else 0,
        )
    with cols[1]:
        st.metric(
            "Part de colonnes driftées",
            drift_summary["DriftedColumnsCount"].get("share", 0)
            if isinstance(drift_summary["DriftedColumnsCount"], dict)
            else 0,
        )
    with cols[2]:
        st.metric("Tests en échec", drift_summary["Failed tests"])

    st.markdown("### 📊 Métriques de drift")
    metrics_df = build_metrics_dataframe(metrics)
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    st.markdown("### ✅ / ❌ Résultat des tests")
    tests_df = pd.DataFrame(tests)
    tests_df = tests_df[["name", "status", "description"]]
    tests_df["status"] = tests_df["status"].replace(
        {"SUCCESS": "✅ SUCCESS", "FAIL": "❌ FAIL"}
    )
    st.dataframe(tests_df, width="stretch", hide_index=True)
