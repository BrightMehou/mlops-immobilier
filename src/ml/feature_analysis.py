import logging
from pathlib import Path

import matplotlib.pyplot as plt
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

housing = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    housing.data,
    housing.target,
    test_size=0.2,
    random_state=42,
)

model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.15,
    max_depth=5,
    random_state=42,
)
model.fit(X_train, y_train)

OUTPUT_DIR = Path("data/feature_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Génération des graphiques de dépendance partielle SHAP...")
columns = X_train.columns

for col in columns:
    logger.info("Génération du graphique pour la feature : %s", col)
    fig, ax = shap.partial_dependence_plot(
        col,
        model.predict,
        X_test,
        ice=False,
        model_expected_value=True,
        feature_expected_value=True,
        show=False,
    )
    filepath = OUTPUT_DIR / f"partial_dependence_{col}.png"
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close(fig)

explainer = shap.Explainer(model.predict, X_test)
shap_values = explainer(X_test)

logger.info("Génération du graphique beeswarm")
ax = shap.plots.beeswarm(shap_values, show=False)
filepath = OUTPUT_DIR / "beeswarm_plot.png"
plt.savefig(filepath, bbox_inches="tight", dpi=150)
plt.close()

for col in columns:
    logger.info("Génération du graphique scatter pour la feature : %s", col)
    ax = shap.plots.scatter(shap_values[:, col], color=shap_values[:, col], show=False)
    filepath = OUTPUT_DIR / f"scatter_plot_{col}.png"
    plt.savefig(filepath, bbox_inches="tight", dpi=150)
    plt.close()

logger.info("Génération du graphique bar plot")
shap.plots.bar(shap_values, max_display=8, show=False)
filepath = OUTPUT_DIR / "bar_plot.png"
plt.savefig(filepath, bbox_inches="tight", dpi=150)
plt.close()

logger.info("Génération du graphique bar plot (valeurs absolues maximales)")
shap.plots.bar(shap_values.abs.max(0), max_display=8, show=False)
filepath = OUTPUT_DIR / "bar_plot_abs_max.png"
plt.savefig(filepath, bbox_inches="tight", dpi=150)
plt.close()
