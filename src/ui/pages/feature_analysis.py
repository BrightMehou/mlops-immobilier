from pathlib import Path

import streamlit as st
from PIL import Image

st.set_page_config(page_title="Feature analysis", page_icon="🏠", layout="wide")

st.title("🧠 Analyse des caractéristiques")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📈 Dépendance Partielle",  # Global   – effet moyen d'une variable
        "🐝 SHAP – Vue d'ensemble",  # Global   – distribution des contributions
        "🔍 SHAP – Par variable",  # Semi-local – zoom feature par feature
        "📊 SHAP – Résumé",  # Agrégé   – importance synthétique
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Dépendance Partielle  [GLOBAL – effet moyen par variable]
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        """
La **dépendance partielle** montre la relation entre une ou plusieurs caractéristiques (*features*)
et la prédiction moyenne du modèle, tout en "neutralisant" l'effet des autres caractéristiques.
Elle répond à la question : *Comment les prédictions changent-elles en fonction d'une
caractéristique donnée, en moyenne ?*

**Limites :**
- Suppose que les caractéristiques sont indépendantes, ce qui n'est pas toujours vrai dans les
  données réelles.
- Les interactions entre variables ne sont pas directement visibles.
""",
    )

    output_dir = Path("data/feature_analysis")
    partial_dependence_pattern = str(output_dir / "partial_dependence_*.png")
    partial_dependence_images = sorted(output_dir.glob("partial_dependence_*.png"))

    if not partial_dependence_images:
        st.warning(
            f"⚠️ Aucune image trouvée pour le motif : {partial_dependence_pattern}",
        )
    else:
        for filepath_str in partial_dependence_images:
            filepath = Path(filepath_str)
            image = Image.open(filepath)
            filename = filepath.stem
            st.subheader(f"📈 {filename}")
            st.image(image, width="stretch")
            st.divider()

    st.markdown(
        """
**MedInc (Revenu Médian par Groupe)**
On observe une corrélation positive entre le revenu médian et la valeur médiane des maisons.
Plus le revenu médian augmente, plus la valeur moyenne des prédictions des maisons croît.
Cela confirme les observations faites lors de l'analyse exploratoire des données.

---

**HouseAge (Âge des Maisons)**
La relation entre l'âge des maisons et leur valeur suit une courbe en forme de U. Les maisons
neuves sont estimées avec une valeur plus élevée que celles ayant entre 5 et 20 ans. Cependant,
à mesure que l'âge des maisons augmente au-delà de 20 ans, leur valeur moyenne tend à augmenter
progressivement. Cela suggère qu'un âge avancé peut également être perçu comme une marque de
caractère ou de localisation avantageuse.

---

**AveRooms (Nombre Moyen de Pièces par Logement)**
La valeur médiane des maisons augmente de manière significative avec le nombre moyen de pièces
par logement, suivant des paliers bien distincts :
- Les logements ayant entre **1 et 3 pièces** sont en moyenne évalués à **200 000 dollars**.
- Ceux ayant entre **3 et 60 pièces** atteignent en moyenne **260 000 dollars**.
- Enfin, les logements avec plus de **60 pièces** sont estimés autour de **340 000 dollars**.

---

**AveBedrms (Nombre Moyen de Chambres par Logement)**
La valeur médiane des maisons varie légèrement par paliers en fonction du nombre moyen de
chambres :
- Les logements ayant entre **1 et 10 chambres** sont estimés en moyenne à **230 000 dollars**.
- Ceux ayant entre **10 et 17 chambres** voient leur valeur légèrement diminuer à **220 000 dollars**.
- Enfin, au-delà de **17 chambres**, la valeur augmente à nouveau pour atteindre **235 000 dollars**.

---

**Population (Densité de Population)**
La densité de population n'a qu'un impact limité sur la valeur médiane des maisons. La valeur des
biens augmente légèrement avec la densité, mais seulement jusqu'à un certain seuil, au-delà duquel
elle stagne.

---

**AveOccup (Nombre Moyen de Personnes par Logement)**
Le nombre moyen de personnes par logement semble avoir une influence très faible sur les
prédictions. La valeur moyenne des biens reste presque constante, quel que soit ce nombre.

---

**Latitude**
La valeur des biens immobiliers diminue de manière progressive à mesure que la latitude augmente.
Cela confirme que les biens situés plus au sud, proches des zones côtières, ont des valeurs plus
élevées.

---

**Longitude**
De manière similaire à la latitude, la valeur des biens diminue lorsque la longitude augmente,
c'est-à-dire en s'éloignant de la côte californienne. Cette observation est cohérente avec les
villes comme **San Diego**, **Los Angeles**, **San Jose**, ou **San Francisco**.
""",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SHAP Beeswarm  [GLOBAL – distribution des contributions]
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        """
### Valeurs SHAP — définition

Les valeurs SHAP mesurent la contribution exacte de chaque caractéristique à la prédiction d'une
observation spécifique, en utilisant les valeurs de Shapley issues de la théorie des jeux.

Pour une observation donnée, les valeurs SHAP calculent la différence entre la prédiction obtenue
avec et sans chaque caractéristique, en considérant toutes les combinaisons possibles.
Les valeurs SHAP sont **additives** : la somme des contributions de toutes les caractéristiques
est égale à la prédiction finale moins la valeur de base (*expected value*).

**Avantages :**
- Prennent en compte les interactions entre caractéristiques.
- Applicables aux explications locales (par observation) et globales (agrégées).

---

Le **beeswarm plot** donne une vue d'ensemble de la distribution des valeurs SHAP sur l'ensemble
des observations. Chaque point représente une observation ; la couleur indique la valeur de la
caractéristique (rouge = élevée, bleue = faible).
""",
    )

    filepath_beeswarm = output_dir / "beeswarm_plot.png"
    try:
        image_beeswarm = Image.open(filepath_beeswarm)
        st.image(image_beeswarm, width="stretch")
    except FileNotFoundError:
        st.warning(f"⚠️ Image introuvable : {filepath_beeswarm}")

    st.markdown(
        """
Le graphique confirme les résultats obtenus lors de l'analyse des dépendances partielles.
Les variables qui influencent le plus les prédictions du modèle sont **Latitude**, **Longitude**,
et **MedInc**. Ces caractéristiques jouent un rôle clé dans la détermination des valeurs médianes
des maisons. Ensuite, des variables comme **AveOccup** et **AveRooms** exercent une influence
moindre, mais non négligeable. Enfin, **HouseAge**, **Population**, et **AveBedrms** ont une
contribution très faible.
""",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SHAP Scatter  [SEMI-LOCAL – zoom par variable]
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        """
Les **scatter plots SHAP** montrent, pour chaque caractéristique, la valeur SHAP de chaque
observation en fonction de la valeur de la caractéristique. Ils permettent de visualiser des
effets non-linéaires et des interactions que la dépendance partielle ne capture pas toujours.
""",
    )

    scatter_plot_pattern = str(output_dir / "scatter_plot_*.png")
    scatter_plot_images = sorted(output_dir.glob("scatter_plot_*.png"))

    if not scatter_plot_images:
        st.warning(f"⚠️ Aucune image trouvée pour le motif : {scatter_plot_pattern}")
    else:
        for filepath_str in scatter_plot_images:
            filepath = Path(filepath_str)
            image = Image.open(filepath)
            filename = filepath.stem
            st.subheader(f"📈 {filename}")
            st.image(image, width="stretch")
            st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SHAP Bar plots  [AGRÉGÉ – importance synthétique]
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        """
Les graphiques ci-dessous agrègent les valeurs SHAP sur l'ensemble des observations pour fournir
une mesure synthétique de l'importance de chaque caractéristique à l'échelle du modèle entier.
""",
    )

    # Bar plot – moyenne des valeurs SHAP absolues
    filepath_bar = output_dir / "bar_plot.png"
    try:
        image_bar = Image.open(filepath_bar)
        st.subheader("📊 Importance moyenne (|SHAP| moyen)")
        st.image(image_bar, width="stretch")
        st.divider()
    except FileNotFoundError:
        st.warning(f"⚠️ Image introuvable : {filepath_bar}")

    st.markdown(
        """
En moyenne, les caractéristiques **Latitude**, **Longitude**, **MedInc**, **AveOccup**,
**AveRooms**, **HouseAge**, **Population**, et **AveBedrms** contribuent respectivement, en valeur
absolue, à hauteur de **0,86**, **0,75**, **0,35**, **0,19**, **0,11**, **0,05**, **0,03**, et
**0,03** à la prédiction du modèle par rapport à la baseline. Ces résultats confirment l'influence
majeure des trois premières caractéristiques et la contribution plus marginale des autres.
""",
    )

    # Bar plot – maximum des valeurs SHAP absolues
    filepath_bar_abs_max = output_dir / "bar_plot_abs_max.png"
    try:
        image_bar_abs_max = Image.open(filepath_bar_abs_max)
        st.subheader("📊 Importance maximale (|SHAP| max)")
        st.image(image_bar_abs_max, width="stretch")
        st.divider()
    except FileNotFoundError:
        st.warning(f"⚠️ Image introuvable : {filepath_bar_abs_max}")

    st.markdown(
        """
Ce graphique illustre que les caractéristiques **Longitude**, **MedInc** et **AveRooms** affichent
les contributions maximales les plus élevées, toutes proches de **2**. Cela montre que la
**localisation géographique** et le **revenu médian des ménages** jouent un rôle déterminant dans
les prédictions du modèle. **AveOccup** et **HouseAge** présentent des contributions
intermédiaires, tandis que **Population**, **AveBedrms** et les autres variables affichent des
contributions plus faibles mais non négligeables.
""",
    )
