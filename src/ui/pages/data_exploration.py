"""Application Streamlit d'exploration du dataset California Housing.
Elle permet de visualiser les données brutes, d'analyser les statistiques descriptives,
d'explorer les corrélations entre variables, et de représenter les distributions
ainsi que la répartition géographique des logements en Californie.
"""

import logging

import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.datasets import fetch_california_housing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Data exploration", page_icon="🏠", layout="wide")


@st.cache_data
def load_data():
    logger.info("Chargement du dataset California Housing...")
    housing = fetch_california_housing(as_frame=True)
    data = housing.data
    data["MedHouseVal"] = housing.target * 100000
    logger.info("Dataset chargé avec succès.")
    return data


data = load_data()

st.title("📊 Exploration des données - California Housing Dataset")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📋 Présentation",
        "📈 Statistiques descriptives",
        "📊 Distributions",
        "🔗 Corrélations",
        "📉 Pairplot",
        "🗺️ Carte",
    ],
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Présentation du dataset + données brutes
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Présentation du dataset")

    st.markdown(
        """
Voici un résumé du dataset California Housing utilisé dans ce projet :

**Caractéristiques :**
- **Instances :** 20 640
- **Attributs :** 8 attributs numériques et 1 variable cible
- **Valeurs Manquantes :** Aucune

**Attributs :**
1. **MedInc** - Revenu médian dans le groupe de blocs
2. **HouseAge** - Âge médian des maisons dans le groupe de blocs
3. **AveRooms** - Nombre moyen de pièces par ménage
4. **AveBedrms** - Nombre moyen de chambres par ménage
5. **Population** - Population du groupe de blocs
6. **AveOccup** - Nombre moyen de membres par ménage
7. **Latitude** - Latitude du groupe de blocs
8. **Longitude** - Longitude du groupe de blocs

**Variable Cible :** Valeur médiane des maisons pour les districts de Californie
(en centaines de milliers de dollars)

**Source :** Dérivé du recensement américain de 1990, obtenu dans le dépôt StatLib.
Il peut être chargé en utilisant la fonction `sklearn.datasets.fetch_california_housing`.

Le dataset représente des données au niveau du groupe de blocs, un groupe de blocs étant une
petite unité géographique avec une population de 600 à 3 000 personnes.

Ce dataset a été référencé dans l'article suivant :
- Pace, R. Kelley et Ronald Barry, *Sparse Spatial Autoregressions*,
  Statistics and Probability Letters, 33 (1997) 291-297
""",
    )

    with st.expander("🔍 Voir les données brutes"):
        st.dataframe(data)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Statistiques descriptives
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Statistiques descriptives")
    st.dataframe(data.describe())

    st.markdown(
        """
**Observations Notables**

**Valeurs Extrêmes:**

- AveRooms, AveBedrms et AveOccup montrent des valeurs maximales très élevées par rapport à leurs
  moyennes et médianes. Cela suggère quelques districts extrêmes.
- Les valeurs extrêmes peuvent fausser les statistiques descriptives et les modèles notamment les
  modèles linéaires ; il peut donc être nécessaire de les traiter spécifiquement (par exemple, en
  les limitant ou en les transformant selon les objectifs de l'analyse).

**Asymétrie:**

- MedInc et MedHouseVal tendent à être asymétriques à droite (longue traîne à droite) dans ce
  dataset, ce qui signifie qu'il y a un petit nombre de districts avec des revenus et des valeurs
  immobilières très élevés.

**Répartition Géographique:**

- Les limites de latitude et de longitude confirment que ces points de données proviennent de
  différentes régions de Californie, couvrant des zones côtières, des régions intérieures et
  possiblement des zones montagneuses ou désertiques.

**Relations Potentielles:**

- En général, MedInc est positivement corrélé avec MedHouseVal.
- HouseAge pourrait avoir une relation plus complexe ou plus faible avec MedHouseVal, mais les
  quartiers plus anciens dans certaines zones prisées peuvent quand même avoir des valeurs élevées.
- Population et AveOccup peuvent affecter indirectement les valeurs immobilières (par exemple,
  densité, urbain vs. rural).
""",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Distribution des caractéristiques
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Distribution des caractéristiques")

    feature = st.selectbox("Choisissez une variable :", data.columns)
    fig = px.histogram(data, x=feature, nbins=30, title=f"Histogramme de {feature}")
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
Nous pouvons d'abord nous concentrer sur les caractéristiques dont les distributions seraient
plus ou moins attendues.

Le **revenu médian** présente une distribution avec une longue queue : les salaires sont
globalement distribués normalement, mais quelques individus ont des revenus très élevés.

L'**âge moyen des maisons** suit une distribution plus ou moins uniforme.

La **variable cible** présente également une longue queue, avec un effet de seuil pour les
maisons de grande valeur : toutes les maisons avec un prix supérieur à 5 se voient attribuer
la valeur 5.

Pour les pièces moyennes, les chambres moyennes, l'occupation moyenne et la population,
l'étendue des données est importante avec des valeurs très élevées et peu fréquentes.
""",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Matrice de corrélation
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Matrice de corrélation linéaire")

    corr_matrix = data.corr()
    fig = ff.create_annotated_heatmap(
        z=np.around(corr_matrix.values, decimals=3),
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.columns.tolist(),
        colorscale="Viridis",
    )
    fig.update_layout(title="Matrice de corrélation linéaire", height=700, width=700)
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
### 1. Corrélations les plus fortes entre les caractéristiques

1. **AveRooms & AveBedrms**
   - C'est la corrélation la plus élevée entre les caractéristiques.
   - Intuitivement, si un district a un nombre moyen élevé de pièces, il aura généralement aussi
     un nombre moyen élevé de chambres.

2. **Latitude & Longitude**
   - Relation très fortement négative.
   - En Californie, lorsque la latitude augmente (vers le nord), la longitude devient souvent plus
     négative (vers l'intérieur ou l'ouest).

---

### 2. Corrélation avec **MedHouseVal** (la cible)

- **MedInc & MedHouseVal** — corrélation la plus forte (~0.69) : les zones plus riches ont des
  valeurs immobilières plus élevées.
- **AveRooms & MedHouseVal** — relation modérée et positive.
- **HouseAge & MedHouseVal** — corrélation faible et positive.
- **Population, AveOccup, AveBedrms** — corrélation très faible (|r| < 0.05).

---

### 3. Enseignements pratiques

1. **Focalisez-vous sur le revenu médian** : prédicteur crucial, envisagez des termes polynomiaux
   ou d'interaction.
2. **Vérifiez la multicolinéarité** : AveRooms et AveBedrms sont fortement corrélées (~0.85),
   ce qui peut augmenter la variance des estimations dans un modèle linéaire.
3. **Insights géospatiaux** : la forte corrélation Latitude/Longitude suggère une distribution
   diagonale structurée à travers la Californie — envisagez une modélisation géospatiale.
4. **Surveillez les valeurs extrêmes** dans AveRooms et AveOccup.

---

### Conclusion

**Le revenu médian** est de loin le prédicteur le plus fort de la valeur médiane des maisons.
Les caractéristiques géospatiales montrent une forte corrélation interne due à la géographie de
la Californie. **AveRooms** et **AveBedrms** sont fortement corrélées entre elles, indiquant une
redondance potentielle dans la modélisation.
""",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Diagramme en paires interactif
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Diagramme en paires interactif")

    cols = data.columns.tolist()

    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("Variable X", cols, index=0)
    with col2:
        y_axis = st.selectbox("Variable Y", cols, index=1)
    with col3:
        color_var = st.selectbox("Couleur (optionnel)", [None] + cols)

    if color_var:
        fig = px.scatter(
            data,
            x=x_axis,
            y=y_axis,
            color=color_var,
            color_continuous_scale="Viridis",
            height=600,
        )
    else:
        fig = px.scatter(data, x=x_axis, y=y_axis, height=600)

    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
Bien qu'il soit toujours compliqué d'interpréter un pairplot en raison de la grande quantité de
données, nous pouvons confirmer que certaines caractéristiques présentent des valeurs extrêmes.
Nous pouvons également observer que le **revenu médian** est utile pour distinguer les maisons
de grande valeur de celles de faible valeur.

Ainsi, lors de la création d'un modèle prédictif, nous pouvons nous attendre à ce que la
longitude, la latitude et le revenu médian soient des caractéristiques particulièrement
pertinentes.
""",
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Répartition géographique
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Répartition géographique des logements")

    st.markdown(
        """
La combinaison de la latitude et de la longitude peut nous aider à déterminer s'il existe des
emplacements associés à des maisons de grande valeur. Dans le graphique ci-dessous, la couleur
et la taille des cercles sont liées à la valeur médiane des maisons dans chaque district.
""",
    )

    fig = px.scatter_mapbox(
        data,
        lat="Latitude",
        lon="Longitude",
        color="MedHouseVal",
        hover_data=[
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
        ],
        color_continuous_scale="Viridis",
        size_max=15,
        zoom=5,
        height=600,
        mapbox_style="open-street-map",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
Les maisons de grande valeur se situent principalement sur la côte, là où se trouvent les
grandes villes de Californie : **San Diego**, **Los Angeles**, **San Jose** et **San Francisco**.
""",
    )
