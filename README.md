# Prévision de Séries Temporelles

## Description
Cette application Streamlit permet de réaliser des prévisions de séries temporelles en utilisant le modèle SARIMA (Seasonal Autoregressive Integrated Moving Average). Elle offre une interface utilisateur interactive pour charger des données, effectuer des analyses saisonnières et générer des prévisions.

## Fonctionnalités
- **Chargement de données** : Importez des données depuis des fichiers locaux au format CSV ou Excel, ou utilisez des données de démonstration.
- **Analyse de la saisonnalité** : Évaluez la saisonnalité des données pour mieux comprendre les tendances.
- **Tests statistiques** : Effectuez des tests pour évaluer la stationnarité des séries temporelles.
- **Configuration des paramètres SARIMA** : Personnalisez les paramètres du modèle SARIMA pour vos prévisions.
- **Visualisation interactive** : Visualisez les résultats avec des graphiques interactifs créés avec Plotly.
- **Évaluation des performances** : Obtenez des métriques de performance telles que l'erreur quadratique moyenne (RMSE) et le coefficient de détermination (R²).

## Installation
1. **Clonez le dépôt** :
   ```bash
   git clone https://github.com/Amavi1er/Prevision_sarima.git
   cd Prevision_sarima

2. Installez les dépendances:
   ```bash
   pip install -r requirements.txt

## Exécution de l'application
    Pour exécuter l'application, utilisez la commande suivante :
    ```bash
    streamlit run app.py

## Utilisation
    1. Chargez vos données au format CSV ou Excel, ou utilisez les données de démonstration fournies.
    2. Configurez les paramètres de prévision selon vos besoins dans l'interface utilisateur.
    3. Visualisez les résultats et les métriques de performance après l'exécution du modèle.

## Auteurs
- Amavi1er