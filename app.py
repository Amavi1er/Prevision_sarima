import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller 
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, Optional, Dict
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import openai
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
from io import StringIO
import json
import os

warnings.filterwarnings('ignore')

class AIAssistant:
    def __init__(self, api_key):
        openai.api_key = api_key
        
    def get_response(self, prompt: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "Tu es un assistant spécialisé dans l'analyse des séries temporelles. "
                              "Tu dois aider les utilisateurs à comprendre leurs données et les résultats "
                              "du modèle SARIMA. Sois précis et pédagogue."
                },
                {
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erreur de communication avec l'assistant: {str(e)}"

class EnhancedSARIMAForecastApp:
    def __init__(self):
        st.set_page_config(
            page_title="Prévision de Séries Temporelles",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialisation des variables de session
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'target_column' not in st.session_state:
            st.session_state.target_column = None
        if 'frequency' not in st.session_state:
            st.session_state.frequency = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'openai_api_key' not in st.session_state:
            st.session_state.openai_api_key = None
        if 'ai_assistant' not in st.session_state:
            st.session_state.ai_assistant = None
        
        self.setup_custom_theme()
        self.setup_ai_assistant()
        
    def setup_custom_theme(self):
        st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            border: none;
        }
        .stProgress > div > div > div {
            background-color: #4CAF50;
        }
        .css-1d391kg {
            padding: 2rem;
        }
        .stMetric {
            background-color:rgb(63, 74, 228);
            padding: 1rem;
            border-radius: 5px;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def setup_ai_assistant(self):
        """Configuration de l'assistant IA avec gestion des erreurs améliorée"""
        # Chercher d'abord dans les variables d'environnement
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Si pas de clé dans l'environnement, vérifier dans la session
        if not api_key and 'openai_api_key' in st.session_state:
            api_key = st.session_state.openai_api_key
        
        # Afficher le statut actuel
        with st.sidebar:
            st.markdown("### Configuration de l'Assistant IA")
            
            if api_key:
                st.success("✅ Assistant IA configuré")
                # Permettre de réinitialiser la configuration
                if st.button("Réinitialiser la configuration"):
                    st.session_state.openai_api_key = None
                   # st.session_state.ai_assistant = None
                    st.rerun()
            else:
                st.warning("⚠️ Assistant IA non configuré")
                # Input pour la clé API
                new_api_key = st.text_input(
                    "Entrez votre clé API OpenAI",
                    type="password",
                    help="Vous pouvez aussi définir la variable d'environnement OPENAI_API_KEY"
                )
                
                if new_api_key:
                    try:
                        # Tester la clé API avant de la sauvegarder
                        test_assistant = AIAssistant(new_api_key)
                        test_response = test_assistant.get_response("Test de connexion")
                        if "Erreur de communication" not in test_response:
                            st.session_state.openai_api_key = new_api_key
                            st.session_state.ai_assistant = test_assistant
                            st.success("✅ Assistant IA configuré avec succès !")
                            st.rerun()
                        else:
                            st.error("❌ Clé API invalide")
                    except Exception as e:
                        st.error(f"❌ Erreur de configuration : {str(e)}")
            
        # Mettre à jour ou créer l'assistant si nécessaire
        if api_key not in st.session_state:
            try:
                st.session_state.ai_assistant = AIAssistant(api_key)
            except Exception as e:
                st.sidebar.error(f"❌ Erreur lors de l'initialisation de l'assistant : {str(e)}")

    def load_local_data(self):
        """Chargement des données depuis un fichier local"""
        file = st.file_uploader("Charger un fichier (CSV ou Excel)", type=['csv', 'xlsx'])
        
        if file is not None:
            try:
                # Lecture du fichier
                if file.name.endswith('.csv'):
                    data = pd.read_csv(file)
                else:
                    data = pd.read_excel(file)
                
                # Configuration des colonnes
                target_column = st.selectbox("Sélectionner la colonne cible", data.columns)
                has_date_column = st.checkbox("Le fichier contient une colonne temporelle")
                
                if has_date_column:
                    date_column = st.selectbox("Sélectionner la colonne temporelle", data.columns)
                    data[date_column] = pd.to_datetime(data[date_column])
                    data.set_index(date_column, inplace=True)
                    frequency = pd.infer_freq(data.index)
                else:
                    start_year = st.number_input(
                        "Entrer l'année de début",
                        min_value=1900,
                        max_value=datetime.now().year,
                        value=2020
                    )
                    frequency = st.selectbox(
                        "Fréquence temporelle",
                        ['M', 'Q', 'Y', 'D', 'W'],
                        format_func=lambda x: {
                            'M': 'Mensuelle',
                            'Q': 'Trimestrielle',
                            'Y': 'Annuelle',
                            'D': 'Journalière',
                            'W': 'Hebdomadaire'
                        }[x]
                    )
                    start_date = f"{start_year}-01-01"
                    data.index = pd.date_range(start=start_date, periods=len(data), freq=frequency)
                
                data = data.asfreq(frequency)
                data[target_column] = data[target_column].interpolate()
                
                # Sauvegarde dans la session
                st.session_state.data = data
                st.session_state.target_column = target_column
                st.session_state.frequency = frequency
                st.session_state.data_loaded = True
                
                st.success("Données chargées avec succès!")
                
            except Exception as e:
                st.error(f"Erreur lors du chargement des données: {str(e)}")

    def load_demo_data(self):
        """Chargement des données de démonstration"""
        # Création de données synthétiques
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=100, freq='M')
        trend = np.linspace(0, 10, 100)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(100) / 12)
        noise = np.random.normal(0, 1, 100)
        values = trend + seasonal + noise
        
        data = pd.DataFrame({
            'valeur': values
        }, index=dates)
        
        # Sauvegarde dans la session
        st.session_state.data = data
        st.session_state.target_column = 'valeur'
        st.session_state.frequency = 'M'
        st.session_state.data_loaded = True
        
        st.success("Données de démonstration chargées avec succès!")

    def display_enhanced_data_overview(self):
        """Affichage amélioré de l'aperçu des données"""
        if not st.session_state.data_loaded:
            return
            
        st.header("📊 Aperçu des Données")
        
        data = st.session_state.data
        target_column = st.session_state.target_column
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre d'observations", len(data))
        with col2:
            st.metric("Période de début", data.index.min().strftime('%Y-%m-%d'))
        with col3:
            st.metric("Période de fin", data.index.max().strftime('%Y-%m-%d'))
        
        # Statistiques descriptives avec mise en forme
        st.subheader("📈 Statistiques Descriptives")
        stats_df = pd.DataFrame({
            "Statistique": ["Moyenne", "Médiane", "Écart-type", "Min", "Max"],
            "Valeur": [
                f"{data[target_column].mean():.2f}",
                f"{data[target_column].median():.2f}",
                f"{data[target_column].std():.2f}",
                f"{data[target_column].min():.2f}",
                f"{data[target_column].max():.2f}"
            ]
        })
        st.table(stats_df)
        
        # Graphique interactif
        fig = self.create_interactive_plot()
        st.plotly_chart(fig, use_container_width=True)

    def create_interactive_plot(self):
        """Création d'un graphique interactif"""
        data = st.session_state.data
        target_column = st.session_state.target_column
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[target_column],
            mode='lines+markers',
            name='Données',
            line=dict(color='#4CAF50')
        ))
        
        fig.update_layout(
            title={
                'text': 'Série Temporelle',
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Date',
            yaxis_title='Valeur',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig

    def seasonality_analysis(self):
        """Analyse de la saisonnalité"""
        if not st.session_state.data_loaded:
            st.warning("⚠️ Veuillez d'abord charger des données")
            return
            
        data = st.session_state.data
        target_column = st.session_state.target_column
        frequency = st.session_state.frequency
        
        st.subheader("🔄 Analyse de la Saisonnalité")
        
        # Décomposition saisonnière
        if frequency=="M":
            decomposition = seasonal_decompose(data[target_column], period=12)
        elif frequency=="Q":
            decomposition = seasonal_decompose(data[target_column], period=4)
        elif frequency=="W":
            decomposition = seasonal_decompose(data[target_column], period=7)
        
        
        fig = go.Figure()
        
        # Tendance
        fig.add_trace(go.Scatter(
            x=data.index,
            y=decomposition.trend,
            name='Tendance',
            line=dict(color='#4CAF50')
        ))
        
        # Saisonnalité
        fig.add_trace(go.Scatter(
            x=data.index,
            y=decomposition.seasonal,
            name='Saisonnalité',
            line=dict(color='#2196F3')
        ))
        
        # Résidus
        fig.add_trace(go.Scatter(
            x=data.index,
            y=decomposition.resid,
            name='Résidus',
            line=dict(color='#F44336')
        ))
        
        fig.update_layout(
            title='Décomposition de la Série Temporelle',
            xaxis_title='Date',
            yaxis_title='Valeur',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def statistical_tests(self):
        """Tests statistiques"""
        if not st.session_state.data_loaded:
            return
            
        data = st.session_state.data
        target_column = st.session_state.target_column
        
        st.subheader("📊 Tests Statistiques")
        
        # Test de stationnarité (ADF)
        adf_result = adfuller(data[target_column].dropna())
        
        st.write("### Test de Stationnarité (Augmented Dickey-Fuller)")
        results_df = pd.DataFrame({
            'Métrique': ['Statistique ADF', 'p-value', 'Valeur critique (1%)', 
                        'Valeur critique (5%)', 'Valeur critique (10%)'],
            'Valeur': [adf_result[0], adf_result[1]] + list(adf_result[4].values())
        })
        st.table(results_df)
        
        # Interprétation
        if adf_result[1] < 0.05:
            st.success("✅ La série est stationnaire (p-value < 0.05)")
        else:
            st.warning("⚠️ La série n'est pas stationnaire (p-value > 0.05)")

    def setup_forecast_params(self):
        """Configuration des paramètres de prévision"""
        if not st.session_state.data_loaded:
            return
            
        st.subheader("⚙️ Configuration des Prévisions")
        
        col1, col2 = st.columns(2)
        with col1:
            self.forecast_steps = st.number_input(
                "Nombre de pas de prévision",
                min_value=1,
                max_value=36,
                value=12
            )
        
        with col2:
            self.confidence_interval = st.slider(
                "Intervalle de confiance (%)",
                min_value=80,
                max_value=99,
                value=95
            )

    def train_and_evaluate(self):
        """Entraînement et évaluation du modèle"""
        if not st.session_state.data_loaded:
            return
            
        data = st.session_state.data
        target_column = st.session_state.target_column
        
        # Séparation train/test
        train_size = int(len(data) * 0.8)
        self.train_data = data[:train_size]
        self.test_data = data[train_size:]
        
        # Recherche des meilleurs paramètres
        with st.spinner("🔍 Recherche des meilleurs paramètres..."):
            self.best_params, self.best_aic = self.find_best_sarima(self.train_data[target_column])
            
            if self.best_params is not None:
                order, seasonal_order = self.best_params
                self.model = SARIMAX(
                    self.train_data[target_column],
                    order=order,
                    seasonal_order=seasonal_order
                )
                self.model_fit = self.model.fit(disp=False)
                
                # Évaluation sur l'ensemble de test
                self.test_predictions = self
                # Évaluation sur l'ensemble de test
                self.test_predictions = self.model_fit.forecast(steps=len(self.test_data))
                self.compute_metrics()
                self.display_evaluation_results()

    def compute_metrics(self):
        """Calcul des métriques de performance"""
        if not st.session_state.data_loaded:
            return
            
        target_column = st.session_state.target_column
        
        self.metrics = {
            'MSE': mean_squared_error(self.test_data[target_column], self.test_predictions),
            'RMSE': np.sqrt(mean_squared_error(self.test_data[target_column], self.test_predictions)),
            'MAE': mean_absolute_error(self.test_data[target_column], self.test_predictions),
            'R2': r2_score(self.test_data[target_column], self.test_predictions)
        }

    def display_evaluation_results(self):
        """Affichage des résultats d'évaluation"""
        if not st.session_state.data_loaded:
            return
            
        target_column = st.session_state.target_column
        
        st.subheader("📊 Résultats de l'Évaluation")
        
        # Métriques
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", f"{self.metrics['MSE']:.4f}")
        col2.metric("RMSE", f"{self.metrics['RMSE']:.4f}")
        col3.metric("MAE", f"{self.metrics['MAE']:.4f}")
        col4.metric("R²", f"{self.metrics['R2']:.4f}")
        
        # Graphique des prédictions vs réalité
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.test_data.index,
            y=self.test_data[target_column],
            name='Valeurs réelles',
            line=dict(color='#4CAF50')
        ))
        
        fig.add_trace(go.Scatter(
            x=self.test_data.index,
            y=self.test_predictions,
            name='Prédictions',
            line=dict(color='#2196F3')
        ))
        
        fig.update_layout(
            title='Prédictions vs Valeurs Réelles',
            xaxis_title='Date',
            yaxis_title='Valeur',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def find_best_sarima(self, data: pd.Series) -> Tuple[Optional[Tuple], float]:
        """
        Recherche des meilleurs paramètres SARIMA
        """
        p = d = q = range(0, 2)
        pdq = list(itertools.product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
        
        best_aic = float('inf')
        best_params = None
        
        progress_bar = st.progress(0)
        total_iterations = len(pdq) * len(seasonal_pdq)
        current_iteration = 0
        
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                current_iteration += 1
                progress_bar.progress(current_iteration / total_iterations)
                
                try:
                    model = SARIMAX(
                        data,
                        order=param,
                        seasonal_order=param_seasonal
                    )
                    results = model.fit(disp=False)
                    
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_params = (param, param_seasonal)
                        
                except Exception:
                    continue
        
        return best_params, best_aic

    def make_forecast(self):
        """
        Génération des prévisions
        """
        if not st.session_state.data_loaded:
            return
            
        data = st.session_state.data
        target_column = st.session_state.target_column
        frequency = st.session_state.frequency
        
        st.subheader("🔮 Prévisions")
        
        # Prévisions
        forecast = self.model_fit.forecast(steps=self.forecast_steps)
        forecast_index = pd.date_range(
            start=data.index[-1],
            periods=self.forecast_steps + 1,
            freq=frequency
        )[1:]
        
        # Intervalles de confiance
        forecast_ci = self.model_fit.get_forecast(steps=self.forecast_steps)
        conf_int = forecast_ci.conf_int(alpha=(100-self.confidence_interval)/100)
        
        # Création du graphique
        fig = go.Figure()
        
        # Données historiques
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[target_column],
            name='Données historiques',
            line=dict(color='#4CAF50')
        ))
        
        # Prévisions
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast,
            name='Prévisions',
            line=dict(color='#2196F3')
        ))
        
        # Intervalles de confiance
        fig.add_trace(go.Scatter(
            x=forecast_index.tolist() + forecast_index.tolist()[::-1],
            y=conf_int.iloc[:, 0].tolist() + conf_int.iloc[:, 1].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(33, 150, 243, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            name=f'Intervalle de confiance ({self.confidence_interval}%)'
        ))
        
        fig.update_layout(
            title='Prévisions avec Intervalles de Confiance',
            xaxis_title='Date',
            yaxis_title='Valeur',
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau des prévisions
        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            'Prévision': forecast,
            'Borne Inférieure': conf_int.iloc[:, 0],
            'Borne Supérieure': conf_int.iloc[:, 1]
        })
        forecast_df.set_index('Date', inplace=True)
        
        st.write("### Détail des Prévisions")
        st.dataframe(forecast_df.round(2))

    def run(self):
        """
        Point d'entrée principal de l'application
        """
        st.title("📈 Application de Prévision de Séries Temporelles")
        
        # Menu de navigation
        selected = option_menu(
            menu_title=None,
            options=["Données", "Analyse", "Prévisions", "Assistant IA"],
            icons=["database", "graph-up", "crystal-ball", "robot"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal"
        )
        
        if selected == "Données":
            st.header("📊 Chargement des Données")
            data_source = st.radio(
                "Source des données",
                ["Fichier Local", "Données de Démonstration"]
            )
            
            if data_source == "Fichier Local":
                self.load_local_data()
            else:
                self.load_demo_data()
            
            if st.session_state.data_loaded:
                self.display_enhanced_data_overview()
        
        elif selected == "Analyse":
            if not st.session_state.data_loaded:
                st.warning("⚠️ Veuillez d'abord charger des données")
                return
            
            self.seasonality_analysis()
            self.statistical_tests()
        
        elif selected == "Prévisions":
            if not st.session_state.data_loaded:
                st.warning("⚠️ Veuillez d'abord charger des données")
                return
            
            self.setup_forecast_params()
            
            if st.button("Lancer l'analyse"):
                self.train_and_evaluate()
                self.make_forecast()
        
        elif selected == "Assistant IA":
            if not st.session_state.data_loaded:
                st.warning("⚠️ Veuillez d'abord charger des données")
                return
                
            
                
            user_question = st.text_input("Posez votre question sur les données")
            if user_question:
                if not st.session_state.data_loaded:
                    st.error("⚠️ Veuillez d'abord charger des données")
                    return
                    
                try:
                    data = st.session_state.data
                    target_column = st.session_state.target_column
                    frequency = st.session_state.frequency
                    
                    context = f"""
                    Contexte des données :
                    - Période : de {data.index.min()} à {data.index.max()}
                    - Fréquence : {frequency}
                    - Variable analysée : {target_column}
                    - Statistiques principales :
                        * Moyenne : {data[target_column].mean():.2f}
                        * Écart-type : {data[target_column].std():.2f}
                        * Min : {data[target_column].min():.2f}
                        * Max : {data[target_column].max():.2f}
                    
                    Question : {user_question}
                    """
                    
                    with st.spinner("🤔 L'assistant réfléchit..."):
                        response = st.session_state.ai_assistant.get_response(context)
                        if "Erreur de communication" in response:
                            st.error(response)
                        else:
                            st.write("Réponse de l'assistant :")
                            st.markdown(response)
                            
                except Exception as e:
                    st.error(f"❌ Une erreur est survenue : {str(e)}")



if __name__ == "__main__":
    app = EnhancedSARIMAForecastApp()
    app.run()