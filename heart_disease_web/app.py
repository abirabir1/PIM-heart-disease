import base64
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import os
import csv

# Fonction pour convertir une image en Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


def main():
    st.set_page_config(
        page_title="Analyse des Risques Cardiaques",
        page_icon="❤️",
        layout="wide",
    )

    # Chemin de l'image et encodage
    image_path = "C:/Users/Abir/Desktop/PIM-Finale/heart_disease_web/pexels-negativespace-48604 (1).jpg"
    base64_image = get_base64_image(image_path)

    # CSS pour personnaliser le style avec l'image de fond
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{base64_image}");
                background-size: cover; /* Couvre toute la zone */
                background-position: center; /* Centre l'image */
                font-family: Arial, sans-serif;
            }}
            .menu-title {{
                font-size: 1.4em;
                font-weight: bold;
                color: #C00000;
                text-align: center;
                margin-bottom: 20px;
            }}
            .main-title {{
                color: #C00000;
                font-size: 2.5em;
                text-align: center;
                font-weight: bold;
            }}
            .section-title {{
                color: #4169E1;
                font-size: 1.8em;
                font-weight: bold;
            }}
            .content-box {{
                background-color: rgba(255, 255, 255, 0.8); /* Fond blanc avec transparence */
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            }}
        </style>
    """, unsafe_allow_html=True)
    
    # Affichage de l'image d'en-tête avec style personnalisé
    st.markdown('<div class="header-image">', unsafe_allow_html=True)
    
    # Création des colonnes
    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 1])
    
    # Affichage de l'image dans la colonne 3
    with col3:
        st.image("C:/Users/Abir/Desktop/PIM-Finale/heart_disease_web/WhatsApp_Image2-removebg-preview.png", width=400, use_container_width=False, output_format="auto", clamp=False, channels="RGB", caption=None)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Menu personnalisé avec style et icônes
    st.sidebar.markdown(""" 
        <p class="menu-title" style="
            color: #C00000;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            margin: 10px 0;
            border-bottom: 2px solid #C00000;
            text-transform: uppercase;
            letter-spacing: 2px;
        ">Menu</p>
    """, unsafe_allow_html=True)
    
    menu_options = {
        "🏠 Accueil": "Accueil",
        "📊 Analyse Exploratoire": "Analyse Exploratoire", 
        "⚙️ Modélisation": "Modélisation",
        "🔍 Prédictions": "Prédictions",
    }

    # Application de styles CSS aux options de menu
    st.sidebar.markdown(""" 
        <style>
            .stRadio label {
                font-size: 20px; /* Taille de police agrandie */
                font-weight: bold; /* Mettre en gras */
                color: #C00000; /* Couleur du texte */
            }
            .stRadio {
                margin: 10px 0; /* Ajout d'un espacement autour des options */
            }
            /* Style pour les boutons radio */
            .stRadio > div[role="radiogroup"] > label > div:first-of-type {
                background-color: #C00000 !important;
                border-color: #C00000 !important;
            }
            [data-testid="stSidebar"] {
                background-color: rgba(255, 255, 255, 0.85);
            }
        </style>
    """, unsafe_allow_html=True)

    choice = st.sidebar.radio("", list(menu_options.keys()))

    # Chargement des données
    @st.cache_data
    def load_data():
        data = pd.read_csv("C:/Users/Abir/Desktop/PIM-Finale/trainTest.csv", encoding="latin-1")
        return data

    data = load_data()

    # Accueil
    if choice == "🏠 Accueil":
        st.markdown(
            """
            <div style="padding: 10px; background-color: rgba(255, 255, 255, 0.85); border-radius: 10px; text-align: center; margin-bottom: 30px;">
                <p style="color: #c20e0e; font-weight: bold; font-size: 2.3em; text-align: center;">Nabdh : La prévention au cœur de la santé.</p>
                <p style="font-size: 1.5em; text-align: center;">Parce qu'une détection précoce peut faire toute la différence.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Nouvelle section pour À propos du projet et Fonctionnalités principales
        col1, col2 = st.columns(2)

        with col1:
            # Ajout d'un cadre
            st.markdown(
                """
                <div style="background-color: rgba(255, 255, 255, 0.85); border-radius: 10px; padding: 10px;">
                <h3 style="color: #002d67; font-weight: bold;">À propos du projet:</h3>
                    <p style="text-align: left; font-size: 20px; ">Ce projet utilise des algorithmes de Machine Learning pour prédire les risques de crise cardiaque à partir de données cliniques. Notre objectif est de faciliter une prise de décision précoce pour les professionnels de santé et de sensibiliser les patients aux risques potentiels.</p>
                    <p style="text-align: left; font-size: 20px; "><strong>Objectifs principaux :</strong></p>
                    <ul style="text-align: left; font-size: 20px;  ">
                        <li>Prédiction fiable des risques.</li>
                        <li>Encourager la prévention.</li>
                        <li>Améliorer la prise en charge des patients.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            # Ajout d'un cadre
            st.markdown(
                """
                <div style="background-color: rgba(255, 255, 255, 0.85); border-radius: 10px; padding: 10px;">
                <h3 style="color: #002d67; font-weight: bold;">Fonctionnalités principales:</h3>
                    <ul style="text-align: left; font-size: 20px; ">
                        <li>Saisie des données cliniques (âge, tension, taux de sucre, fréquence cardiaque, etc.).</li>
                        <li>Résultat immédiat de la prédiction des risques.</li>
                        <li>Visualisation graphique des données (histogrammes, corrélations, etc.).</li>
                        <li>Suggestions basées sur le résultat.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
# Analyse Exploratoire
    elif choice == "📊 Analyse Exploratoire":
        st.markdown(
            "<h2 style='color:#c20e0e; font-weight: bold; text-align: center;'>Analyse Exploratoire des Données</h2>",
            unsafe_allow_html=True,
        )
        
        # Colonnes avec alignement centré
        col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
        with col1:
            st.metric(":blue[Total Patients]", len(data), help="Nombre total de patients dans la base de données")
        with col2:
            st.metric(":red[Patients à Risque]", len(data[data['target'] == 1]), delta_color="inverse", help="Patients identifiés comme étant à risque")
        with col3:
            st.metric(":green[Patients sans Risque]", len(data[data['target'] == 0]), help="Patients identifiés comme n'étant pas à risque")

        # Centrer les éléments suivants
        st.markdown("<h3 style='text-align: center;'>Aperçu des Données</h3>", unsafe_allow_html=True)
        st.dataframe(data.head(), height=200)

        st.markdown("<h3 style='text-align: center;'>Statistiques Descriptives</h3>", unsafe_allow_html=True)
        st.dataframe(data.describe().transpose())
        
        # Sélection de colonne pour l'exploration
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        col = st.selectbox("Choisissez une variable pour explorer:", numeric_cols)

        # Histogramme centré
        st.markdown("<h4 style='text-align: center;'>Distribution des valeurs</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, color="#c20e0e")
        plt.title(f"Distribution de {col}", fontsize=16)
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        st.pyplot(fig)

        # Boxplot centré
        st.markdown("<h4 style='text-align: center;'>Box Plot par Classe</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.boxplot(x=data['target'], y=data[col], palette="coolwarm")
        plt.title(f"Box Plot de {col} par classe", fontsize=16)
        plt.xlabel("Classe (0: Sans risque, 1: À risque)")
        plt.ylabel(col)
        st.pyplot(fig)

        # Matrice de corrélation centrée
        st.markdown("<h4 style='text-align: center;'>Matrice de Corrélation</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Matrice de Corrélation", fontsize=16)
        st.pyplot(fig)

        # Répartition des classes cibles centrée
        st.markdown("<h3 style='text-align: center;'>Répartition des Classes Cibles</h3>", unsafe_allow_html=True)
        class_counts = data['target'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6))
        class_counts.plot.pie(
            autopct='%1.1f%%',
            colors=['#8B0000', '#4682B4'],
            startangle=90,
            explode=(0.1, 0),
            shadow=True,
            ax=ax,
            labels=['Sans risque', 'À risque']
        )
        plt.title("Répartition des classes cibles", fontsize=14)
        plt.ylabel('')
        st.pyplot(fig)

        # Corrélation entre les features et la target
        st.markdown("<h3 style='text-align: center;'>Corrélation entre les Features et la Target</h3>", unsafe_allow_html=True)
        correlation_matrix = data.corr()
        correlation_with_target = correlation_matrix['target'].sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            x=correlation_with_target.index,
            y=correlation_with_target.values,
            palette=['darkblue', 'mediumblue', 'royalblue', 'darkred', 'firebrick', 'indianred'][:len(correlation_with_target)],
            ax=ax
        )
        plt.title('Corrélation entre les Features et la Target', fontsize=16)
        plt.xlabel('Features')
        plt.ylabel('Corrélation')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)


    # Modélisation
    elif choice == "⚙️ Modélisation":
        st.markdown(
            "<h2 style='color: #c20e0e; font-weight: bold;'>Modélisation avec Comparaison des Modèles</h2>",
            unsafe_allow_html=True,
        )

        # Préparation des données
        X = data.drop('target', axis=1)
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Normalisation des caractéristiques
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Modèles à entraîner
        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "SVM (RBF Kernel)": SVC(probability=True, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5, min_samples_split=10),
        }

        # Comparaison des courbes ROC
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Ligne de base")

        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

        plt.title("Comparaison des courbes ROC des modèles")
        plt.xlabel("Taux de Faux Positifs")
        plt.ylabel("Taux de Vrais Positifs")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Matrice de Confusion pour Random Forest
        st.write("### Matrice de Confusion pour Random Forest")
        rf_model = models["Random Forest"]
        y_pred_rf = rf_model.predict(X_test_scaled)
        conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Classe 0', 'Classe 1'], yticklabels=['Classe 0', 'Classe 1'])
        ax.set_title("Matrice de confusion pour Random Forest")
        ax.set_xlabel("Prédictions")
        ax.set_ylabel("Vérités terrain")
        st.pyplot(fig)

        # Tuning du modèle Random Forest
        st.write("### Tuning du Modèle Random Forest")
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [5, 10, 15]
        }

        def tune_model(model, param_grid, X, y):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X, y)
            return grid_search
         
        rf_model = RandomForestClassifier(random_state=42)
        grid_search = tune_model(rf_model, param_grid, X_train_scaled, y_train)
        best_rf = grid_search.best_estimator_
        st.write(f"Meilleurs paramètres pour Random Forest: {grid_search.best_params_}")
        st.write(f"Meilleur score CV: {grid_search.best_score_:.4f}")

        # Évaluer le modèle optimisé avec cross-validation
        cv_scores = cross_val_score(
            best_rf,
            X_train_scaled,
            y_train,
            cv=5,
            scoring='roc_auc'
        )
        st.write(f"Random Forest CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

       # Visualiser la courbe ROC pour le modèle optimisé
        y_proba_best_rf = best_rf.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba_best_rf)
        roc_auc = auc(fpr, tpr)

        # Création de la figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='black', label="Ligne de base")
        
        # Configuration de l'affichage
        ax.set_title("Courbe ROC sur le Validation Set", fontsize=14, fontweight='bold')
        ax.set_xlabel("Taux de Faux Positifs", fontsize=12)
        ax.set_ylabel("Taux de Vrais Positifs", fontsize=12)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)   
        # Afficher la figure dans Streamlit
        st.pyplot(fig)
    # Prédictions
    if choice == "🔍 Prédictions":
        st.markdown(
            "<h2 style='color:#c20e0e ; font-weight: bold;'>Faire une Prédiction</h2>",
            unsafe_allow_html=True,
        )
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Âge", min_value=20, max_value=100, value=30)
                sex = st.selectbox("Sexe (0=Femme, 1=Homme)", [0, 1])
                cp = st.selectbox("Type de Douleur Thoracique (0 à 3)", [0, 1, 2, 3])
                trestbps = st.number_input("Tension Artérielle", min_value=90, max_value=200)
                
            with col2:
                chol = st.number_input("Cholestérol", min_value=100, max_value=600)
                fbs = st.selectbox("Glycémie à jeun > 120 mg/dl (0=Non, 1=Oui)", [0, 1])
                restecg = st.selectbox("Résultats ECG (0 à 2)", [0, 1, 2])

            submitted = st.form_submit_button("Faire une Prédiction")
            
            if submitted:
                # Organiser les données en un dictionnaire
                patient_data = {
                    "age": age,
                    "sex": sex,
                    "cp": cp,
                    "trestbps": trestbps,
                    "chol": chol,
                    "fbs": fbs,
                    "restecg": restecg,
                }

                # Chemin vers le fichier CSV
                file_path = "C:/Users/Abir/Desktop/PIM-Finale/patient_data.csv"

                try:
                    # Créer le répertoire parent si nécessaire
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    # Vérifier si le fichier existe déjà
                    if not os.path.exists(file_path):
                        # Ajouter les en-têtes si le fichier est nouveau
                        with open(file_path, mode="w", newline="", encoding="utf-8") as file:
                            writer = csv.DictWriter(file, fieldnames=patient_data.keys())
                            writer.writeheader()
                            writer.writerow(patient_data)
                    else:
                        # Ajouter une nouvelle ligne au fichier existant
                        with open(file_path, mode="a", newline="", encoding="utf-8") as file:
                            writer = csv.DictWriter(file, fieldnames=patient_data.keys())
                            writer.writerow(patient_data)

                    # Afficher une alerte de succès
                    st.success("Données du patient enregistrées avec succès dans le fichier CSV.")
                except Exception as e:
                    # Afficher une alerte d'erreur en cas de problème
                    st.error(f"Une erreur s'est produite lors de l'enregistrement des données : {str(e)}")
if __name__ == '__main__':
    main()
    