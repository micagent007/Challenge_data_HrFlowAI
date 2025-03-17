import pandas as pd
import ast
import numpy as np

# Charger le fichier d'entraînement original
def load_training_data(file_path):
    df = pd.read_csv(file_path)
    # Convertir les chaînes de caractères en listes
    df['job_ids'] = df['job_ids'].apply(ast.literal_eval)
    df['actions'] = df['actions'].apply(ast.literal_eval)
    return df

# Charger le fichier de prédictions original
def load_prediction_data(file_path):
    return pd.read_csv(file_path)

# Générer de nouvelles données d'entraînement en supprimant le dernier élément
def generate_augmented_training_data(training_df, prediction_df):
    augmented_df = pd.DataFrame(columns=training_df.columns)
    
    # Ajouter les données originales
    augmented_df = pd.concat([augmented_df, training_df.copy()])
    
    # Trouver le plus grand session_id pour commencer à incrémenter à partir de là
    max_session_id = max(
        training_df['session_id'].max(),
        prediction_df['session_id'].max() if 'session_id' in prediction_df.columns else 0
    )
    next_session_id = max_session_id + 1
    
    # Dictionnaire pour garder une trace des nouveaux session_id et de leurs originaux
    session_id_mapping = {}
    
    # Créer de nouvelles séquences en supprimant le dernier élément
    for idx, row in training_df.iterrows():
        if len(row['job_ids']) > 1:  # Vérifier qu'il y a au moins 2 éléments
            # Utiliser un numéro séquentiel pour le nouvel ID de session
            new_session_id = next_session_id
            next_session_id += 1
            
            # Stocker le mapping entre le nouvel ID et l'ID original
            session_id_mapping[new_session_id] = row['session_id']
            
            new_job_ids = row['job_ids'][:-1]  # Supprimer le dernier élément
            new_actions = row['actions'][:-1]  # Supprimer la dernière action
            
            # Ajouter la nouvelle ligne
            new_row = pd.DataFrame({
                'session_id': [new_session_id],
                'job_ids': [new_job_ids],
                'actions': [new_actions]
            })
            augmented_df = pd.concat([augmented_df, new_row], ignore_index=True)
    
    return augmented_df, session_id_mapping

# Générer de nouvelles prédictions correspondant aux données d'entraînement augmentées
def generate_augmented_predictions(training_df, prediction_df, session_id_mapping):
    augmented_pred_df = pd.DataFrame(columns=prediction_df.columns)
    
    # Ajouter les prédictions originales
    augmented_pred_df = pd.concat([augmented_pred_df, prediction_df.copy()])
    
    # Créer un dictionnaire pour une recherche rapide
    training_dict = {}
    for idx, row in training_df.iterrows():
        training_dict[row['session_id']] = row
    
    # Ajouter de nouvelles prédictions pour les séquences augmentées
    for new_session_id, original_session_id in session_id_mapping.items():
        if original_session_id in training_dict:
            original_row = training_dict[original_session_id]
            
            # Utiliser le dernier job_id de la séquence originale comme prédiction
            last_job_id = original_row['job_ids'][-1]
            last_action = original_row['actions'][-1]
            
            new_row = pd.DataFrame({
                'session_id': [new_session_id],
                'job_id': [last_job_id],
                'action': [last_action]
            })
            augmented_pred_df = pd.concat([augmented_pred_df, new_row], ignore_index=True)
    
    return augmented_pred_df

# Exécution complète
def augment_dataset(training_file, prediction_file, output_training_file, output_prediction_file):
    # Charger les données
    training_df = load_training_data(training_file)
    prediction_df = load_prediction_data(prediction_file)
    
    # Générer les données augmentées
    augmented_training_df, session_id_mapping = generate_augmented_training_data(training_df, prediction_df)
    augmented_pred_df = generate_augmented_predictions(training_df, prediction_df, session_id_mapping)
    
    # Enregistrer les nouveaux fichiers
    # Convertir les listes en chaînes pour l'enregistrement
    augmented_training_df['job_ids'] = augmented_training_df['job_ids'].apply(str)
    augmented_training_df['actions'] = augmented_training_df['actions'].apply(str)
    
    augmented_training_df.to_csv(output_training_file, index=False)
    augmented_pred_df.to_csv(output_prediction_file, index=False)
    
    print(f"Données d'entraînement augmentées enregistrées dans {output_training_file}")
    print(f"Prédictions augmentées enregistrées dans {output_prediction_file}")
    print(f"Nombre total de lignes dans le fichier d'entraînement augmenté: {len(augmented_training_df)}")
    print(f"Nombre total de lignes dans le fichier de prédiction augmenté: {len(augmented_pred_df)}")
    print(f"Nombre de nouvelles séquences créées: {len(session_id_mapping)}")
    
    return augmented_training_df, augmented_pred_df

# Exemple d'utilisation
augment_dataset('x_train_Meacfjr.csv', 'y_train_SwJNMSu.csv', 'augmented_training.csv', 'augmented_prediction.csv')