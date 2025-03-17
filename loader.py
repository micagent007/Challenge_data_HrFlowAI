import pandas as pd
import numpy as np
import os
import torch
import json

def save_predictions_to_csv(top10_job_ids, action_pred, output_file="predictions.csv"):
    """
    Convertit les sorties du modèle en fichier CSV avec des listes formatées.
    
    Args:
        top10_job_ids: Tensor des 10 job_ids les plus probables [batch_size, 10]
        action_pred: Tensor des actions prédites [batch_size]
        output_file: Nom du fichier CSV de sortie
        
    Returns:
        str: Chemin du fichier CSV créé
    """
    top10_job_ids -= 1
    
    # Convertir les tensors en numpy arrays si nécessaire
    if torch.is_tensor(top10_job_ids):
        top10_job_ids = top10_job_ids.cpu().numpy().tolist()
    if torch.is_tensor(action_pred):
        action_pred = action_pred.cpu().numpy().tolist()
        
    batch_size = len(action_pred)
    
    # Créer la colonne session_id (entiers de 0 à batch_size-1)
    session_ids = list(range(batch_size))
    
    # Préparer les données
    data = []
    for i in range(batch_size):
        # Convertir les valeurs numériques des actions en labels
        if action_pred[i] == 1:
            action_labels = 'apply'
        elif action_pred[i] == 2:
            action_labels = 'view'
        else:
            action_labels = 'null'
            
        # Ajouter à la liste de données
        data.append({
            'session_id': session_ids[i],
            'job_ids': top10_job_ids[i],
            'actions': action_labels
        })
    
    # Créer un DataFrame
    df = pd.DataFrame(data)
    
    # Sauvegarder en CSV
    df.to_csv(output_file, index=False, quoting=1)  # quoting=1 pour mettre des guillemets autour des champs avec des virgules
    
    print(f"Prédictions sauvegardées dans {os.path.abspath(output_file)}")
    return output_file
