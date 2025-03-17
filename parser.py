import json
import re
import pandas as pd
import numpy as np
import torch
from ast import literal_eval

def parse_json(file_path, output_file):
    """Lit un fichier JSON, nettoie les artefacts et écrit son contenu dans un fichier texte."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Nettoyage des artefacts _X000D_
        #clean_content = content.replace("_x000D_" + " " + "_x000D_" + " " + "_x000D_" + " " + "-", "\\n" + "-")
        #clean_content = content.replace("_x000D_" + " " + "_x000D_" + " " + "-", "\\n" + "-")
        #clean_content = content.replace("_x000D_" + " " + "-", "\\n" + "-")
        clean_content = content.replace("_x000D_", "\\n")
    
        
        # Vérification du JSON avant chargement
        try:
            data = json.loads(clean_content)
        except json.JSONDecodeError as e:
            print(f"Erreur de décodage JSON après nettoyage : {e}")
            return
        
        with open(output_file, 'w', encoding='utf-8') as out_file:
            json.dump(data, out_file, indent=4, ensure_ascii=False)
        
        print(f"Contenu nettoyé et écrit dans {output_file}")
        return data
    except FileNotFoundError:
        print("Fichier non trouvé.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

# Charger les données
def load_and_pad_data(csv_file, pad_length=None):
    # Lire le CSV
    df = pd.read_csv(csv_file)
    
    # Convertir les chaînes en listes Python
    df['job_ids'] = df['job_ids'].apply(literal_eval)
    df['actions'] = df['actions'].apply(literal_eval)
    
    # Déterminer la longueur maximale si non spécifiée
    if pad_length is None:
        pad_length = max(df['job_ids'].apply(len).max(), df['actions'].apply(len).max())
    
    # Préallouer les tableaux numpy
    job_ids_array = np.full((len(df), pad_length), -1, dtype=np.int32)
    actions_array = np.full((len(df), pad_length), "null", dtype=object)
    
    # Remplir les tableaux avec les données disponibles
    for i, (jobs, acts) in enumerate(zip(df['job_ids'], df['actions'])):
        job_len = len(jobs)
        act_len = len(acts)
        
        job_ids_array[i, :job_len] = jobs
        actions_array[i, :act_len] = acts
    
    return job_ids_array, actions_array

# Convertir en tenseurs PyTorch 
def numpy_to_torch(job_ids_array, actions_array):
    # Pour les job_ids, on peut directement convertir en tenseur
    job_ids_tensor = torch.tensor(job_ids_array, dtype=torch.long)
    job_ids_tensor += 1
    
    # Pour les actions, on utilise les valeurs spécifiques demandées
    # Mapping personnalisé: "null" -> -1, "apply" -> -2, "view" -> -3
    action_to_idx = {
        "null": 0,
        "apply": 1,
        "view": 2
    }
    
    # Convertir les actions en valeurs numériques spécifiques
    actions_numeric = np.zeros_like(job_ids_array, dtype=np.int32)
    for i in range(actions_array.shape[0]):
        for j in range(actions_array.shape[1]):
            actions_numeric[i, j] = action_to_idx[actions_array[i, j]]
    
    actions_tensor = torch.tensor(actions_numeric, dtype=torch.long)
    
    return job_ids_tensor, actions_tensor, action_to_idx