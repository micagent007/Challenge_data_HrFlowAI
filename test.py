import numpy as np
import pandas as pd
import torch
from ast import literal_eval

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
    
    # Pour les actions, on utilise les valeurs spécifiques demandées
    # Mapping personnalisé: "null" -> -1, "apply" -> -2, "view" -> -3
    action_to_idx = {
        "null": -1,
        "apply": -2,
        "view": -3
    }
    
    # Convertir les actions en valeurs numériques spécifiques
    actions_numeric = np.zeros_like(job_ids_array, dtype=np.int32)
    for i in range(actions_array.shape[0]):
        for j in range(actions_array.shape[1]):
            actions_numeric[i, j] = action_to_idx[actions_array[i, j]]
    
    actions_tensor = torch.tensor(actions_numeric, dtype=torch.long)
    
    return job_ids_tensor, actions_tensor, action_to_idx

# Exemple d'utilisation
# Charger et padder les données
job_ids_array, actions_array = load_and_pad_data('x_train_Meacfjr.csv', pad_length=None)

# Convertir en tenseurs PyTorch
job_ids_tensor, actions_tensor, action_mapping = numpy_to_torch(job_ids_array, actions_array)

print("Job IDs shape:", job_ids_tensor.shape)
print("Actions shape:", actions_tensor.shape)
print("Action mapping:", action_mapping)

print(job_ids_tensor[2], actions_tensor[2], actions_array)
print(torch.unique(job_ids_tensor))