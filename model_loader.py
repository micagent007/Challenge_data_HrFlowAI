import torch
import numpy as np
import pandas as pd
import json
from Model_with_desc import JobRecommendationTransformer, TextEmbedder
from parser import load_and_pad_data, numpy_to_torch, parse_json
from loader import save_predictions_to_csv

def load_model(model_path, device=None):
    """
    Charge un modèle entraîné.
    
    Args:
        model_path: Chemin vers le fichier de modèle sauvegardé
        device: Dispositif sur lequel charger le modèle (None pour auto-détection)
        
    Returns:
        tuple: (model, job_vocab_size, action_vocab_size)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger le checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extraire les paramètres du modèle
    job_vocab_size = checkpoint['job_vocab_size']
    action_vocab_size = checkpoint['action_vocab_size']
    
    # Créer et charger le modèle
    model = JobRecommendationTransformer(
        job_vocab_size=job_vocab_size,
        action_vocab_size=action_vocab_size
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, job_vocab_size, action_vocab_size

def prepare_sequence_for_inference(job_ids, actions, action_to_idx=None):
    """
    Prépare une séquence d'historique pour l'inférence.
    
    Args:
        job_ids: Liste des IDs d'emplois
        actions: Liste des actions ('view', 'apply')
        action_to_idx: Dictionnaire pour mapper les actions aux indices
        
    Returns:
        tuple: (job_ids_tensor, actions_tensor)
    """
    if action_to_idx is None:
        action_to_idx = {
            "null": 0,
            "apply": 1,
            "view": 2
        }
    
    # Convertir en tableaux numpy
    job_ids_array = np.array(job_ids, dtype=np.int32)
    
    # Convertir les actions en indices
    actions_numeric = np.array([action_to_idx.get(action, 0) for action in actions], dtype=np.int32)
    
    # Convertir en tenseurs PyTorch
    job_ids_tensor = torch.tensor(job_ids_array, dtype=torch.long).unsqueeze(0)
    actions_tensor = torch.tensor(actions_numeric, dtype=torch.long).unsqueeze(0)
    
    return job_ids_tensor, actions_tensor

def get_recommendations(model, job_ids, actions, job_descriptions_dict=None, description_encoder=None, device=None):
    """
    Obtient des recommandations d'emplois et prédit la prochaine action.
    
    Args:
        model: Modèle JobRecommendationTransformer
        job_ids: Liste des IDs d'emplois
        actions: Liste des actions ('view', 'apply')
        job_descriptions_dict: Dictionnaire des descriptions d'emplois (optionnel)
        description_encoder: Encodeur de texte (optionnel)
        device: Dispositif sur lequel exécuter l'inférence (None pour auto-détection)
        
    Returns:
        tuple: (top10_job_ids, predicted_action)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Préparer les données d'entrée
    job_ids_tensor, actions_tensor = prepare_sequence_for_inference(job_ids, actions)
    job_ids_tensor = job_ids_tensor.to(device)
    actions_tensor = actions_tensor.to(device)
    
    # Mettre le modèle en mode évaluation
    model.eval()
    
    # Obtenir les prédictions
    with torch.no_grad():
        top10_job_ids, action_pred = model(
            job_ids_tensor,
            actions_tensor,
            job_descriptions_dict,
            description_encoder
        )
    
    # Convertir les prédictions en listes
    recommended_job_ids = top10_job_ids.cpu().numpy()[0].tolist()
    
    # Mapper l'action prédite à une chaîne
    idx_to_action = {
        1: "apply",
        2: "view"
    }
    predicted_action = idx_to_action.get(action_pred.cpu().numpy()[0].item(), "unknown")
    
    return recommended_job_ids, predicted_action

def main():
    # Paramètres
    model_path = 'job_recommendation_model.pt'
    test_file_path = 'x_test_jCBBNP2.csv'
    
    job_ids_array, actions_array = load_and_pad_data(test_file_path)
    job_ids_tensor, actions_tensor, action_to_idx = numpy_to_torch(job_ids_array, actions_array)

    user_job_ids, user_actions = job_ids_tensor, actions_tensor

    # Exemple d'historique d'un utilisateur
    user_job_ids = [305, 299, 300, 290, 282, 274, 264, 261]
    user_actions = ['view', 'view', 'view', 'view', 'view', 'view', 'view', 'view']
    
    # Vérifier la disponibilité du GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Charger le modèle
    print("Chargement du modèle...")
    model, job_vocab_size, action_vocab_size = load_model(model_path, device)
    
    # Charger le modèle d'embedding de texte et les descriptions (optionnel)
    fichier = "job_listings/job_listings.json"  # Remplace par le chemin de ton fichier JSON
    sortie = "job_listings.txt"  # Remplace par le chemin de sortie
    job_descriptions_dict = parse_json(fichier, sortie)
    description_encoder = TextEmbedder()
    
    # Obtenir des recommandations
    print("Génération des recommandations...")
    recommended_job_ids, predicted_action = get_recommendations(
        model, user_job_ids, user_actions, job_descriptions_dict, description_encoder, device
    )

    save_predictions_to_csv(recommended_job_ids, predicted_action, output_file="predictions_1.csv")
    
    print("\nRecommandations d'emplois:")
    for i, job_id in enumerate(recommended_job_ids):
        print(f"{i+1}. Job ID: {job_id}")
    
    print(f"\nAction prédite: {predicted_action}")

if __name__ == "__main__":
    main()