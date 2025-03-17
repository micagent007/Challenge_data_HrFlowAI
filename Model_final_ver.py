import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from ast import literal_eval
import json
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import des classes depuis les fichiers existants
from Model_with_desc import JobRecommendationTransformer, TextEmbedder
from parser import load_and_pad_data, numpy_to_torch, parse_json

class JobRecommendationDataset(Dataset):
    """Dataset pour l'entraînement du modèle de recommandation d'emplois."""
    
    def __init__(self, job_ids, actions, target_job_ids, target_actions):
        """
        Initialise le dataset.
        
        Args:
            job_ids: Tensor des séquences d'IDs de jobs [batch_size, seq_len]
            actions: Tensor des séquences d'actions [batch_size, seq_len]
            target_job_ids: Tensor des job_ids cibles [batch_size]
            target_actions: Tensor des actions cibles [batch_size]
        """
        self.job_ids = job_ids
        self.actions = actions
        self.target_job_ids = target_job_ids
        self.target_actions = target_actions
        
    def __len__(self):
        return len(self.job_ids)
    
    def __getitem__(self, idx):
        return {
            'job_ids': self.job_ids[idx],
            'actions': self.actions[idx],
            'target_job_id': self.target_job_ids[idx],
            'target_action': self.target_actions[idx]
        }

def load_training_data(x_train_path, y_train_path):
    """
    Charge et prépare les données d'entraînement.
    
    Args:
        x_train_path: Chemin vers le fichier x_train_Meacfjr.csv
        y_train_path: Chemin vers le fichier y_train_SwJNMSu.csv
        
    Returns:
        tuple: (job_ids_tensor, actions_tensor, target_job_ids, target_actions)
    """
    # Charger les données d'entrée
    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    
    # S'assurer que les données sont dans le même ordre
    x_train = x_train.sort_values('session_id').reset_index(drop=True)
    y_train = y_train.sort_values('session_id').reset_index(drop=True)
    
    # Convertir les chaînes en listes Python
    x_train['job_ids'] = x_train['job_ids'].apply(literal_eval)
    x_train['actions'] = x_train['actions'].apply(literal_eval)
    
    # Déterminer la longueur maximale pour le padding
    max_length = x_train['job_ids'].apply(len).max()
    
    # Préallouer les tableaux numpy
    n_samples = len(x_train)
    job_ids_array = np.zeros((n_samples, max_length), dtype=np.int32)
    actions_array = np.full((n_samples, max_length), "null", dtype=object)
    
    # Remplir les tableaux avec les données disponibles
    for i, (jobs, acts) in enumerate(zip(x_train['job_ids'], x_train['actions'])):
        job_len = len(jobs)
        job_ids_array[i, :job_len] = jobs
        actions_array[i, :job_len] = acts
    
    # Mapping des actions en indices
    action_to_idx = {
        "null": 0,
        "apply": 1,
        "view": 2
    }
    
    # Convertir les actions en valeurs numériques
    actions_numeric = np.zeros_like(job_ids_array, dtype=np.int32)
    for i in range(actions_array.shape[0]):
        for j in range(actions_array.shape[1]):
            actions_numeric[i, j] = action_to_idx.get(actions_array[i, j], 0)
    
    # Convertir en tenseurs PyTorch
    job_ids_tensor = torch.tensor(job_ids_array, dtype=torch.long)
    actions_tensor = torch.tensor(actions_numeric, dtype=torch.long)
    
    # Préparer les cibles
    target_job_ids = torch.tensor(y_train['job_id'].values, dtype=torch.long)
    target_actions = torch.tensor([action_to_idx[act] for act in y_train['action']], dtype=torch.long)
    
    return job_ids_tensor, actions_tensor, target_job_ids, target_actions

def compute_mrr(predictions, targets):
    """
    Calcule le Mean Reciprocal Rank.
    
    Args:
        predictions: Tensor des top-k prédictions [batch_size, k]
        targets: Tensor des job_ids cibles [batch_size]
        
    Returns:
        float: Score MRR moyen
    """
    batch_size = targets.size(0)
    ranks = torch.zeros(batch_size, device=predictions.device)
    
    for i in range(batch_size):
        # Trouver la position de la cible dans les prédictions
        target = targets[i]
        prediction_ranks = (predictions[i] == target).nonzero(as_tuple=True)[0]
        
        if len(prediction_ranks) > 0:
            # +1 car les rangs commencent à 1
            rank = prediction_ranks[0].item() + 1
            ranks[i] = 1.0 / rank
        # Si la cible n'est pas dans les prédictions, le rang reste 0
    
    return ranks.mean().item()

def compute_action_accuracy(predictions, targets):
    """
    Calcule la précision de la prédiction d'action.
    
    Args:
        predictions: Tensor des prédictions d'action [batch_size]
        targets: Tensor des actions cibles [batch_size]
        
    Returns:
        float: Précision
    """
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total

def compute_combined_score(mrr, accuracy, mrr_weight=0.7, accuracy_weight=0.3):
    """
    Calcule le score combiné.
    
    Args:
        mrr: Score MRR
        accuracy: Précision de l'action
        mrr_weight: Poids du MRR dans le score final
        accuracy_weight: Poids de la précision dans le score final
        
    Returns:
        float: Score combiné
    """
    return mrr * mrr_weight + accuracy * accuracy_weight

def load_job_descriptions(json_file_path):
    """
    Charge les descriptions d'emplois à partir d'un fichier JSON.
    
    Args:
        json_file_path: Chemin vers le fichier JSON contenant les descriptions
        
    Returns:
        dict: Dictionnaire {job_id: description}
    """
    if not os.path.exists(json_file_path):
        print(f"Fichier de descriptions non trouvé: {json_file_path}")
        return {}
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            descriptions = json.load(file)
        return descriptions
    except Exception as e:
        print(f"Erreur lors du chargement des descriptions: {e}")
        return {}
    
def differentiable_mrr_loss(logits, targets, k=10, epsilon=1e-10):
    """
    Une approximation différentiable de la perte MRR.
    
    Args:
        logits: [batch_size, vocab_size] - scores pour chaque item
        targets: [batch_size] - indices des items cibles
        k: nombre d'items top-k à considérer
        epsilon: petit nombre pour éviter la division par zéro
        
    Returns:
        Tensor: perte MRR différentiable (à minimiser)
    """
    batch_size = logits.size(0)
    vocab_size = logits.size(1)
    
    # Créer des scores softmax pour approximer le classement
    scores = torch.softmax(logits, dim=1)
    
    # Créer un masque où chaque position indique si l'élément est la cible
    target_mask = torch.zeros_like(scores)
    for i in range(batch_size):
        target_mask[i, targets[i]] = 1.0
    
    # Calculer un classement "doux" avec une fonction sigmoïde
    # Au lieu d'un tri discret, nous calculons combien d'éléments ont un score plus élevé
    # que la cible de manière différentiable
    target_scores = torch.sum(scores * target_mask, dim=1, keepdim=True)  # [batch_size, 1]
    
    # Compter combien d'éléments ont un score plus élevé (de manière douce)
    comparison = torch.sigmoid((scores - target_scores) * 10)  # Le facteur 10 rend la sigmoïde plus abrupte
    
    # Soustraire la comparaison de la cible avec elle-même
    comparison = comparison * (1 - target_mask)
    
    # Calculer le rang approximatif (1-based) de manière différentiable
    soft_rank = 1.0 + torch.sum(comparison, dim=1)  # [batch_size]
    
    # Calculer le MRR approximatif (1/rank)
    mrr = 1.0 / (soft_rank + epsilon)
    
    # Transformer en perte (à minimiser)
    loss = -torch.mean(mrr)
    
    return loss

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=10, 
                job_descriptions_dict=None, description_encoder=None):
    """
    Entraîne le modèle de recommandation avec MRR dans la fonction de perte.
    """
    model.to(device)
    
    # Fonction de perte pour les actions (classification)
    action_criterion = nn.CrossEntropyLoss()
    
    # Poids pour combiner les pertes
    mrr_weight = 0.7
    action_weight = 0.3
    
    history = {
        'train_mrr': [],
        'train_action_acc': [],
        'train_combined': [],
        'val_mrr': [],
        'val_action_acc': [],
        'val_combined': []
    }
    
    for epoch in range(num_epochs):
        # Mode entraînement
        model.train()
        train_mrr = 0.0
        train_action_acc = 0.0
        train_combined = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            # Déplacer les données sur le dispositif
            job_ids = batch['job_ids'].to(device)
            actions = batch['actions'].to(device)
            target_job_ids = batch['target_job_id'].to(device)
            target_actions = batch['target_action'].to(device)
            
            # Forward pass
            # Modifier cette partie pour obtenir les logits avant topk
            embedded_input = model.job_embedding(job_ids) + model.action_embedding(actions)
            
            # Si vous utilisez des descriptions
            if job_descriptions_dict is not None and description_encoder is not None:
                # Code pour ajouter les embeddings de description (comme dans votre modèle)
                # ...
                # Ajouter aux embeddings
                # embedded_input = embedded_input + projected_desc_emb
                pass
                
            # Obtenir la sortie du transformer
            transformer_output = model.transformer_encoder(embedded_input.transpose(0, 1)).transpose(0, 1)
            last_hidden = transformer_output[:, -1, :]
            
            # Logits pour jobs et actions
            job_logits = model.job_predictor(last_hidden)  # [batch_size, job_vocab_size]
            action_logits = model.action_predictor(last_hidden)  # [batch_size, action_vocab_size]
            
            # Calculer top10 pour les métriques (pas pour le gradient)
            job_logits_copy = job_logits.clone()
            job_logits_copy[:, 0] = float('-inf')  # Exclure le padding
            top10_job_ids = torch.topk(job_logits_copy, 10, dim=-1).indices
            
            # Perte pour l'action (classification standard)
            action_loss = action_criterion(action_logits, target_actions)
            
            # Perte pour la recommandation de job (approximation de MRR)
            # Utiliser une approche de similarité avec les cibles
            batch_size = job_ids.size(0)
            
            # Créer un vecteur one-hot ou un masque pour les cibles
            job_targets_mask = torch.zeros_like(job_logits)
            for i in range(batch_size):
                job_targets_mask[i, target_job_ids[i]] = 1.0
            
            # Perte basée sur la similarité (version différentiable de MRR)
            # On utilise une perte de type multi-classe adaptée pour favoriser le classement
            # Option 1: CrossEntropy sur les logits softmax (recommandé pour commencer)
            #job_loss = nn.CrossEntropyLoss()(job_logits, target_job_ids)
            
            # Option 2: Perte de classement personnalisée inspirée de BPR ou ListNet
            # Plus proche de l'objectif MRR mais plus complexe
            job_softmax = torch.softmax(job_logits, dim=1)
            job_loss = -torch.log(torch.sum(job_softmax * job_targets_mask, dim=1)).mean()
            
            # Option 3: Perte basée sur ListMLE (List Maximum Likelihood Estimation)
            # Encore plus proche de l'optimisation de rang mais plus complexe à implémenter
            # Dans la boucle d'entraînement
            #job_loss = differentiable_mrr_loss(job_logits, target_job_ids)
            
            # Combiner les pertes
            loss = mrr_weight * job_loss + action_weight * action_loss
            
            # Métriques pour suivi (pas pour gradient)
            batch_mrr = compute_mrr(top10_job_ids, target_job_ids)
            print(f"job_loss = {job_loss}")
            print(f"batch_mrr = {batch_mrr}")
            
            if action_logits.dim() > 1:
                action_preds = torch.argmax(action_logits, dim=-1)
            else:
                action_preds = action_logits
                
            batch_action_acc = compute_action_accuracy(action_preds, target_actions)
            batch_combined = compute_combined_score(batch_mrr, batch_action_acc, 
                                                  mrr_weight=mrr_weight, 
                                                  accuracy_weight=action_weight)
            
            # Backward pass et optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumuler les métriques
            train_mrr += batch_mrr
            train_action_acc += batch_action_acc
            train_combined += batch_combined
            train_batches += 1
            
            # Mise à jour de la barre de progression
            pbar.set_postfix({
                'MRR': batch_mrr,
                'Action Acc': batch_action_acc,
                'Combined': batch_combined,
                'Loss': loss.item()
            })
        
        # Moyennes d'entraînement
        train_mrr /= train_batches
        train_action_acc /= train_batches
        train_combined /= train_batches
        
        # Mode évaluation
        model.eval()
        val_mrr = 0.0
        val_action_acc = 0.0
        val_combined = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # Déplacer les données sur le dispositif
                job_ids = batch['job_ids'].to(device)
                actions = batch['actions'].to(device)
                target_job_ids = batch['target_job_id'].to(device)
                target_actions = batch['target_action'].to(device)
                
                # Forward pass
                top10_job_ids, action_preds = model(
                    job_ids, actions, job_descriptions_dict, description_encoder
                )
                
                # FIX: Même vérification pour la validation
                if action_preds.dim() == 1:
                    # Si c'est déjà un indice, on l'utilise directement
                    action_indices = action_preds
                else:
                    # Sinon, on prend l'indice avec la plus grande valeur
                    action_indices = torch.argmax(action_preds, dim=-1)
                
                # Métriques
                batch_mrr = compute_mrr(top10_job_ids, target_job_ids)
                batch_action_acc = compute_action_accuracy(action_indices, target_actions)
                batch_combined = compute_combined_score(batch_mrr, batch_action_acc)
                
                # Accumuler les métriques
                val_mrr += batch_mrr
                val_action_acc += batch_action_acc
                val_combined += batch_combined
                val_batches += 1
        
        # Moyennes de validation
        val_mrr /= val_batches
        val_action_acc /= val_batches
        val_combined /= val_batches
        
        # Enregistrer les métriques
        history['train_mrr'].append(train_mrr)
        history['train_action_acc'].append(train_action_acc)
        history['train_combined'].append(train_combined)
        history['val_mrr'].append(val_mrr)
        history['val_action_acc'].append(val_action_acc)
        history['val_combined'].append(val_combined)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train - MRR: {train_mrr:.4f}, Action Acc: {train_action_acc:.4f}, Combined: {train_combined:.4f}")
        print(f"Val - MRR: {val_mrr:.4f}, Action Acc: {val_action_acc:.4f}, Combined: {val_combined:.4f}")
    
    return history

def main():
    # Paramètres
    x_train_path = 'augmented_training.csv'
    y_train_path = 'augmented_prediction.csv'
    #job_descriptions_path = 'job_descriptions.json'  # Optionnel
    
    batch_size = 64
    num_epochs = 15
    learning_rate = 0.1
    test_size = 0.1
    random_state = 42
    
    # Vérifier la disponibilité du GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Charger les données
    print("Chargement des données...")
    job_ids_tensor, actions_tensor, target_job_ids, target_actions = load_training_data(x_train_path, y_train_path)
    
    # Obtenir la taille du vocabulaire des jobs
    job_vocab_size = job_ids_tensor.max().item() + 1
    action_vocab_size = 3  # null (0), apply (1), view (2)
    
    print(f"Taille du vocabulaire des jobs: {job_vocab_size}")

    # Fractionner les données en ensembles d'entraînement et de validation
    train_indices, val_indices = train_test_split(
        np.arange(len(job_ids_tensor)),
        test_size=test_size,
        random_state=random_state
    )
    
    # Créer les ensembles d'entraînement et de validation
    train_job_ids = job_ids_tensor[train_indices]
    train_actions = actions_tensor[train_indices]
    train_target_job_ids = target_job_ids[train_indices]
    train_target_actions = target_actions[train_indices]
    
    val_job_ids = job_ids_tensor[val_indices]
    val_actions = actions_tensor[val_indices]
    val_target_job_ids = target_job_ids[val_indices]
    val_target_actions = target_actions[val_indices]
    
    # Créer les datasets
    train_dataset = JobRecommendationDataset(
        train_job_ids, train_actions, train_target_job_ids, train_target_actions
    )
    val_dataset = JobRecommendationDataset(
        val_job_ids, val_actions, val_target_job_ids, val_target_actions
    )
    
    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialiser le modèle
    print("Initialisation du modèle...")
    model = JobRecommendationTransformer(
        job_vocab_size=job_vocab_size,
        action_vocab_size=action_vocab_size,
        embed_dim=256,
        transformer_dim=512,
        n_heads=4,
        n_layers=2,
        desc_embed_dim=384,  # Dimension des embeddings du modèle de description
        weight_job=7,
        weight_actions=3,
        weight_desc=5
    )
    
    # Charger le modèle d'embedding de texte et les descriptions (optionnel)
    fichier = "job_listings/job_listings.json"  # Remplace par le chemin de ton fichier JSON
    sortie = "job_listings.txt"  # Remplace par le chemin de sortie
    job_descriptions_dict = parse_json(fichier, sortie)
    description_encoder = TextEmbedder()
    
    """if os.path.exists(job_descriptions_path):
        print("Chargement des descriptions d'emplois...")
        job_descriptions_dict = load_job_descriptions(job_descriptions_path)
        
        print("Initialisation de l'encodeur de texte...")
        description_encoder = TextEmbedder()
        print(f"Dimension des embeddings de description: {description_encoder.embed_dim}")"""
    
    # Initialiser l'optimiseur
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Entraîner le modèle
    print("Début de l'entraînement...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        job_descriptions_dict=job_descriptions_dict,
        description_encoder=description_encoder
    )
    
    # Sauvegarder le modèle
    print("Sauvegarde du modèle...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'job_vocab_size': job_vocab_size,
        'action_vocab_size': action_vocab_size
    }, 'job_recommendation_model.pt')
    
    print("Entraînement terminé!")
    
    # Évaluer le modèle sur l'ensemble de validation
    model.eval()
    val_mrr = 0.0
    val_action_acc = 0.0
    val_combined = 0.0
    val_batches = 0
    
    print("Évaluation finale du modèle...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            # Déplacer les données sur le dispositif
            job_ids = batch['job_ids'].to(device)
            actions = batch['actions'].to(device)
            target_job_ids = batch['target_job_id'].to(device)
            target_actions = batch['target_action'].to(device)
            
            # Forward pass
            top10_job_ids, action_preds = model(
                job_ids, actions, job_descriptions_dict, description_encoder
            )
            
            # Métriques
            batch_mrr = compute_mrr(top10_job_ids, target_job_ids)
            batch_action_acc = compute_action_accuracy(action_preds, target_actions)
            batch_combined = compute_combined_score(batch_mrr, batch_action_acc)
            
            # Accumuler les métriques
            val_mrr += batch_mrr
            val_action_acc += batch_action_acc
            val_combined += batch_combined
            val_batches += 1
    
    # Moyennes de validation
    val_mrr /= val_batches
    val_action_acc /= val_batches
    val_combined /= val_batches
    
    print("\nRésultats finaux:")
    print(f"MRR: {val_mrr:.4f}")
    print(f"Action Accuracy: {val_action_acc:.4f}")
    print(f"Score combiné: {val_combined:.4f}")
    
    # Visualisation des métriques (optionnel)
    try:
        import matplotlib.pyplot as plt
        
        epochs = range(1, num_epochs + 1)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(epochs, history['train_mrr'], label='Train')
        plt.plot(epochs, history['val_mrr'], label='Validation')
        plt.title('Mean Reciprocal Rank (MRR)')
        plt.xlabel('Epochs')
        plt.ylabel('MRR')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs, history['train_action_acc'], label='Train')
        plt.plot(epochs, history['val_action_acc'], label='Validation')
        plt.title('Action Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(epochs, history['train_combined'], label='Train')
        plt.plot(epochs, history['val_combined'], label='Validation')
        plt.title('Combined Score (70% MRR + 30% Acc)')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
        
        print("\nGraphiques des métriques enregistrés dans 'training_metrics.png'")
    except ImportError:
        print("\nMatplotlib non disponible. Les graphiques n'ont pas été générés.")

if __name__ == "__main__":
    main()