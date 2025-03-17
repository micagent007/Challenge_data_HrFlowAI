import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np

class TextEmbedder:
    """Classe pour convertir du texte en embeddings vectoriels."""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialise le modèle d'embedding.
        
        Args:
            model_name (str): Nom du modèle pré-entraîné à utiliser.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.embed_dim = self.model.config.hidden_size
        
    def mean_pooling(self, model_output, attention_mask):
        """Applique le mean pooling sur les token embeddings."""
        token_embeddings = model_output[0]  # Dernier hidden state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def encode(self, texts, batch_size=32):
        """
        Convertit une liste de textes en embeddings.
        
        Args:
            texts (list): Liste de descriptions textuelles
            batch_size (int): Taille des lots pour le traitement
            
        Returns:
            np.ndarray: Matrice d'embeddings normalisés
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            # Préparer le batch
            batch_texts = texts[i:i+batch_size]
            
            # Tokeniser
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Calculer les embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            # Appliquer le mean pooling
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normaliser les embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Convertir en numpy et stocker
            all_embeddings.append(embeddings.cpu().numpy())
            
        # Concaténer tous les embeddings
        return np.vstack(all_embeddings)

class JobRecommendationTransformer(nn.Module):
    def __init__(self, job_vocab_size=27369, action_vocab_size=3, embed_dim=128, transformer_dim=256, n_heads=4, n_layers=2, desc_embed_dim=384, weight_job=10, weight_actions=5, weight_desc=10):
        """
        Initialise le modèle de recommandation de jobs.
        
        Args:
            job_vocab_size: Taille du vocabulaire des jobs (y compris padding)
            action_vocab_size: Taille du vocabulaire des actions (y compris padding)
                               0: padding, 1: apply, 2: view
            embed_dim: Dimension des embeddings
            transformer_dim: Dimension interne du transformer
            n_heads: Nombre de têtes d'attention
            n_layers: Nombre de couches du transformer
            desc_embed_dim: Dimension des embeddings de description
        """
        self.weight_job = weight_job
        self.weight_actions = weight_actions
        self.weight_desc = weight_desc

        super().__init__()
        
        # Dimensions d'embedding
        self.embed_dim = embed_dim
        self.desc_embed_dim = desc_embed_dim  # Dimension du modèle d'embedding
        self.padding_idx = 0  # Valeur utilisée pour le padding
        
        # Créer des embeddings distincts pour jobs et actions avec padding_idx=0
        self.job_embedding = nn.Embedding(job_vocab_size, embed_dim, padding_idx=self.padding_idx)
        self.action_embedding = nn.Embedding(action_vocab_size, embed_dim, padding_idx=self.padding_idx)
        
        # Projection pour les embeddings de description
        self.desc_projection = nn.Linear(desc_embed_dim, embed_dim)
        
        # Couche de normalisation pour les embeddings combinés
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Transformer Encoder pour le traitement de séquence
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Couches de sortie
        self.job_predictor = nn.Linear(embed_dim, job_vocab_size)
        self.action_predictor = nn.Linear(embed_dim, action_vocab_size)
    
    def forward(self, job_ids, actions, job_descriptions_dict=None, description_encoder=None):
        """
        Forward pass du modèle avec chargement automatique des descriptions basé sur les job_ids.
        
        Args:
            job_ids: Tensor de sequences d'IDs de jobs [batch_size, seq_len]
            actions: Tensor de sequences d'actions [batch_size, seq_len]
                    0: padding, 1: apply, 2: view
            job_descriptions_dict: Dictionnaire {job_id: description}
            description_encoder: Encodeur TextEmbedder pour convertir les descriptions en embeddings
        
        Returns:
            tuple: (top10_job_ids, action_pred)
                  top10_job_ids: [batch_size, 10] - IDs des 10 jobs les plus probables
                  action_pred: [batch_size] - Actions prédites (1: apply, 2: view)
        """
        batch_size, seq_len = job_ids.shape
        
        # Créer un masque pour ignorer les positions de padding (0)
        mask = (job_ids != 0)
        
        # Embed job IDs et actions
        job_emb = self.job_embedding(job_ids)
        action_emb = self.action_embedding(actions)
        
        # Normaliser individuellement chaque type d'embedding
        job_emb = torch.nn.functional.normalize(job_emb, p=2, dim=-1)
        action_emb = torch.nn.functional.normalize(action_emb, p=2, dim=-1)
        
        # Combiner les embeddings de base avec les poids
        x = self.weight_job * job_emb + self.weight_actions * action_emb
        
        # Ajouter les embeddings de description si le dictionnaire et l'encodeur sont fournis
        if job_descriptions_dict is not None and description_encoder is not None:
            # Obtenir les descriptions pour chaque job_id non masqué
            batch_descriptions = []
            
            # Pour chaque exemple dans le batch
            for i in range(batch_size):
                sequence_desc_embeddings = []
                
                # Pour chaque position dans la séquence
                for j in range(seq_len):
                    job_id = job_ids[i, j].item()
                    
                    # Si c'est un padding (0), utiliser un embedding zéro
                    if job_id == 0:
                        # Créer un embedding zéro de la même dimension que les autres embeddings
                        if hasattr(description_encoder, 'embed_dim'):
                            zero_embedding = torch.zeros(description_encoder.embed_dim)
                        else:
                            # Utiliser une dimension par défaut si l'attribut n'existe pas
                            zero_embedding = torch.zeros(384)  # Dimension typique pour BERT/DistilBERT
                        sequence_desc_embeddings.append(zero_embedding)
                    else:
                        # Récupérer la description du job_id, ou une chaîne vide si non trouvée
                        description = job_descriptions_dict.get(str(job_id), "")
                        
                        # Encoder la description (un seul élément)
                        desc_embedding = description_encoder.encode([description])[0]
                        sequence_desc_embeddings.append(torch.tensor(desc_embedding))
                
                # Empiler les embeddings de description pour cette séquence
                batch_descriptions.append(torch.stack(sequence_desc_embeddings))
            
            # Empiler les embeddings pour tout le batch
            desc_embeddings = torch.stack(batch_descriptions).to(job_ids.device)
            
            # Projeter les embeddings de description dans l'espace d'embedding du modèle
            projected_desc_emb = self.desc_projection(desc_embeddings)
            
            # Normaliser les embeddings de description projetés
            projected_desc_emb = torch.nn.functional.normalize(projected_desc_emb, p=2, dim=-1)
            
            # Ajouter aux autres embeddings
            x = x + self.weight_desc * projected_desc_emb
        
        # Normaliser l'embedding combiné final
        x = self.layer_norm(x)
        
        # Transposer pour le Transformer (seq_len, batch_size, embed_dim)
        x = x.transpose(0, 1)
        
        # Transformer Encoder avec masque de padding
        x = self.transformer_encoder(x, src_key_padding_mask=~mask)
        
        # Retransposer (batch_size, seq_len, embed_dim)
        x = x.transpose(0, 1)
        
        # Prédire les prochains job_ids et actions
        job_pred = self.job_predictor(x[:, -1, :])
        action_pred = self.action_predictor(x[:, -1, :])
        
        # Obtenir les 10 meilleures prédictions de jobs (ignorer l'indice 0 qui est le padding)
        job_pred[:, 0] = float('-inf')  # Assurer que le padding (indice 0) n'est jamais recommandé
        top10_job_ids = torch.topk(job_pred, 10, dim=-1).indices
        
        # Obtenir l'action la plus probable (sans inclure le padding)
        action_pred[:, 0] = float('-inf')  # Assurer que le padding (indice 0) n'est jamais prédit
        action_pred = torch.argmax(action_pred, dim=-1)
        
        return top10_job_ids, action_pred