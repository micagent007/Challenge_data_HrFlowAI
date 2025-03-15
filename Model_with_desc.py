import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer

# Chargement du modèle pré-entraîné pour encoder les descriptions
description_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def encode_descriptions(descriptions):
    return torch.tensor(description_encoder.encode(descriptions), dtype=torch.float)

class JobRecommendationTransformer(nn.Module):
    def __init__(self, job_vocab_size=27369, action_vocab_size=2, embed_dim=128, transformer_dim=256, n_heads=4, n_layers=2):
        super().__init__()
        
        # Embeddings
        self.job_embedding = nn.Embedding(job_vocab_size, embed_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(action_vocab_size, embed_dim, padding_idx=0)
        
        # Transformer Encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layers
        self.job_predictor = nn.Linear(embed_dim, job_vocab_size)
        self.action_predictor = nn.Linear(embed_dim, action_vocab_size)
    
    def forward(self, job_ids, actions, job_desc_emb):
        # Embed job IDs and actions
        job_emb = self.job_embedding(job_ids)
        action_emb = self.action_embedding(actions)
        
        # Combine embeddings
        x = job_emb + action_emb + job_desc_emb
        
        # Masking for padding
        mask = (job_ids != 0).to(torch.bool)
        
        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=~mask)
        
        # Predict next job_ids and action
        job_pred = self.job_predictor(x[:, -1, :])
        action_pred = self.action_predictor(x[:, -1, :])
        
        # Get top 10 job predictions
        top10_job_ids = torch.topk(job_pred, 10, dim=-1).indices
        
        # Get most probable action
        action_pred = torch.argmax(action_pred, dim=-1)
        
        return top10_job_ids, action_pred

# Dataset personnalisé
class JobDataset(Dataset):
    def __init__(self, job_ids, actions, job_descriptions, next_jobs, next_actions):
        self.job_ids = [torch.tensor(j) for j in job_ids]
        self.actions = [torch.tensor(a) for a in actions]
        
        # Encoder les descriptions
        self.job_descriptions = [encode_descriptions(d) for d in job_descriptions]
        
        self.next_jobs = torch.tensor(next_jobs)
        self.next_actions = torch.tensor(next_actions)
        
        # Padding des séquences
        self.job_ids = pad_sequence(self.job_ids, batch_first=True, padding_value=0)
        self.actions = pad_sequence(self.actions, batch_first=True, padding_value=0)
        self.job_descriptions = pad_sequence(self.job_descriptions, batch_first=True, padding_value=0)
    
    def __len__(self):
        return len(self.job_ids)
    
    def __getitem__(self, idx):
        return self.job_ids[idx], self.actions[idx], self.job_descriptions[idx], self.next_jobs[idx], self.next_actions[idx]

# Fonction d'entraînement
def train_model(model, dataloader, epochs=5, lr=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_job = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for job_ids, actions, job_desc_emb, next_jobs, next_actions in dataloader:
            job_ids, actions, job_desc_emb, next_jobs, next_actions = job_ids.to(device), actions.to(device), job_desc_emb.to(device), next_jobs.to(device), next_actions.to(device)
            
            optimizer.zero_grad()
            top10_job_ids, action_pred = model(job_ids, actions, job_desc_emb)
            
            loss_job = criterion_job(model.job_predictor(job_ids[:, -1, :]), next_jobs)
            loss_action = criterion_action(action_pred, next_actions)
            loss = loss_job + loss_action
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Initialisation du modèle
model = JobRecommendationTransformer()

# Exemple d'utilisation après entraînement
model.load_state_dict(torch.load("job_recommendation_model.pth"))
model.eval()

# Exemple de session
job_ids = torch.tensor([[305, 299, 300, 290, 282, 274, 264, 261]])  # IDs des offres consultées
actions = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]])  # Actions correspondantes (0: view, 1: apply)
job_descriptions = ["Software Engineer", "Data Scientist", "ML Engineer", "AI Researcher", "Backend Developer", "Frontend Developer", "DevOps Engineer", "Cybersecurity Analyst"]
job_desc_emb = encode_descriptions(job_descriptions).unsqueeze(0)  # Encodage des descriptions

# Padding si nécessaire
job_ids = pad_sequence([job_ids], batch_first=True, padding_value=0)
actions = pad_sequence([actions], batch_first=True, padding_value=0)
job_desc_emb = pad_sequence([job_desc_emb], batch_first=True, padding_value=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
job_ids, actions, job_desc_emb = job_ids.to(device), actions.to(device), job_desc_emb.to(device)

# Prédiction
with torch.no_grad():
    top10_job_ids, predicted_action = model(job_ids, actions, job_desc_emb)

print("Top 10 job recommendations:", top10_job_ids.tolist())
print("Predicted next action:", predicted_action.item())