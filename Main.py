import Model_with_desc
import parser
import loader

# Exemple d'utilisation
# Initialiser le modèle
description_encoder = Model_with_desc.TextEmbedder()

model = Model_with_desc.JobRecommendationTransformer(
    job_vocab_size=27369,
    action_vocab_size=3,  # -1: null, -2: apply, -3: view
    embed_dim=128, 
    transformer_dim=256,
    n_heads=4,
    n_layers=2,
    desc_embed_dim=description_encoder.embed_dim,  # Dimension de l'embedding de texte (384 pour all-MiniLM-L6-v2)
)

job_ids_array, actions_array = parser.load_and_pad_data("x_train_Meacfjr.csv")
job_ids_tensor, actions_tensor, action_to_idx = parser.numpy_to_torch(job_ids_array, actions_array)

job_ids = job_ids_tensor[:4]
actions = actions_tensor[:4]

fichier = "job_listings/job_listings.json"  # Remplace par le chemin de ton fichier JSON
sortie = "job_listings.txt"  # Remplace par le chemin de sortie
desc_dict = parser.parse_json(fichier, sortie)

top10_jobs, predicted_actions = model(job_ids, actions, desc_dict, description_encoder)

print(f"Top 10 job recommendations: {top10_jobs}")
print(f"Predicted actions: {predicted_actions}")

loader.save_predictions_to_csv(top10_jobs, predicted_actions, output_file="predictions.csv")