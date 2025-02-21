import json
import re

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

# Exemple d'utilisation
if __name__ == "__main__":
    fichier = "job_listings/job_listings.json"  # Remplace par le chemin de ton fichier JSON
    sortie = "job_listings.txt"  # Remplace par le chemin de sortie
    data = parse_json(fichier, sortie)
    print(data["27365"])