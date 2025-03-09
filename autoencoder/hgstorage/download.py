from huggingface_hub import hf_hub_download, HfApi
import os

def get_model(repo_id: str = 'NaTaB/VAE-AV', modelname: str = "vae32", save_dir: str = "autoencoder/model"):
    api = HfApi()
    
    # List all files in the repository
    files = api.list_repo_files(repo_id)

    # Filter files that are inside the modelname folder
    model_files = [f for f in files if f.startswith(f"{modelname}/")]

    if not model_files:
        print(f"No models found in {repo_id} under {modelname}/")
        return

    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    downloaded_paths = []
    for filename in model_files:
        local_path = os.path.join(save_dir, os.path.basename(filename))
        
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=save_dir)
        downloaded_paths.append(downloaded_path)
        
        print(f"Downloaded: {filename} -> {downloaded_path}")

    print("All models downloaded successfully!")
    return downloaded_paths  # Return the list of downloaded files

# Example usage
if '__name___' == "__main__":
    get_model(repo_id="NaTaB/VAE-AV", modelname="vae32", save_dir="../model")

