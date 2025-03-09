import os
from huggingface_hub import HfApi

def modelupload(repo_id: str = 'NaTaB/VAE-AV', paths: list = []):
    api = HfApi()

    for path in paths:
        # Extract the part of the path after "autoencoder/model/"
        sub_path = path.split("autoencoder/model/")[-1]  # Extract vae32/best/decoder_model.pth
        path_in_repo = sub_path.replace("\\", "/")  # Ensure it uses forward slashes

        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path_in_repo,  # Keep the original structure dynamically
            repo_id=repo_id,
        )
        print(f"Uploaded: {path} -> {path_in_repo}")

    print("All models uploaded successfully!")


if '__name__' == "__main__":
    modelupload(repo_id="NaTaB/VAE-AV", paths=["../model/vae32/best/decoder_model.pth", "../model/vae32/best/encoder_model.pth"])


