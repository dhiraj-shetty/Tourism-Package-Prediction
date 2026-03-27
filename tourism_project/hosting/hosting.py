from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Push all deployment files to the Hugging Face Space
api.upload_folder(
    folder_path="tourism_project/deployment",     # Local folder containing deployment files
    repo_id="dhirajshetty/Tourism-Package-Prediction",  # Target HF Space
    repo_type="space",
    path_in_repo="",
)
print("Deployment files pushed to Hugging Face Space successfully.")
