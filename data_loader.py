import os
import shutil
import kagglehub

def load_and_prepare_data():
    print("Downloading dataset from Kaggle...")
    # Download latest version of the nutsedge dataset
    download_path = kagglehub.dataset_download("amarachari0003/nutsedge-weed-images-and-annotations")
    print(f"Dataset downloaded to temporary path: {download_path}")
    
    # Extract directly into the root directory of the project
    root_dir = os.getcwd()
    
    # Check if a marker file like 'data.yaml' already exists to avoid redundant copying
    if not os.path.exists(os.path.join(root_dir, "data.yaml")):
        print("Extracting files to the root directory...")
        for item in os.listdir(download_path):
            s = os.path.join(download_path, item)
            d = os.path.join(root_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        print("Dataset successfully extracted to root folder.")
    else:
        print("Dataset files (data.yaml) already exist in the root folder.")
        
    return root_dir

if __name__ == "__main__":
    load_and_prepare_data()