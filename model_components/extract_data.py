# Data Preperation
import shutil
import os
import zipfile

file_name = "../rpc_image_dataset.zip"
target_dir = "../rpc_image_dataset"

# Extract the zip file
def data_set(file_name, target_dir):
    print("Extracting RPC image dataset from *.zip \n")
    
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        # Verify the extraction
        print("Extraction complete. Contents of the target directory:")
        print(os.listdir(target_dir))
        remove_ipynb(target_dir)
    except zipfile.BadZipFile:
        print("Error: The file is not a zip file or it is corrupted.")
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# remove .ipynb_checkpoints directory if it exists
def remove_ipynb(target_dir):
    print("removing any ipynb from target directory \n")
    ipynb_checkpoints_dir = os.path.join(target_dir, '.ipynb_checkpoints')
    if os.path.exists(ipynb_checkpoints_dir):
        shutil.rmtree(ipynb_checkpoints_dir)
    