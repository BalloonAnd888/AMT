import os
from preprocessing.constants import *
from preprocessing.dataset import MAESTRO

def loadDataset(dataset_root):
    print(f"Initializing MAESTRO dataset from {dataset_root}...")
    dataset = MAESTRO(path=dataset_root, groups=['train'], sequence_length=SEQUENCE_LENGTH)

    if len(dataset) == 0:
        print("Error: Dataset is empty. Check your path and metadata")
        return 
    return dataset 

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    if os.path.exists(DATA_PATH):
        loadDataset(DATA_PATH)
    else:
        print(f"Path not found: {DATA_PATH}")
        print("Please edit DATA_PATH in loadDataset.py")
