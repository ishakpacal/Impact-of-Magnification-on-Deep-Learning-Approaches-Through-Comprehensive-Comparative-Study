# Impact-of-Magnification-on-Deep-Learning-Approaches-Through-Comprehensive-Comparative-Study

In this study, we conducted a comprehensive analysis on two publicly available pathology datasets, examining different magnification levels (40x, 100x, 200x, and 400x) and their combinations. 
Using 10 state-of-the-art CNN models and 10 state-of-the-art ViT-based models, we aimed to determine the optimal magnification, providing definitive insights in this area.


The dataset, models, and codes for this article, titled "Impact of Magnification on Deep Learning Approaches Through a Comprehensive Comparative Study of Histopathological Breast Cancer Classification," will be shared.

We randomly divided the BACH and BreakHis datasets into 70% for training, 10% for validation, and 20% for testing to establish the optimal standard for deep learning models. 
Dataset details are provided below; please refer to the respective publications for dataset citations.


# Dataset 1: BreakHis: This dataset includes 40x, 100x, 200x, 400x and mixed (40x + 100x + 200x + 400x) magnification (binary and multiclass). So, this dataset includes 10 subdataset. 
For example BreakHis dataset1 is 40x binary, dataset2 is 100x binary....dataset6 is 40x multiclass....dataset10 is mixed multiclass (40x + 100+ 200x + 400x) 
# All this details can be accessed here: https://drive.google.com/file/d/1f4ep8P46AHY1ZRbnZo95Fs25BVQHWi9X/view?usp=sharing

# Dataset2: BACH dataset comprises histopathology images at a fixed magnification of 200x, offering a more contemporary and varied collection of high-resolution images compared to the Breakhis dataset 
# All this details can be accessed here: https://drive.google.com/file/d/1ZHF_b1m5EJY5fGM3T7vATsddaHH36rVq/view?usp=sharing

# Confuison matrixes can be accessed here: https://drive.google.com/file/d/1DFaOdFBrOL2oBPZw33zdbl3nf0LVDInR/view?usp=sharing
# Codes are borrowed from the huggingface repo and models and details will be shared after publication. 


Of course. Here is a professional `README.md` file written in English, incorporating the instructions for setup with the Hugging Face `datasets` library and a note about future updates post-publication.

You can copy this entire text and save it as `README.md` in your GitHub repository's root folder.

-----

# BREAKHIS Dataset Preparation Script for Machine Learning

This repository provides a Python script to organize and preprocess the [BREAKHIS (Breast Cancer Histopathological Image Classification)](https://www.google.com/search?q=https://web.inf.ufpr.br/vri/databases/breakhis/) dataset. The script is designed to prepare the data for machine learning and deep learning workflows by splitting it into standardized sets and generating corresponding CSV files, which can be easily loaded using libraries like Hugging Face `datasets`.

## 🚀 Project Overview

The BREAKHIS dataset is a valuable resource for breast cancer classification, but its directory structure, which is organized by magnification levels, can be cumbersome to use directly with standard data loaders in frameworks like PyTorch or TensorFlow. This script automates the entire preparation process, solving this challenge.

This repository is a companion to a research paper currently under review. After the paper's acceptance, this repository will be fully updated with models, training scripts, and results.

## ✨ Key Features

The `split_csv.py` script performs the following tasks automatically:

1.  **Train-Validation-Test Split (70-10-20):**

      - **Binary Classification (Benign/Malignant):**
          - Creates separate CSVs for each magnification level (`40x`, `100x`, `200x`, `400x`).
          - Creates a "combined" CSV using images from all magnification levels.
      - **Multi-Class Classification (8 Sub-types):**
          - Creates separate CSVs for each magnification level.
          - Creates a "combined" CSV using images from all magnification levels.

2.  **5-Fold Cross-Validation Split:**

      - Generates 5 distinct CSV files for 5-fold cross-validation.
      - This is performed *only* for the **40x** magnification level for both binary and multi-class tasks.

## 📁 Required Project Structure

For the script to work correctly, your project must follow the directory structure below. The script expects the `dataset_cancer_v1` folder to be in the same root directory where you run `split_csv.py`.

```
/your_project_root/
├── dataset_cancer_v1/
│   ├── classificacao_binaria/
│   │   ├── 40X/
│   │   ├── 100X/
│   │   └── ...
│   └── classificacao_multiclasse/
│       ├── 40X/
│       └── ...
├── split_csv.py             <- The script from this repository
└── README.md                <- (This file)
```

## ⚙️ Setup and Installation

1.  **Clone the Repository:**

    ```bash
    git clone <your-repository-url>
    cd <your-repository-folder>
    ```

2.  **Set up a Virtual Environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file in the project's root directory with the following content:

    ```
    pandas
    scikit-learn
    numpy
    datasets
    Pillow
    ```

    Then, install the libraries using pip:

    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♀️ How to Run

Ensure your folder structure is correct as described above. Then, simply run the script from your terminal in the project's root directory:

```bash
python split_csv.py
```

## ✅ Expected Output

The script will create a new folder named `output_csvs` in your project directory. This folder will contain all the generated CSV files, clearly named according to their task, magnification, and split type.

**Example CSV files:**

  - `classificacao_binaria_40x_split_70_10_20.csv`
  - `classificacao_multiclasse_kombine_split_70_10_20.csv`
  - `classificacao_binaria_40x_crossval_fold_1.csv`

Each CSV file contains three columns:

  - `filepath`: The relative path to the image file.
  - `label`: The class label of the image (e.g., `benign`, `malignant`, `adenosis`).
  - `split`: The dataset split the image belongs to (`train`, `validation`, or `test`).


split.py:

import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np

# --- SETTINGS ---

# 1. Name of the main directory containing your dataset
#    Updated to 'dataset_cancer_v1' based on your 'ls -R' output.
BASE_DATASET_DIR = 'dataset_cancer_v1' 

# 2. Directory where the generated CSV files will be saved
OUTPUT_CSV_DIR = 'output_csvs'

# 3. Number of folds for cross-validation
N_SPLITS_FOR_CROSS_VALIDATION = 5


# --- SCRIPT START ---

def create_split_csvs(base_dir, output_dir):
    """
    Splits the dataset into 70% train, 10% validation, and 20% test sets and creates CSV files.
    This process is done for each magnification level individually and for a "combined" set of all levels.
    """
    print("="*50)
    print("Initiating 70-10-20 Split...")
    print("="*50)

    classification_types = ['classificacao_binaria', 'classificacao_multiclasse']

    for class_type in classification_types:
        print(f"\n[Classification Type: {class_type}]")
        all_magnifications_data = []
        magnifications = ['40x', '100x', '200x', '400x']

        for mag_lower in magnifications:
            # Get the actual directory name (e.g., 40x -> 40X)
            mag_upper = mag_lower.upper()
            print(f"  -> Magnification: {mag_upper}")
            
            mag_path = os.path.join(base_dir, class_type, mag_upper)
            if not os.path.isdir(mag_path):
                print(f"    WARNING: Directory not found, skipping: {mag_path}")
                continue

            image_paths = []
            labels = []
            
            # Scan for class folders (e.g., benign/malignant)
            class_labels = [d for d in os.listdir(mag_path) if os.path.isdir(os.path.join(mag_path, d))]
            if not class_labels:
                print(f"    WARNING: No class folders found inside: {mag_path}")
                continue

            for label_folder in class_labels:
                label_path = os.path.join(mag_path, label_folder)
                for filename in os.listdir(label_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        # Saving the file path relatively is more portable
                        relative_path = os.path.join(base_dir, class_type, mag_upper, label_folder, filename)
                        image_paths.append(relative_path)
                        labels.append(label_folder)

            if not image_paths:
                print(f"    WARNING: No image files found in {mag_upper} magnification.")
                continue

            df = pd.DataFrame({'filepath': image_paths, 'label': labels})
            all_magnifications_data.append(df)  # Accumulate data for the combined set

            # Splitting the data (stratify is used to maintain class proportions)
            # First, 70% train, 30% temp
            train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
            # Then, split the 30% temp set into 10% validation and 20% test (1/3 of temp is validation, 2/3 is test)
            val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=42, stratify=temp_df['label'])

            train_df['split'] = 'train'
            val_df['split'] = 'validation'
            test_df['split'] = 'test'

            final_df = pd.concat([train_df, val_df, test_df])
            output_filename = os.path.join(output_dir, f'{class_type}_{mag_lower}_split_70_10_20.csv')
            final_df.to_csv(output_filename, index=False)
            print(f"     -> CSV created: {output_filename}")

        # Split the combined dataset from all magnifications
        if all_magnifications_data:
            print("  -> Magnification: Combined (40x+100x+200x+400x)")
            combined_df = pd.concat(all_magnifications_data, ignore_index=True)

            train_df, temp_df = train_test_split(combined_df, test_size=0.30, random_state=42, stratify=combined_df['label'])
            val_df, test_df = train_test_split(temp_df, test_size=2/3, random_state=42, stratify=temp_df['label'])

            train_df['split'] = 'train'
            val_df['split'] = 'validation'
            test_df['split'] = 'test'

            final_combined_df = pd.concat([train_df, val_df, test_df])
            output_filename_combined = os.path.join(output_dir, f'{class_type}_kombine_split_70_10_20.csv')
            final_combined_df.to_csv(output_filename_combined, index=False)
            print(f"     -> CSV created: {output_filename_combined}")

def create_cross_validation_csvs(base_dir, output_dir, n_splits):
    """
    Creates K-fold cross-validation sets for the 40x magnification level only.
    It produces a separate CSV file for each fold.
    """
    print("\n" + "="*50)
    print(f"{n_splits}-Fold Cross-Validation Process Starting (For 40x only)...")
    print("="*50)

    classification_types = ['classificacao_binaria', 'classificacao_multiclasse']
    mag_lower = '40x'
    mag_upper = '40X' # Actual directory name

    for class_type in classification_types:
        print(f"\n[Classification Type: {class_type}, Magnification: {mag_upper}]")
        mag_path = os.path.join(base_dir, class_type, mag_upper)
        if not os.path.isdir(mag_path):
            print(f"  WARNING: Directory not found, skipping: {mag_path}")
            continue

        image_paths = []
        labels = []
        class_labels = [d for d in os.listdir(mag_path) if os.path.isdir(os.path.join(mag_path, d))]
        if not class_labels:
            print(f"    WARNING: No class folders found inside: {mag_path}")
            continue

        for label_folder in class_labels:
            label_path = os.path.join(mag_path, label_folder)
            for filename in os.listdir(label_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    relative_path = os.path.join(base_dir, class_type, mag_upper, label_folder, filename)
                    image_paths.append(relative_path)
                    labels.append(label_folder)

        if not image_paths:
            print(f"  WARNING: No image files found in {mag_upper} magnification.")
            continue

        df = pd.DataFrame({'filepath': image_paths, 'label': labels})

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold_idx, (train_indices, test_indices) in enumerate(skf.split(df, df['label'])):
            fold_num = fold_idx + 1
            print(f"  -> Creating Fold {fold_num}/{n_splits}...")

            train_df = df.iloc[train_indices].copy()
            test_df = df.iloc[test_indices].copy()
            train_df['split'] = 'train'
            test_df['split'] = 'test'

            fold_df = pd.concat([train_df, test_df])
            output_filename = os.path.join(output_dir, f'{class_type}_{mag_lower}_crossval_fold_{fold_num}.csv')
            fold_df.to_csv(output_filename, index=False)
            print(f"     -> CSV created: {output_filename}")


if __name__ == '__main__':
    # Create the output directory to save CSV files
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

    # Check if the main dataset directory exists
    if not os.path.isdir(BASE_DATASET_DIR):
        print(f"ERROR: Dataset directory not found: '{BASE_DATASET_DIR}'")
        print(f"Please update the `BASE_DATASET_DIR` variable with the correct directory name.")
    else:
        # Task 1: Perform the 70-10-20 splits
        create_split_csvs(BASE_DATASET_DIR, OUTPUT_CSV_DIR)

        # Task 2: Perform the 5-fold cross-validation splits
        create_cross_validation_csvs(BASE_DATASET_DIR, OUTPUT_CSV_DIR, N_SPLITS_FOR_CROSS_VALIDATION)

        print("\n\nAll processes completed.")
        print(f"All generated CSV files have been saved to the '{OUTPUT_CSV_DIR}' directory.")


        


## 🤗 Loading with Hugging Face `datasets`

The generated CSV files are perfectly formatted for use with the Hugging Face `datasets` library, which provides a powerful and efficient way to handle data.

Here is an example of how to load the 70-10-20 split for the binary 40x task and prepare it for training:

```python
from datasets import load_dataset, DatasetDict, Image
import os

# Define the root path of your project
# The script assumes your images are in 'dataset_cancer_v1'
# and your CSVs are in 'output_csvs'
PROJECT_ROOT = "." 
CSV_PATH = os.path.join(PROJECT_ROOT, "output_csvs/classificacao_binaria_40x_split_70_10_20.csv")

# 1. Load the dataset from the CSV file
# This loads all rows into a single 'train' split by default
full_dataset = load_dataset('csv', data_files=CSV_PATH, split='train')

# 2. Filter the dataset to create train, validation, and test splits
train_ds = full_dataset.filter(lambda example: example['split'] == 'train')
val_ds = full_dataset.filter(lambda example: example['split'] == 'validation')
test_ds = full_dataset.filter(lambda example: example['split'] == 'test')

# 3. Combine them into a single DatasetDict object
breakhis_splits = DatasetDict({
    'train': train_ds,
    'validation': val_ds,
    'test': test_ds
})

# 4. (Optional but recommended) Cast the 'filepath' column to the Image feature
# This will automatically load images when you access this column.
# Note: The file paths in the CSV are relative to the project root.
def get_full_path(example):
    example['filepath'] = os.path.join(PROJECT_ROOT, example['filepath'])
    return example

breakhis_splits = breakhis_splits.map(get_full_path)
breakhis_splits = breakhis_splits.cast_column("filepath", Image())

# Rename 'filepath' to 'image' and 'label' to 'labels' for common model compatibility
breakhis_splits = breakhis_splits.rename_column("filepath", "image")
breakhis_splits = breakhis_splits.rename_column("label", "labels")

print("Dataset successfully loaded and processed:")
print(breakhis_splits)

# You can now access an example
# The image will be loaded automatically as a PIL object
first_example = breakhis_splits['train'][0]
image = first_example['image']
label = first_example['labels']
print(f"\nFirst training image size: {image.size}")
print(f"First training image label: {label}")
```

## 📝 Note on Publication & Future Work

This repository is supplementary material for a research paper currently undergoing peer review. Upon the official acceptance and publication of the paper, this repository will be significantly updated to include:

  - The complete source code used for model training and evaluation.
  - Pre-trained model weights for all experiments.
  - Detailed results, performance metrics, and analysis scripts.
  - A link to the published paper.

Thank you for your interest in our work.
