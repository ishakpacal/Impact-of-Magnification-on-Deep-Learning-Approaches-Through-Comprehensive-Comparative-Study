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
