# Zeroshot Gene-compound target gene

This repository contains code for gene classification of compound perturbations. The experiment aims to achieve the following objectives:

## Steps to Follow:

### Step 1: Feature Transformation
- Run the notebook `Feature Transformation.ipynb` to generate transformed features.

### Step 2: Classification Evaluation
- Run the notebook `Classification Evaluation.ipynb` to evaluate the classification accuracy of MLP and SLPP.
- Note: In the notebook, make sure to change the parameter `mode` to either `top_1` or `top_10` to obtain the desired result.

### Step 3: Data Split and Evaluation Setup
- Run the first part of the cells in the notebook `Run this to set up datasplit and evaluation result.ipynb` located in the `gzsda.main` directory.
- This step will generate the required `.mat` files for further processing.

### Step 4: CCVAE and MLP/1-Nearest-Neighbor Evaluation
- In your terminal, run the command `bash run_xray` to execute CCVAE and MLP/1-Nearest-Neighbor evaluation.
- Note 1: You may need to manually change the MLP/1-Nearest-Neighbor evaluation strategy in the file `train_vae2_xray.py`.
- Note 2: Similar to Step 2, adjust the parameter `mode` to `top_1` or `top_10` to obtain the desired results.

### Step 5: Analyzing Accuracy and Transformed Features
- Run the second and third parts of the cells in the notebook `Run this to set up datasplit and evaluation result.ipynb` after completing Step 4.
- This will provide the analyzed accuracy and the transformed features required for mAP evaluation.

### Step 6: mAP Evaluation
- Run the notebooks in the `mAP Umap Analysis folder` to perform mAP evaluation.

Notes:
1. Make sure to adjust the file paths to match your coding environment.
2. The notebook `Classification Evaluation.ipynb` also generates data for CCVAE training and mAP evaluation.
3. The notebook `Feature Transformation.ipynb` generates data for MLP and SLPP mAP evaluation. Do not confuse it with `Classification Evaluation.ipynb`!
4. If you want to perform cross-validation, modify the `random.seed()` values in both `Classification Evaluation.ipynb` and `Feature Transformation.ipynb`. There are three instances in `Classification Evaluation.ipynb` and two in `Feature Transformation.ipynb`. Ensure that you don't skip any `random.seed()` calls. After modifying the `random.seed()` values, repeat steps 3-6.

Supplementary: You can also explore splitting the data by cell line or time point. Refer to steps 1, 2, and 6, and open the corresponding notebooks.

## Datasets
- Cellprofiler, Dinov2, and Effnetb0 datasets can be accessed from this link: https://www.terabox.com/sharing/link?surl=gwHoxwJsKQ3WBOXU8jYnJw&path=%2FCPJUMP1

If you have any questions or need further assistance, feel free to contact me.
