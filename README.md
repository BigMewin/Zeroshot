# Zeroshot Gene-compound target gene
# Step 1. run Feature Transformation.ipynb to generate transformed features
# Step 2. run Classification Evaluation.ipynb to evaluate the classification accuracy of MLP and SLPP
# Step 3. run the ipynb files in "mAP Umap Analysis folder" for mAP evaluation
# Step 4. run "Run this to set up datasplit and evaluation result.ipynb" first in gzsda.main to generate required .mat files
# Step 5. In your terminal, run "bash run_xray" to procees CCVAE and MLP/1-NN evaluation (note: you need to mannually change the MLP/1-NN evaluation strategy in train_vae2_xray.py)
# Step 6. The mAP evaluation process of CCVAE is still in progress, you can mannually extract transformed features from CCVAE and do mAP evaluation in "mAP Umap Analysis folder"
# Note: You need to change the paths of files to fit your coding environment. Contact me to fetch the "total_new.csv" file
