# Zeroshot Gene-compound target gene
* This is for gene classification of compound perturbations. This experiment aims to:
* 1. Apply supervised learning methods for feature transformation to filter out undesired features.
  2. Reduce the domain gap of domain adaptation problem
  3. Deal with unlabeled data in target domain in zero-shot domain adaptation problem
* Please follow the steps below:
* Step 1. run Feature Transformation.ipynb to generate transformed features
* Step 2. run Classification Evaluation.ipynb to evaluate the classification accuracy of MLP and SLPP (note: you need to change the parameter 'mode' as 'top_1' or 'top_10' to get the desired result)
* Step 3. run the FIRST part of cells in "Run this to set up datasplit and evaluation result.ipynb" first in gzsda.main to generate required .mat files
* Step 4. In your terminal, run "bash run_xray" to procees CCVAE and MLP/Nearest-Class-Mean evaluation (note1: you need to mannually change the MLP/Nearest-Class-Mean evaluation strategy in train_vae2_xray.py) (note2: you need to change the parameter 'mode' as 'top_1' or 'top_10' to get the desired results)
* Step 5. run the SECOND and THIRD part of cells in "Run this to set up datasplit and evaluation result.ipynb" after running bash run_xray to get the analyzed accuracy and the transformed the features for mAP evaluation.
* Step 6. run the ipynb files in "mAP Umap Analysis folder" for mAP evaluation
* Note:
  1.You need to change the paths of files to fit your coding environment. Contact me to fetch the "total_new.csv" file
  2.Classification Evaluation.ipynb also aims to generate the data for CCVAE training and mAP evaluation
  3.Feature Transformation.ipynb aims to generate the data for MLP and SLPP mAP evaluation. DO NOT MIX IT UP WITH Classification Evaluation.ipynb!
  4.For cross validation, you need to change the random.seed() in Classification Evaluation.ipynb and Feature Transformation.ipynb. There are three in Classification Evaluation.ipynb and two in Feature Transformation.ipynb. Do not omit any random.seed()! After you changed the random.seed(), redo step 3-6.
* Supplementary: You can also try to split the data by cell line or by time point. Refer to step 1, 2, and 6 and open the corresponding notebooks 
