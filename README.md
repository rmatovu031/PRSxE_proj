# PRS - Environment interaction analysis for Type 2 Diabetes in the UK Biobank
This is the readme file for the entire project outline done in steps A to D
## Step A : Data Extraction 
- Data extraction from the UK Biobank using UKB-RAP using the instructions outlined in https://github.com/rmatovu031/PRSxE_proj/blob/main/A_data_prep/gxe_data_updated.ipynb
- Ancestry inference for the African participants done using instructions outlined in https://github.com/rmatovu031/PRSxE_proj/blob/main/A_data_prep/gxe_data_updated.ipynb
- Baseline Characteristics between cases and controls for African and European participants performed using https://github.com/rmatovu031/PRSxE_proj/blob/main/A_data_prep/Baseline_xtics.ipynb 

## Step B : PRS
A PRS for T2D for the African participants was developed and using the GenoPred pipeline. The pipeline was run on the UKB-RAP using an RStudio server session with 
https://github.com/rmatovu031/PRSxE_proj/blob/main/B_PRS/1_setup%20the%20Rstudio_server
The GenoPred configuration file - 
From the GenoPred output, we can get comparison of the computational requirements for all the steps in running the PRS is outlined as 
https://github.com/rmatovu031/PRSxE_proj/blob/main/B_PRS/R_computational_time.ipynb and a comparison of the comparison for the PRS methods used
https://github.com/rmatovu031/PRSxE_proj/blob/main/B_PRS/R_pgs_compare.ipynb

## Step C : Statistical Analysis for Association testing and PRSxEnvironment interaction
Used logistic regression to run association testing to get the predictors for T2D in Africans and Europeans
AFR - https://github.com/rmatovu031/PRSxE_proj/blob/main/C_Stat_approach/R_gxe_AFR_finale.ipynb
EUR - https://github.com/rmatovu031/PRSxE_proj/blob/main/C_Stat_approach/R_gxe_EUR_finale.ipynb
The analysis was run on ilifu

## Step D : Machine Learning based approach
Used XGBoost model to run a classifer model and used SHAP for interpretability. 
This analysis was run on ilifu using the environment https://github.com/rmatovu031/PRSxE_proj/blob/main/D_ML_approach/gxe_evn.yml
AFR - The shell script https://github.com/rmatovu031/PRSxE_proj/blob/main/D_ML_approach/run_afr_full.sh was used to run the python script
https://github.com/rmatovu031/PRSxE_proj/blob/main/D_ML_approach/afr_full_script.py
EUR - The shell script https://github.com/rmatovu031/PRSxE_proj/blob/main/D_ML_approach/run_eur_full.sh was used to run the python script
https://github.com/rmatovu031/PRSxE_proj/blob/main/D_ML_approach/eur_full_script.py
