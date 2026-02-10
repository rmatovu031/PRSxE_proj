import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
import shap
from sklearn.model_selection import KFold
import logging

def main():
    log_file = "/users/rmatovu/proj_GxE/results_afr/suzuki/run_log.txt"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info("=== SCRIPT STARTED ===")
           
    # STEP1: Loading the dataset
    input_path = os.getenv("INPUT_FILE", "/users/rmatovu/proj_GxE/new_dataset.csv")
    output_dir = os.getenv("OUTPUT_DIR", "/users/rmatovu/proj_GxE/results_afr/suzuki/")
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Reading input file: {input_path}")
    
    # Log dataset shape and columns

    df = pd.read_csv(input_path, sep=",")
    
    # Log dataset shape and columns
    logging.info(f"Dataset loaded. Shape: {df.shape}")
    logging.info(f"Columns found: {df.columns.tolist()}")

    ## African dataset
    df_AFR = df[df["pop"] == "AFR"]

    # STEP2: case-control balance
    case_count = df_AFR[df_AFR['t2d_cc'] == 1].shape[0]
    control_count = df_AFR[df_AFR['t2d_cc'] == 0].shape[0]

    if control_count > case_count:
        cases = df_AFR[df_AFR['t2d_cc'] == 1]
        controls = df_AFR[df_AFR['t2d_cc'] == 0].sample(n=case_count, random_state=42)
        df_afr = pd.concat([cases, controls], axis=0)
        df_afr = df_afr.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"New balanced counts - Cases: {df_afr[df_afr['t2d_cc'] == 1].shape[0]}, "
            f"Controls: {df_afr[df_afr['t2d_cc'] == 0].shape[0]}")
    else:
        df_afr = df_AFR.copy()
        print("Controls are already equal or fewer than cases - no downsampling needed")

    # STEP3: Feature distribution
    numeric_cols = df_afr.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten()

    for i, column in enumerate(numeric_cols):
        sns.histplot(df_afr[column], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {column}')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.title("Feature Distribution for sampled African participants in the UK Biobank", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"afr_features_distributions.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close() 

    # STEP4: Training the classifier model
    X_afr = df_afr.drop(columns=["IID", "t2d_cc","pop", "weight", "height", "prs_t2d"])
    y_afr = df_afr["t2d_cc"]

    X_afr_A, X_afr_B, y_afr_A, y_afr_B = train_test_split(X_afr, y_afr, test_size=0.2, stratify=y_afr, random_state=42)
    print("Using best parameters and proceeding with analysis")
    
    # Using best parameters already generated from bayesian optimisation
    best_params = {
        'n_estimators': 568,
        'max_depth': 3,
        'learning_rate': 0.04964046381061715,
        'subsample': 0.798635413797731,
        'colsample_bytree': 0.839284202267856,
        'gamma': 3.217167494115418,
        'min_child_weight': 4,
        'reg_alpha': 2.410119897494607,
        'reg_lambda': 5.229623057980013,
        'eval_metric': 'auc',
        'enable_categorical': False,
        'tree_method': 'hist',
        'random_state': 42,
        'verbosity': 0,
        'device': 'cuda'
    }

    print(f"Best params: {best_params}")

    best_clf = xgb.XGBClassifier(**best_params)
    best_clf.fit(X_afr_A, y_afr_A)

    # STEP5: Using SHAP to rank the features
    explainer_A = shap.Explainer(best_clf)
    shap_values_A = explainer_A(X_afr_A)

    shap.plots.bar(shap_values_A, max_display=None, show=False)
    plt.title("Top Predictors of Type 2 Diabetes in Africans in the UK Biobank", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_features_afr.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # STEP6: Ranking feature interactions
    explainer = shap.TreeExplainer(best_clf)
    shap_interactions = explainer.shap_interaction_values(X_afr_A)
    mean_interactions = np.abs(shap_interactions).mean(axis=0)

    prs_col = "prs_T2D"
    prs_idx = list(X_afr_A.columns).index(prs_col)

    prs_interactions = mean_interactions[prs_idx, :]
    prs_interactions[prs_idx] = 0  # remove self-interaction

    indices = np.argsort(prs_interactions)[::-1][:11]
    features = X_afr_A.columns[indices]
    strengths = prs_interactions[indices]

    top_df = pd.DataFrame({
        "Interaction": [f"{prs_col} × {feat}" for feat in features],
        "Strength": strengths
    })

    plt.figure(figsize=(8,6))
    sns.barplot(y="Interaction", x="Strength", data=top_df, hue="Strength", palette="viridis_r", legend=False)
    plt.title("Top SHAP Interactions with PRS", fontsize=12, pad=20)
    plt.savefig(os.path.join(output_dir, "top_interactions_afr.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("onto Permutation-based significance testing")

    ## STEP7. Permutation-based significance testing
    def permutation_test_cv(X, y, prs_col, best_params, n_splits=5, n_perm=100, random_state=42):
        """
        Cross-validated permutation test for PRS × feature SHAP interactions.
        Stores full permutation distributions for plotting.
        """
        # Define allowed parameters
        allowed_params = [
            'n_estimators', 'max_depth', 'learning_rate', 'subsample',
            'colsample_bytree', 'gamma', 'min_child_weight', 'reg_alpha',
            'reg_lambda', 'tree_method', 'device', 'objective', 'random_state', 'eval_metric'
        ]
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        prs_idx = list(X.columns).index(prs_col)

        clf_params = {k: best_params[k] for k in allowed_params if k in best_params}
        clf_params["objective"] = "binary:logistic"

        # ---- Observed interaction strengths (CV averaged) ----
        fold_interactions = []
        for train_idx, _ in kf.split(X):
            model = xgb.XGBClassifier(**clf_params).fit(X.iloc[train_idx], y.iloc[train_idx])
            explainer = shap.TreeExplainer(model)
            shap_interactions = explainer.shap_interaction_values(X.iloc[train_idx])
            mean_interactions = np.abs(shap_interactions).mean(axis=0)
            fold_interactions.append(mean_interactions[prs_idx, :])

        obs_strengths = np.mean(fold_interactions, axis=0)
        obs_strengths[prs_idx] = 0  # remove self interaction

        # ---- Permutation test (keep full distributions) ----
        null_distributions = {i: [] for i in range(len(X.columns))}

        for _ in range(n_perm):
            y_perm = np.random.permutation(y)
            model = xgb.XGBClassifier(**clf_params).fit(X, y_perm)
            explainer = shap.TreeExplainer(model)
            shap_interactions_perm = explainer.shap_interaction_values(X)
            mean_interactions_perm = np.abs(shap_interactions_perm).mean(axis=0)
            for j in range(len(X.columns)):
                null_distributions[j].append(mean_interactions_perm[prs_idx, j])

        # ---- Compute p-values and permutation means ----
        perm_means = np.zeros(len(X.columns))
        pvals = np.ones(len(X.columns))
        for j in range(len(X.columns)):
            if j == prs_idx:
                continue
            null_vals = np.array(null_distributions[j])
            perm_means[j] = null_vals.mean()
            pvals[j] = (np.sum(null_vals >= obs_strengths[j]) + 1) / (len(null_vals) + 1)

        # ---- Build results table ----
        results = pd.DataFrame({
            "Feature": X.columns,
            "ObsStrength": obs_strengths,
            "PermMean": perm_means,
            "pval": pvals,
            "NullDist": [np.array(null_distributions[j]) for j in range(len(X.columns))]
        })

        results = results[results["Feature"] != prs_col].sort_values("ObsStrength", ascending=False)
        return results.reset_index(drop=True)

    # Call the permutation test function
    results = permutation_test_cv(X_afr_A, y_afr_A, prs_col, best_params)
    ####
    # Add significance stars
    def pval_to_star(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        else: return ""

    results["Significance"] = results["pval"].apply(pval_to_star)

    # Select columns to export
    export_cols = ["Feature", "ObsStrength", "PermMean", "pval", "Significance"]

    # Save ranked results as TSV
    tsv_path = os.path.join(output_dir, "ranked_interactions.tsv")
    results[export_cols].to_csv(tsv_path, sep="\t", index=False)

    print(f"Ranked interaction table saved to:\n  {tsv_path}")



    # ---- Plot Top N with boxplots ----
    top_n = 13
    top_df = results.head(top_n).copy()

    def pval_to_star(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        else: return ""

    top_df["Significance"] = top_df["pval"].apply(pval_to_star)

    plt.figure(figsize=(10,7))

    # --- Plot permutation distributions as boxplots ---
    for i, null_vals in enumerate(top_df["NullDist"]):
        plt.boxplot(
            null_vals, positions=[i], widths=0.5,
            patch_artist=True, boxprops=dict(facecolor="lightgray", alpha=0.6)
        )

    # --- Overlay observed strengths as bars ---
    plt.bar(
        range(len(top_df)), top_df["ObsStrength"],
        color=sns.color_palette("viridis_r", len(top_df)), alpha=0.8
    )

    # --- Add significance stars ---
    for i, (val, star) in enumerate(zip(top_df["ObsStrength"], top_df["Significance"])):
        plt.text(i, val + 0.01*max(top_df["ObsStrength"]), star,
                ha="center", va="bottom", fontsize=12, weight="bold")

    plt.xticks(range(len(top_df)), top_df["Feature"], rotation=45, ha="right")
    plt.ylabel("SHAP Interaction Strength")
    plt.title("Top PRS × Feature Interactions with Permutation Null Distributions", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "permutation_test_results_afr.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close() 

    ## STEP8. Model Validation

    # Get predicted class and probabilities
    y_pred = best_clf.predict(X_afr_B)
    y_proba = best_clf.predict_proba(X_afr_B)[:, 1]  # Probabilities for class 1

    # Accuracy
    accuracy = accuracy_score(y_afr_B, y_pred)
    print(f"Optimized Model Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:\n", classification_report(y_afr_B, y_pred))

    # Confusion matrix
    print("\nConfusion Matrix:\n", confusion_matrix(y_afr_B, y_pred))

    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_afr_B, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", lw=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ROC_curve_afr.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close() 

    # Precision-Recall Curve and AUPR
    precision, recall, _ = precision_recall_curve(y_afr_B, y_proba)
    aupr = average_precision_score(y_afr_B, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AUPR = {aupr:.2f}", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", fontsize=12, pad=20)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "PR_AUPR_curve_afr.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close() 

    ## STEP9: Interpreting SHAP interaction values
    # Dictionary of environmental features (key: column name, value: display name)
    env_features = {
        "bmi": "BMI",
        "whr": "WHR",
        "age": "Age",
        "alcohol_intake": "Alcohol Intake",
        "total_met": "Physical Activity",
        "townsend": "Social Deprivation",
        "ever_smoked": "Smoking",
        "sleep_dur": "Sleep Duration",
        "cmcs": "Childhood Trauma",
        "ipvs": "Intimate Partner Violence",
        "diet_score": "Diet",
        "sex": "Sex",
        "bmr": "Resting Metabolism"
    }
    # Loop through each environmental feature
    for env_key, env_name in env_features.items():
        # Create the dependence plot
        shap.dependence_plot(
            ("prs_T2D", env_key),  # PRS-environment interaction
            shap_interactions, 
            X_afr_A,
            show=False  # Disable auto-display
        )
    
        # Set a dynamic title
        plt.title(f"PRS - {env_name} Interaction for Type 2 Diabetes in Africans", fontsize=12, pad=20)
    
        # Adjust layout & save
        plt.tight_layout()
        #plt.savefig(f"prs_{env_key}_afr.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(output_dir, f"prs_{env_key}_afr.png"), dpi=300, bbox_inches='tight', facecolor='white')

        plt.close()  # Free memory
    
        print(f" :) Saved: prs_{env_key}_afr.png")

    print("All PRS-environment plots generated successfully!")
if __name__ == "__main__":
    main()

