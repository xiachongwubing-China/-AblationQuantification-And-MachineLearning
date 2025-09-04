import os
import glob
import pandas as pd
import numpy as np
from radiomics import featureextractor

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.metrics import roc_curve, roc_auc_score
from lifelines import CoxPHFitter, KaplanMeierFitter

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ================= 配置 =================
img_dir = r"D:\AIEnv\TotalSegmentator\resample2"
mask_dir = r"D:\AIEnv\TotalSegmentator\label2"
excel_file = r"zzbanv4.xlsx"
feature_file = r"radiomics_features.csv"
output_dir = r"Plots"
os.makedirs(output_dir, exist_ok=True)

N_BOOTSTRAPS = 1000
RANDOM_SEED = 42
ICC_THRESHOLD = 0.75  

def extract_features():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    extractor.enableImageTypes(Original={}, Wavelet={})
    records = []
    for img_path in glob.glob(os.path.join(img_dir, "*.nii.gz")):
        fname = os.path.basename(img_path)
        patient_code = fname[:4]
        mask_path = os.path.join(mask_dir, f"{patient_code}.nii.gz")
        if not os.path.exists(mask_path):
            continue
        result = extractor.execute(img_path, mask_path)
        row = {k: v for k, v in result.items() if k.startswith("original") or k.startswith("wavelet")}
        row["PatientCode"] = patient_code
        records.append(row)
    df = pd.DataFrame(records)
    df.to_csv(feature_file, index=False)
    return df

if os.path.exists(feature_file):
    ans = input("An existing feature file is detected. Do you want to skip feature extraction?(Y/N): ")
    if ans.strip().upper() == 'Y':
        features = pd.read_csv(feature_file)
    else:
        features = extract_features()
else:
    features = extract_features()

# ================= ICC  =================
def compute_icc(data):
    from pingouin import intraclass_corr
    icc_results = {}
    features_only = [c for c in data.columns if c not in ["PatientCode", "Rater"]]
    for feat in features_only:
        tmp = data[["PatientCode", "Rater", feat]].dropna()
        if tmp[feat].nunique() <= 1:
            icc_results[feat] = np.nan
            continue
        try:
            icc_val = intraclass_corr(data=tmp, targets="PatientCode", raters="Rater", ratings=feat)
            icc_val = icc_val.loc[icc_val["Type"] == "ICC2", "ICC"].values[0]
            icc_results[feat] = icc_val
        except:
            icc_results[feat] = np.nan
    return pd.Series(icc_results, name="ICC")

if "Rater" not in features.columns:
    icc_filtered_features = features.copy()
else:
    icc_values = compute_icc(features)
    good_feats = icc_values[icc_values >= ICC_THRESHOLD].index.tolist()
    icc_filtered_features = features[["PatientCode"] + good_feats]
    icc_values.to_csv("icc_values.csv", index=True, encoding="utf-8-sig")

# ================= Read Excel =================
train_df = pd.read_excel(excel_file, sheet_name="sheet1")
valid_df = pd.read_excel(excel_file, sheet_name="sheet2")
test_df  = pd.read_excel(excel_file, sheet_name="sheet3")
for df in [train_df, valid_df, test_df]:
    df["PatientCode"] = df["PatientID"].apply(lambda x: f"W{int(x):03d}")

def merge_data(clinical_df, features):
    return clinical_df.merge(features, on="PatientCode", how="inner")

train = merge_data(train_df, icc_filtered_features)
valid = merge_data(valid_df, icc_filtered_features)
test  = merge_data(test_df, icc_filtered_features)

img_features = [c for c in icc_filtered_features.columns if c not in ['PatientCode']]
for df in [train, valid, test]:
    df[img_features] = df[img_features].replace([np.inf, -np.inf], np.nan).fillna(0)

min_train_val = train[img_features].min().min()
for df in [train, valid, test]:
    df[img_features] = np.log1p(df[img_features] - min_train_val + 1e-3)

scaler = StandardScaler()
X_train_img_scaled = scaler.fit_transform(train[img_features])
X_valid_img_scaled = scaler.transform(valid[img_features])
X_test_img_scaled  = scaler.transform(test[img_features])

y_train = train["Relapse"].values
y_valid = valid["Relapse"].values
y_test  = test["Relapse"].values

# Univariate + LASSO
F, pvals = f_classif(X_train_img_scaled, y_train)
mask = pvals < 0.05
X_train_img_filtered = X_train_img_scaled[:, mask]
selected_img_features = np.array(img_features)[mask]

lasso = LassoCV(cv=5, alphas=np.logspace(-4,0,50)).fit(X_train_img_filtered, y_train)
coef_mask = lasso.coef_ != 0
final_img_features = selected_img_features[coef_mask]

# Save LASSO result
pd.DataFrame(final_img_features, columns=["SelectedFeature"]).to_csv("lasso_selected_features.csv", index=False, encoding="utf-8-sig")
lasso_coef_df = pd.DataFrame({"Feature": selected_img_features, "Coefficient": lasso.coef_})
lasso_coef_df = lasso_coef_df[lasso_coef_df["Coefficient"] != 0].sort_values("Coefficient", ascending=False)
lasso_coef_df.to_csv("lasso_features_with_coef.csv", index=False, encoding="utf-8-sig")
print(f"[INFO] LASSO: {len(final_img_features)} features saved -> lasso_selected_features.csv")

clinical_features = ['GGOMAM','FIB']
for df in [train, valid, test]:
    df[clinical_features] = df[clinical_features].apply(pd.to_numeric, errors='coerce').fillna(0)

X_train_final = pd.concat([train[final_img_features], train[clinical_features]], axis=1)
X_valid_final = pd.concat([valid[final_img_features], valid[clinical_features]], axis=1)
X_test_final  = pd.concat([test[final_img_features],  test[clinical_features]],  axis=1)


base_models = {
    "XGBoost": xgb.XGBClassifier(n_estimators=80,
                                    max_depth=2,
                                    learning_rate=0.03,
                                    subsample=0.6,
                                    colsample_bytree=0.7,
                                    reg_lambda=1.5,
                                    reg_alpha=0.5,
                                    use_label_encoder=False,
                                    eval_metric='logloss',
                                    random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=RANDOM_SEED),
    "LightGBM": lgb.LGBMClassifier(   n_estimators=80,         
                                        learning_rate=0.05,     
                                        max_depth=2,             
                                        num_leaves=3,           
                                        min_child_samples=50,   
                                        subsample=0.6,           
                                        colsample_bytree=0.6,    
                                        reg_alpha=0.01,         
                                        reg_lambda=0.01,        
                                        random_state=RANDOM_SEED),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_split=5,
                                           min_samples_leaf=3, max_features='sqrt', random_state=RANDOM_SEED),
    "AdaBoost": AdaBoostClassifier( estimator=DecisionTreeClassifier(max_depth=1),
                                        n_estimators=60,
                                        learning_rate=0.03,
                                        random_state=42),
    "GNB": GaussianNB(),
    "CNB": ComplementNB(),
    "SVM": SVC(probability=True, random_state=RANDOM_SEED),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

fitted_models = {}
for name, model in base_models.items():
    try:
        model.fit(X_train_final, y_train)
        
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
        calibrated_model.fit(X_valid_final, y_valid)
        
        fitted_models[name] = calibrated_model
        print(f"[INFO] Trained and calibrated: {name}")
    except Exception as e:
        print(f"[WARN] Model {name} failed to train or calibrate: {e}")

def bootstrap_ci(metric_func, y_true, y_score, n_bootstraps=N_BOOTSTRAPS, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boots = []
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            val = metric_func(y_true[idx], y_score[idx])
            boots.append(val)
        except:
            continue
    if len(boots) == 0:
        return np.nan, (np.nan, np.nan)
    boots = np.array(boots)
    lower = np.percentile(boots, 2.5)
    upper = np.percentile(boots, 97.5)
    return metric_func(y_true, y_score), (lower, upper)

def sens_spec_at_threshold(y_true, y_score, threshold):
   # y_pred = (y_score >= threshold).astype(int)
    y_pred = (y_score >= 0.5).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sens, spec

def bootstrap_ci_sens_spec(y_true, y_score, threshold, n_bootstraps=N_BOOTSTRAPS, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    sens_list = []
    spec_list = []
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        s, sp = sens_spec_at_threshold(y_true[idx], y_score[idx], threshold)
        if np.isnan(s) or np.isnan(sp):
            continue
        sens_list.append(s)
        spec_list.append(sp)
    if len(sens_list) == 0:
        return (np.nan, (np.nan, np.nan)), (np.nan, (np.nan, np.nan))
    sens_arr = np.array(sens_list)
    spec_arr = np.array(spec_list)
    sens_ci = (np.percentile(sens_arr, 2.5), np.percentile(sens_arr, 97.5))
    spec_ci = (np.percentile(spec_arr, 2.5), np.percentile(spec_arr, 97.5))
    sens0, spec0 = sens_spec_at_threshold(y_true, y_score, threshold)
    return (sens0, sens_ci), (spec0, spec_ci)
def evaluate_models_on_dataset(X, y_true, dataset_name, plot_auc_only=True):
    results = []
    plt.figure(figsize=(7,6))
    plt.title(f"{dataset_name} ROC Curve")
    for name, model in fitted_models.items():
        try:
            y_prob = model.predict_proba(X)[:,1]
        except Exception as e:
            print(f"[WARN] {name} failed predict_proba on {dataset_name}: {e}")
            continue

        # AUC and CI (bootstrap)
        auc, auc_ci = bootstrap_ci(roc_auc_score, y_true, y_prob)
        # Youden threshold
        fpr, tpr, th = roc_curve(y_true, y_prob)
        youden_idx = np.argmax(tpr - fpr)
        best_thr = th[youden_idx]

        # sensitivity & specificity at threshold + bootstrap CI
        (sens, sens_ci), (spec, spec_ci) = bootstrap_ci_sens_spec(y_true, y_prob, best_thr)

        if plot_auc_only:
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
        else:
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}])")

        results.append({
            "Model": name,
            "AUC": auc,
            "AUC_CI_low": auc_ci[0],
            "AUC_CI_high": auc_ci[1],
            "Threshold_Youden": best_thr,
            "Sensitivity": sens,
            "Sensitivity_CI_low": sens_ci[0],
            "Sensitivity_CI_high": sens_ci[1],
            "Specificity": spec,
            "Specificity_CI_low": spec_ci[0],
            "Specificity_CI_high": spec_ci[1]
        })

    plt.plot([0,1],[0,1],'--', color='grey')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_ROC.png"), dpi=300)
    plt.show()

    df_res = pd.DataFrame(results).sort_values("AUC", ascending=False)
    df_res.to_csv(os.path.join(output_dir, f"metrics_{dataset_name}.csv"), index=False, encoding='utf-8-sig')
    print(f"[INFO] Saved metrics table -> {os.path.join(output_dir, f'metrics_{dataset_name}.csv')}")
    return df_res


def plot_calibration_curve(X, y_true, dataset_name, window=7):

    plt.figure(figsize=(7,6))
    plt.title(f"{dataset_name} Calibration Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for name, model in fitted_models.items():
        try:
            y_prob = model.predict_proba(X)[:, 1]
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")

            prob_true_smooth = np.convolve(prob_true, np.ones(window)/window, mode='same')
            prob_pred_smooth = np.convolve(prob_pred, np.ones(window)/window, mode='same')

            plt.plot(prob_pred_smooth, prob_true_smooth, lw=2, label=name)  
        except Exception as e:
            print(f"[WARN] Calibration plot failed for {name}: {e}")
            continue

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed probability")
    plt.legend(loc="best", fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_Calibration.png"), dpi=300)
    plt.show()


def plot_standardized_net_benefit(X, y_true, dataset_name):
    thresholds = np.linspace(0.01, 0.99, 100)
    prevalence = (y_true == 1).mean()
    n = len(y_true)
    plt.figure(figsize=(7,6))
    plt.title(f"{dataset_name} Standardized Net Benefit")
    
    for name, model in fitted_models.items():
        try:
            y_prob = model.predict_proba(X)[:,1]
        except:
            continue
        std_net_benefit = []
        for t in thresholds:
            tp = ((y_prob >= t) & (y_true == 1)).sum()
            fp = ((y_prob >= t) & (y_true == 0)).sum()
            nb = (tp / n) - (fp / n) * (t / (1 - t))
            std_nb = nb / prevalence if prevalence > 0 else np.nan
            std_net_benefit.append(std_nb)
        plt.plot(thresholds, std_net_benefit, label=name)
    
    plt.plot(thresholds, np.zeros_like(thresholds), '--', color='grey', label='Treat None')

    tp_all = (y_true == 1).sum()
    fp_all = (y_true == 0).sum()
    std_net_benefit_all = [(tp_all/n - fp_all/n * (t/(1-t))) / prevalence for t in thresholds]
    plt.plot(thresholds, std_net_benefit_all, '--', color='blue', label='Treat All')
    
    plt.xlabel("High Risk Threshold")
    plt.ylabel("Standardized Net Benefit")
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_StandardizedNetBenefit.png"), dpi=300)
    plt.show()

def plot_dca(X, y_true, dataset_name):
    thresholds = np.linspace(0.01, 0.99, 100)
    prevalence = (y_true == 1).mean()
    plt.figure(figsize=(7,6))
    plt.title(f"{dataset_name} Decision Curve Analysis")
    for name, model in fitted_models.items():
        try:
            y_prob = model.predict_proba(X)[:,1]
        except:
            continue
        net_benefit = []
        for t in thresholds:
            tp = ((y_prob >= t) & (y_true == 1)).sum()
            fp = ((y_prob >= t) & (y_true == 0)).sum()
            n = len(y_true)
            nb = (tp / n) - (fp / n) * (t / (1 - t))
            net_benefit.append(nb)
        plt.plot(thresholds, net_benefit, label=name)
    treat_none = np.zeros_like(thresholds)
    treat_all  = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    plt.plot(thresholds, treat_none, '--', color='grey', label='Treat None')
    plt.plot(thresholds, treat_all,  '--', color='blue', label='Treat All')
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_DCA.png"), dpi=300)
    plt.show()

cox_data_train = pd.concat([X_train_final, train[['Time','Relapse']]], axis=1)
cox_data_train = cox_data_train.loc[:, cox_data_train.nunique()>1]

cph = CoxPHFitter()
cph.fit(cox_data_train, duration_col="Time", event_col="Relapse")

def add_risk_score(df_features, df_time_event, model, median_risk=None):
    data = pd.concat([df_features, df_time_event], axis=1)
    data = data.loc[:, data.nunique()>1]
    data['risk_score'] = model.predict_partial_hazard(data)
    if median_risk is None:
        median_risk = data['risk_score'].median()
    data['risk_group'] = (data['risk_score'] >= median_risk).astype(int)
    return data, median_risk

cox_train, median_risk = add_risk_score(X_train_final, train[['Time','Relapse']], cph)
cox_valid, _           = add_risk_score(X_valid_final, valid[['Time','Relapse']], cph, median_risk)
cox_test, _            = add_risk_score(X_test_final,  test[['Time','Relapse']],  cph, median_risk)

cox_test['risk_group'] = (cox_test['risk_score'] >= cox_test['risk_score'].median()).astype(int)

def plot_km(cox_df, dataset_name):
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(7,6))
    for group in [0,1]:
        mask = cox_df['risk_group']==group
        times  = pd.to_numeric(cox_df.loc[mask,'Time'],  errors='coerce')
        events = pd.to_numeric(cox_df.loc[mask,'Relapse'],errors='coerce')
        valid_idx = times.notna() & events.notna()
        times, events = times[valid_idx], events[valid_idx]
        if len(times) == 0:
            print(f"[WARN] {dataset_name} group {group} has 0 samples, skip KM")
            continue
        kmf.fit(times, event_observed=events, label=f"Risk group {group}")
        kmf.plot_survival_function()
    plt.title(f"Kaplan-Meier Survival Curve ({dataset_name})")
    plt.xlabel("Progression-Free Survival (months)")
    plt.ylabel("Survival probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_KM.png"), dpi=300)
    plt.show()

datasets = {
    "Train": (X_train_final, y_train),
    "Valid": (X_valid_final, y_valid),
    "Test":  (X_test_final,  y_test)
}

for dname, (X, y_true) in datasets.items():
    print(f"\n===== Evaluating: {dname} =====")
    df_metrics = evaluate_models_on_dataset(X, y_true, dname)
    print(df_metrics[["Model","AUC","AUC_CI_low","AUC_CI_high","Sensitivity","Sensitivity_CI_low","Sensitivity_CI_high","Specificity","Specificity_CI_low","Specificity_CI_high"]])
    
    plot_calibration_curve(X, y_true, dname)
    
    plot_standardized_net_benefit(X, y_true, dname)
    
    plot_dca(X, y_true, dname)

plot_km(cox_train, "Train")
plot_km(cox_valid, "Valid")
plot_km(cox_test,  "Test")
def save_risk_scores(df, dataset_name, output_dir):
    save_path = os.path.join(output_dir, f"risk_scores_{dataset_name}.csv")
    df[['risk_score', 'risk_group', 'Time', 'Relapse']].to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved risk scores -> {save_path}")

save_risk_scores(cox_train, "Train", output_dir)
save_risk_scores(cox_valid, "Valid", output_dir)
save_risk_scores(cox_test,  "Test",  output_dir)
print("[DONE] All plots and metrics saved under:", output_dir)