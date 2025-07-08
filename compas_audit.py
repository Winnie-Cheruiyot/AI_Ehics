import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import AIF360 components
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder # Added LabelEncoder
from IPython.display import Markdown, display # For displaying markdown in notebooks

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# --- 1. Load and Prepare the COMPAS Dataset ---
# The COMPAS dataset is often pre-processed for AIF360 examples.
# We'll simulate loading and initial preprocessing.
print("Loading and preparing COMPAS dataset...")

# Create a synthetic dataset that mimics the structure and known biases of COMPAS
# In a real scenario, you'd load the actual COMPAS dataset.
# This synthetic data will have 'race' as the sensitive attribute and 'two_year_recid' as the target.
np.random.seed(42)
num_samples = 2000

data = {
    'age': np.random.randint(18, 70, num_samples),
    'c_charge_degree': np.random.choice(['F', 'M'], num_samples, p=[0.6, 0.4]), # Felony/Misdemeanor
    'race': np.random.choice(['Caucasian', 'African-American', 'Other'], num_samples, p=[0.4, 0.4, 0.2]),
    'sex': np.random.choice(['Male', 'Female'], num_samples, p=[0.7, 0.3]),
    'priors_count': np.random.randint(0, 10, num_samples), # Number of prior offenses
    'days_since_arrest': np.random.uniform(0, 365, num_samples).round(0),
    'decile_score': np.random.randint(1, 11, num_samples), # COMPAS risk score (1-10)
    'two_year_recid': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) # 0: No recidivism, 1: Recidivism (target)
}
df = pd.DataFrame(data)

# Introduce synthetic bias: African-Americans are more likely to have higher decile scores and recidivism
df.loc[df['race'] == 'African-American', 'decile_score'] = np.clip(df.loc[df['race'] == 'African-American', 'decile_score'] + np.random.randint(1, 4, sum(df['race'] == 'African-American')), 1, 10)
df.loc[df['race'] == 'African-American', 'two_year_recid'] = np.random.choice([0, 1], sum(df['race'] == 'African-American'), p=[0.55, 0.45]) # Higher recidivism rate

df.loc[df['race'] == 'Caucasian', 'decile_score'] = np.clip(df.loc[df['race'] == 'Caucasian', 'decile_score'] - np.random.randint(0, 2, sum(df['race'] == 'Caucasian')), 1, 10)
df.loc[df['race'] == 'Caucasian', 'two_year_recid'] = np.random.choice([0, 1], sum(df['race'] == 'Caucasian'), p=[0.8, 0.2]) # Lower recidivism rate


# Define sensitive attributes and favorable/unfavorable labels
# 'race' is the sensitive attribute
sensitive_attribute_names = ['race']

# Define the label column and favorable label (non-recidivism)
label_name = 'two_year_recid'
favorable_label = 0 # 0 for no recidivism, 1 for recidivism (unfavorable)

# Convert to AIF360 StandardDataset format
# Ensure all columns are handled (numeric, categorical, sensitive)
# For simplicity, we'll treat decile_score > 4 as 'high_risk' (predictive outcome)
# and two_year_recid as the true outcome.
df['high_risk'] = (df['decile_score'] > 4).astype(int) # This is our 'prediction' based on COMPAS score

# Map categorical features to numerical for AIF360
df_processed = df.copy()
for col in ['c_charge_degree', 'sex', 'race']:
    df_processed[col] = LabelEncoder().fit_transform(df_processed[col])

# Get encoded values for privileged and unprivileged groups for use in metrics
encoder = LabelEncoder().fit(df['race']) # Fit on original 'race' column to get consistent encoding
encoded_caucasian = encoder.transform(['Caucasian'])[0]
encoded_african_american = encoder.transform(['African-American'])[0]

# Define these for use in metrics
unprivileged_groups_metric = [{'race': encoded_african_american}]
privileged_groups_metric = [{'race': encoded_caucasian}]


# Create AIF360 dataset
# Removed privileged_classes and unprivileged_classes from StandardDataset constructor
aif_dataset = StandardDataset(
    df=df_processed,
    label_name=label_name,
    favorable_classes=[favorable_label],
    protected_attribute_names=sensitive_attribute_names,
    features_to_drop=['decile_score', 'high_risk'] # Drop score and our derived prediction for training, we'll use 'high_risk' later as the prediction
)

# Split into train and test
dataset_train, dataset_test = aif_dataset.split([0.7], shuffle=True)

# --- 2. Analyze Initial Bias ---
print("\n--- Initial Bias Analysis (COMPAS Scores) ---")

# Create a BinaryLabelDatasetMetric for the test set
metric_orig_test = BinaryLabelDatasetMetric(
    dataset_test,
    unprivileged_groups=unprivileged_groups_metric, # Use the defined metric groups
    privileged_groups=privileged_groups_metric      # Use the defined metric groups
)

# Disparate Impact
# Ratio of favorable outcomes for unprivileged group to privileged group
di = metric_orig_test.disparate_impact()
print(f"Disparate Impact (P(favorable|unprivileged) / P(favorable|privileged)): {di:.4f}")
if di < 0.8 or di > 1.25:
    print("  --> Significant disparate impact detected (common threshold 0.8-1.25).")
else:
    print("  --> Disparate impact within acceptable range (common threshold 0.8-1.25).")

# Mean Difference of Favorable Outcome
mdfo = metric_orig_test.mean_difference()
print(f"Mean Difference of Favorable Outcome (P(favorable|unprivileged) - P(favorable|privileged)): {mdfo:.4f}")
if mdfo < -0.1 or mdfo > 0.1:
    print("  --> Significant mean difference detected (common threshold +/- 0.1).")
else:
    print("  --> Mean difference within acceptable range (common threshold +/- 0.1).")

# --- 3. Train a Simple Classifier and Evaluate Bias ---
print("\n--- Classifier Training and Bias Evaluation ---")

# Prepare data for sklearn Logistic Regression
X_train = dataset_train.features
y_train = dataset_train.labels.ravel()
X_test = dataset_test.features
y_test = dataset_test.labels.ravel()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic Regression model
lr_model = LogisticRegression(solver='liblinear', random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Create AIF360 dataset with predictions
dataset_test_pred = dataset_test.copy()
dataset_test_pred.labels = y_pred_lr.reshape(-1, 1) # Set predictions as labels for metric calculation

# Create ClassificationMetric for the test set with predictions
metric_lr_test = ClassificationMetric(
    dataset_test,
    dataset_test_pred,
    unprivileged_groups=unprivileged_groups_metric, # Use the defined metric groups
    privileged_groups=privileged_groups_metric      # Use the defined metric groups
)

# Evaluate fairness metrics for the classifier
print("\nFairness Metrics for Logistic Regression Classifier:")
print(f"  Accuracy: {lr_model.score(X_test_scaled, y_test):.4f}")
print(f"  Statistical Parity Difference (SPD): {metric_lr_test.statistical_parity_difference():.4f}")
print(f"  Equal Opportunity Difference (EOD) (TPR difference): {metric_lr_test.equal_opportunity_difference():.4f}")
print(f"  Average Odds Difference (AOD) (Avg of TPR and FPR diff): {metric_lr_test.average_odds_difference():.4f}")
print(f"  False Positive Rate Difference (FPRD): {metric_lr_test.false_positive_rate_difference():.4f}")

# --- 4. Generate Visualizations ---
print("\n--- Generating Visualizations ---")

# Get counts for each group and outcome for original data
orig_df = dataset_test.convert_to_dataframe()
orig_df['race_label'] = orig_df['race'].map({encoded_caucasian: 'Caucasian', encoded_african_american: 'African-American'})
orig_df['recidivism_label'] = orig_df['two_year_recid'].map({0: 'No Recidivism', 1: 'Recidivism'})

plt.figure(figsize=(8, 6))
sns.countplot(data=orig_df, x='race_label', hue='recidivism_label', palette='coolwarm')
plt.title('Actual Recidivism Distribution by Race (Test Set)')
plt.xlabel('Race')
plt.ylabel('Count')
plt.show()

# Visualize False Positive Rates (FPR) by group
# For this, we need to manually calculate FPR for each group
# We'll use the 'high_risk' prediction based on decile score for this visualization
# as it's directly from the COMPAS system.
# For the LR model, we'd use y_test and y_pred_lr

# Calculate FPR for original COMPAS scores (high_risk)
fpr_data = []
for group_label, group_val in zip(['Caucasian', 'African-American'], [encoded_caucasian, encoded_african_american]):
    group_df = orig_df[orig_df['race'] == group_val]
    # Assuming 'high_risk' is 1 (positive prediction) and 'two_year_recid' is 1 (actual positive)
    # FPR = FP / (FP + TN) = FP / Actual Negatives
    actual_negatives = sum(group_df['two_year_recid'] == 0)
    false_positives = sum((group_df['high_risk'] == 1) & (group_df['two_year_recid'] == 0))
    fpr = false_positives / actual_negatives if actual_negatives > 0 else 0
    fpr_data.append({'Race': group_label, 'False Positive Rate': fpr})

fpr_df = pd.DataFrame(fpr_data)

plt.figure(figsize=(7, 5))
sns.barplot(x='Race', y='False Positive Rate', data=fpr_df, palette='plasma')
plt.title('False Positive Rate Disparity in COMPAS Scores by Race')
plt.ylabel('False Positive Rate (Incorrectly Predicted High Risk)')
plt.ylim(0, max(fpr_df['False Positive Rate']) * 1.2)
plt.show()

print("\nAudit complete. Proceed to the report.")
