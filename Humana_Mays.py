import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import LabelEncoder

# Data Loading

# RAW
humana_mays_target_members = pd.read_csv('/Users/jyunru/Downloads/Training_final/humana_mays_target_members.csv')
humana_mays_target_member_details = pd.read_csv('/Users/jyunru/Downloads/Training_final/humana_mays_target_member_details.csv')
humana_mays_target_member_visit_claims = pd.read_csv('/Users/jyunru/Downloads/Training_final/humana_mays_target_member_visit_claims.csv')
humana_mays_target_member_conditions = pd.read_csv('/Users/jyunru/Downloads/Training_final/humana_mays_target_member_conditions.csv')
additional_features = pd.read_csv('/Users/jyunru/Downloads/Training_final/Additional Features.csv')
control_point = pd.read_csv('/Users/jyunru/Downloads/Training_final/Control Point.csv')
cost_utilization = pd.read_csv('/Users/jyunru/Downloads/Training_final/Cost & Utilization.csv')
demographics = pd.read_csv('/Users/jyunru/Downloads/Training_final/Demographics.csv')
member_data = pd.read_csv('/Users/jyunru/Downloads/Training_final/MEMBER_DATA.csv')
pharmacy_utilization = pd.read_csv('/Users/jyunru/Downloads/Training_final/Pharmacy Utilization.csv')
quality_data = pd.read_csv('/Users/jyunru/Downloads/Training_final/QUALITY_DATA.csv')
sales_channel = pd.read_csv('/Users/jyunru/Downloads/Training_final/Sales Channel.csv')
social_determinants_of_health = pd.read_csv('/Users/jyunru/Downloads/Training_final/Social Determinants of Health.csv')
web_activity = pd.read_csv('/Users/jyunru/Downloads/Training_final/Web Activity.csv')




## 1. Additional Features Dataset Processing

# Check for duplicates in 'id'
duplicate_count = additional_features['id'].duplicated().sum()
print(f"Number of duplicate 'id's: {duplicate_count}")

#Checked the cms_frailty_ind column for missing values and dropped the rows (When checking the csv file, the rows that had missing values in cms_frailty_ind, the other values were all missing also)
additional_features['cms_frailty_ind'].isnull().sum()
additional_features.dropna(subset=['cms_frailty_ind'], inplace=True)

#Select the columns we are going to use in 'target member details' and merge together
sel_humana_mays_target_member_details = humana_mays_target_member_details[['id', 'sex_cd', 'state_of_residence', 'county_of_residence']]
additional_features = additional_features.merge(sel_humana_mays_target_member_details, on = 'id')
variables = additional_features.columns[3:8]

#Impute missing values by the group mean based on 'state_of_residence' and 'county_of_residence'
additional_features[variables] = additional_features.groupby(['state_of_residence', 'county_of_residence'])[variables].transform(lambda x: x.fillna(x.mean()))

#Optional: If there are still missing values (e.g., either 'state_of_residence' or 'county_of_residence' is NaN), fill with overall mean
additional_features[variables] = additional_features[variables].fillna(additional_features[variables].mean())
additional_features = additional_features.drop(['sex_cd','state_of_residence', 'county_of_residence'],axis=1)

#These two columns represent the payment amount, and there were no other 0s, so filled in with 0 to make it numerice (float)
additional_features['cms_tot_ma_payment_amt'].fillna(0.0, inplace = True)
additional_features['cms_tot_partd_payment_amt'].fillna(0.0, inplace = True)
cleaned_additional_features = additional_features.copy()













## 2. Control Point Dataset Processing

# Select specified columns and fill missing values with 0.0
selected_columns = control_point[['cnt_cp_emails_pmpm_ct', 'cnt_cp_print_pmpm_ct', 'cnt_cp_vat_pmpm_ct',
                                  'cnt_cp_webstatement_pmpm_ct', 'cnt_cp_livecall_pmpm_ct', 'id']]
selected_columns_filled = selected_columns.fillna(0.0)

# Define a function to categorize each column
def categorize_column(column):
    column = pd.to_numeric(column, errors='coerce')
    zero_mask = column == 0
    non_zero_values = column[~zero_mask]
    if non_zero_values.size < 5:
        categorized_non_zero = pd.Series(['non-zero'] * non_zero_values.size, index=non_zero_values.index)
    else:
        categorized_non_zero = pd.qcut(non_zero_values, q=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                       labels=['20%', '40%', '60%', '80%', '100%'], duplicates='drop')
    result = pd.Series(index=column.index, dtype="object")
    result[zero_mask] = '0%'
    result[~zero_mask] = categorized_non_zero
    return result

# Apply the categorization function to selected columns
for col in selected_columns_filled.columns:
    if col != 'id':
        try:
            selected_columns_filled[f'{col}_group'] = categorize_column(selected_columns_filled[col])
        except Exception as e:
            print(f"Error processing column {col}: {e}")

# Drop unnecessary columns
selected_columns_filled.drop(['cnt_cp_livecall_pmpm_ct', 'cnt_cp_livecall_pmpm_ct_group'], axis=1, inplace=True)
cleaned_control_point = selected_columns_filled.copy()










## 3. Cost Utilization Dataset Processing

# Assume the cleaned data is saved as a CSV file
cleaned_cost_utilization = pd.read_csv('/Users/jyunru/Downloads/Training_final/cleaned/cu.csv')









## 4. Demographics Dataset Processing

# Assume the cleaned data is saved as a CSV file
cleaned_demographics = pd.read_csv('/Users/jyunru/Downloads/Training_final/cleaned/Demographics.csv')











## 5. Humana Mays Target Member Conditions Dataset Processing

df_conditions = humana_mays_target_member_conditions.copy()

# Define all possible 'cond_key' and 'hcc_model_type' values
cond_keys = [226, 94, 329, 81, 47, 280, 238, 37, 48, 108, 19, 224, 328, 38, 327, 18,
             122, 136, 134, 161, 79, 23, 189, 39, 96, 59, 88, 227, 52, 199, 264, 263,
             139, 22, 126, 50, 201, 127, 20, 75, 383, 326, 298, 155, 93, 40, 300, 193,
             137, 21, 112, 186, 29, 111, 85, 463, 63, 109, 253, 180, 380, 65, 1, 401,
             92, 154, 181, 279, 151, 115, 182, 138, 196, 17, 409, 382, 381, 114, 80,
             211, 202, 153, 64, 72, 198, 125, 278, 254, 454, 192, 36, 200, 9, 10, 82,
             387, 221, 223, 169, 12, 46, 106, 276, 55, 379, 27, 28, 35, 190, 197, 191,
             107, 222, 277, 188, 158, 11, 56, 57, 78, 62, 159, 51, 8, 103, 124, 58,
             157, 49, 104, 77, 70, 34, 71, 73, 60, 54, 76, 74, 110]
hcc_model_types = ['MEDICAL', 'ESRD']

# Define a function to create pivot tables and indicator variables
def pivot_indicator(df, group_col, group_list, id_col='id', prefix=''):
    df['indicator'] = 1
    pivot_df = df.pivot_table(
        index=id_col,
        columns=group_col,
        values='indicator',
        aggfunc='max',
        fill_value=0
    )
    all_columns = sorted(group_list)
    pivot_df = pivot_df.reindex(columns=all_columns, fill_value=0)
    pivot_df.columns = [f'{prefix}{col}' for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    return pivot_df

# Process 'cond_key'
pivot_cond_key = pivot_indicator(
    df_conditions.copy(),
    group_col='cond_key',
    group_list=cond_keys,
    id_col='id',
    prefix='cond_key_'
)

# Process 'hcc_model_type'
pivot_hcc_model_type = pivot_indicator(
    df_conditions.copy(),
    group_col='hcc_model_type',
    group_list=hcc_model_types,
    id_col='id',
    prefix='hcc_model_type_'
)

# Merge the two pivot tables
final_conditions_df = pd.merge(pivot_cond_key, pivot_hcc_model_type, on='id', how='outer')

# Convert indicator columns to integer type
indicator_columns = [col for col in final_conditions_df.columns if col != 'id']
final_conditions_df[indicator_columns] = final_conditions_df[indicator_columns].astype(int)

cleaned_humana_mays_target_member_conditions = final_conditions_df.copy()











## 6. Humana Mays Target Member Details Dataset Processing

# Check for duplicates in 'id'
duplicates_details = humana_mays_target_member_details['id'].duplicated().sum()

# Fill missing values in categorical columns with 'Unknown'
humana_mays_target_member_details.fillna('Unknown', inplace=True)

cleaned_humana_mays_target_member_details = humana_mays_target_member_details.copy()














## 7. Humana Mays Target Member Visit Claims Dataset Processing

# Remove unnecessary columns
humana_mays_target_member_visit_claims.drop(['clm_unique_key', 'serv_date_skey'], axis=1, inplace=True)
columns = [humana_mays_target_member_visit_claims.columns[1]] + humana_mays_target_member_visit_claims.columns[4:6].tolist() + humana_mays_target_member_visit_claims.columns[8:24].tolist()

for col in columns:
    humana_mays_target_member_visit_claims[col] = humana_mays_target_member_visit_claims[col].apply(lambda x: 'Y' if x == 'Y' else np.nan)

humana_mays_target_member_visit_claims.replace('Y', 1, inplace=True)
humana_mays_target_member_visit_claims.fillna(0, inplace=True)

grouped = humana_mays_target_member_visit_claims.groupby(['id', 'dos_year'])[columns].max().reset_index()

pivoted = grouped.pivot(index='id', columns='dos_year', values=columns)

if isinstance(pivoted.columns, pd.MultiIndex):
    pivoted.columns = [f"{col}_{year}" for col, year in pivoted.columns]
else:
    pivoted.columns = [f"{col}_{year}" for col, year in zip(columns, pivoted.columns)]

pivoted = pivoted.reset_index()
pivoted = pivoted.fillna(0)

pivoted = pivoted.apply(lambda x: x.astype(int) if x.dtype == 'float' else x)

#Check how many values are 1
(pivoted == 1).sum().sum()

cleaned_humana_mays_target_member_visit_claims = pivoted.copy()











## 8. Member Data Dataset Processing

# Assume the cleaned data is saved as a CSV file
cleaned_member_data = pd.read_csv('/Users/jyunru/Downloads/Training_final/cleaned/MEMBER_DATA.csv')





## 9. Pharmacy Utilization Dataset Processing

# Assume the cleaned data is saved as a CSV file
cleaned_pharmacy_utilization = pd.read_csv('/Users/jyunru/Downloads/Training_final/cleaned/pu.csv')








## 10. Quality Data Dataset Processing

# Load the cleaned QUALITY_DATA
cleaned_quality_data = pd.read_csv('/Users/jyunru/Downloads/Training_final/cleaned/QUALITY_DATA.csv')

# Drop unnecessary columns
cleaned_quality_data = cleaned_quality_data.drop(['measure_desc', 'base_event_date', 'eligible_cnt'], axis=1)

# Process 'measure_type' and 'measure_name'
df_quality = cleaned_quality_data.copy()

measure_types = ['Patient Safety', 'HEDIS', 'Patient Experience']
measure_names = [
    'ADH (DIAB)', 'ADH (ACE)', 'ABA', 'ADH (STATIN)', 'PCR', 'TRC (MRP)',
    'CBP', 'COL', 'CDC (EYE)', 'CDC (NPH)', 'CDC (HbA1c)', 'BCS',
    'SUPD', 'SPC STATIN', 'ART', 'OMW', 'COA (MDR)', 'COA (PNS)',
    'COA (FSA)', 'COL (45-50)', 'EED', 'FMC', 'HBD', 'KED', 'MRP',
    'TRC (PED)', 'ESA', 'SWT', 'EGR', 'RXC', 'ETA', 'MDR', 'ASV',
    'TFP', 'SFT', 'TBC', 'SBT', 'TEX'
]
years = [2020, 2021, 2022]

def pivot_indicator_quality(df, group_col, group_list, years, id_col='id', value_col='indicator'):
    df[f'{group_col}_year'] = df[group_col] + "_" + df['measurement_year'].astype(str)
    df[value_col] = 1
    pivot_df = df.pivot_table(
        index=id_col,
        columns=f'{group_col}_year',
        values=value_col,
        aggfunc='max',
        fill_value=0
    )
    all_combinations = [f"{g}_{year}" for g in group_list for year in years]
    sorted_columns = sorted(all_combinations, key=lambda x: (group_list.index(x.split('_')[0]), x.split('_')[1]))
    pivot_df = pivot_df.reindex(columns=sorted_columns, fill_value=0)
    pivot_df = pivot_df.reset_index()
    return pivot_df

# Process 'measure_type'
pivot_type = pivot_indicator_quality(df_quality.copy(), 'measure_type', measure_types, years, id_col='id', value_col='indicator_type')

# Process 'measure_name'
pivot_name = pivot_indicator_quality(df_quality.copy(), 'measure_name', measure_names, years, id_col='id', value_col='indicator_name')

# Merge the two pivot tables
final_quality_df = pd.merge(pivot_type, pivot_name, on='id', how='outer')

# Aggregate 'compliant_cnt' by 'id' and 'measurement_year'
compliant_counts = df_quality.groupby(['id', 'measurement_year'])['compliant_cnt'].sum().unstack(fill_value=0).reset_index()

# Rename columns
compliant_counts.columns = ['id'] + [f'compliant_cnt_{int(col)}' for col in compliant_counts.columns if col != 'id']

# Merge 'compliant_cnt' data into 'final_quality_df'
final_quality_df = final_quality_df.merge(compliant_counts, on='id', how='left')

# Fill NaN with 0 and ensure data types are integers
final_quality_df = final_quality_df.fillna(0)
indicator_cols = [col for col in final_quality_df.columns if col != 'id' and not col.startswith('compliant_cnt_')]
final_quality_df[indicator_cols] = final_quality_df[indicator_cols].astype(int)
compliant_cnt_cols = [col for col in final_quality_df.columns if col.startswith('compliant_cnt_')]
final_quality_df[compliant_cnt_cols] = final_quality_df[compliant_cnt_cols].astype(int)

cleaned_quality_data = final_quality_df.copy()











## 11. Sales Channel Dataset Processing

# Drop rows where 'id' or 'channel' is missing
cleaned_sales_channel = sales_channel.dropna(subset=['id', 'channel'])








## 12. Social Determinants of Health Dataset Processing

# Drop the 'rwjf_drinkwater_violate_ind' column
sel_social_determinants_of_health = social_determinants_of_health.drop('rwjf_drinkwater_violate_ind', axis=1)

# Select necessary columns from 'humana_mays_target_member_details' and merge
sel_details = humana_mays_target_member_details[['id', 'sex_cd', 'state_of_residence', 'county_of_residence']]
sel_social_determinants_of_health = sel_social_determinants_of_health.merge(sel_details, on='id')

# Group by 'state_of_residence' and 'county_of_residence' and fill missing values with mean
variables = sel_social_determinants_of_health.columns[0:76]
sel_social_determinants_of_health[variables] = sel_social_determinants_of_health.groupby(
    ['state_of_residence', 'county_of_residence'])[variables].transform(lambda x: x.fillna(x.mean()))

# Fill remaining missing values with overall mean
sel_social_determinants_of_health[variables] = sel_social_determinants_of_health[variables].fillna(
    sel_social_determinants_of_health[variables].mean())

# Drop unnecessary columns
sel_social_determinants_of_health.drop(['sex_cd', 'state_of_residence', 'county_of_residence'], axis=1, inplace=True)

cleaned_social_determinants_of_health = sel_social_determinants_of_health.copy()















## 13. Web Activity Dataset Processing

# Assume the cleaned data is saved as a CSV file
cleaned_web_activity = pd.read_csv('/Users/jyunru/Downloads/Training_final/cleaned/wa.csv')
















# Data Merging

dfs_to_merge = [
    humana_mays_target_members,
    cleaned_humana_mays_target_member_conditions,
    cleaned_humana_mays_target_member_visit_claims,
    cleaned_humana_mays_target_member_details,
    cleaned_additional_features,
    cleaned_control_point,
    cleaned_cost_utilization,
    cleaned_demographics,
    cleaned_member_data,
    cleaned_pharmacy_utilization,
    cleaned_quality_data,
    cleaned_sales_channel,
    cleaned_social_determinants_of_health,
    cleaned_web_activity
]

merged_df = reduce(lambda left, right: pd.merge(left, right, on='id', how='left'), dfs_to_merge)

# Drop specified columns
merged_df.drop(['calendar_year', 'product_type', 'plan_category','mco_contract_nbr', 'region', 'race', 'county_of_residence', 'login_count_0','login_count_1','login_count_2','login_count_3','login_count_4','login_count_5','login_count_6','login_count_7','login_count_8','login_count_9','login_count_10','login_count_11'], axis=1, inplace=True)

# Data cleaning again
merged_df.iloc[:, 2:182] = merged_df.iloc[:, 2:182].fillna(0)
for col in range(190, 201):
    mode_value = merged_df.iloc[:, col].mode()[0]  
    merged_df.iloc[:, col] = merged_df.iloc[:, col].fillna(mode_value)  
for col in range(209, 244):
    mode_value = merged_df.iloc[:, col].mode()[0] 
    merged_df.iloc[:, col] = merged_df.iloc[:, col].fillna(mode_value)  
for col in range(255, 271):
    mode_value = merged_df.iloc[:, col].mode()[0]  
    merged_df.iloc[:, col] = merged_df.iloc[:, col].fillna(mode_value)  
merged_df.iloc[:, 271:397] = merged_df.iloc[:, 271:397].fillna(0)
merged_df.iloc[:, 397] = merged_df.iloc[:, 397].fillna("Unknown")
for col in range(474, 476):
    mode_value = merged_df.iloc[:, col].mode()[0]  
    merged_df.iloc[:, col] = merged_df.iloc[:, col].fillna(mode_value)


# Transfer object columns into label columns
# Select object columns
object_columns = merged_df.select_dtypes(include='object').columns.tolist()

# Create a copy with only object columns and 'id'
merged_df_object = merged_df[object_columns + ['id']]
merged_df = merged_df.drop(object_columns, axis=1)

# Create an empty dictionary to store label encoders for each column
label_encoders = {}

# Apply LabelEncoder to each object column
for col in object_columns:
    le = LabelEncoder()
    merged_df_object[f'{col}_label'] = le.fit_transform(merged_df_object[col])
    label_encoders[col] = le

# Create a dictionary to store the label mappings for each object column
label_dict = {col: dict(zip(le.classes_, range(len(le.classes_)))) for col, le in label_encoders.items()}

merged_df_object = merged_df_object.drop(object_columns, axis=1)

merged_df = merged_df.merge(merged_df_object, on='id', how='left')
# Save the final data cleaning result
merged_df.to_csv('merged_df.csv')




































# Xgboost
# Import necessary libraries
import pandas as pd
import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt

# Sample 150,000 rows from the dataset
aa = merged_df.sample(n=150000)

# Drop specific columns (columns 144 to 181 and 257 to 382)
columns_to_drop = list(range(144, 182)) + list(range(257, 383))
aa = aa.drop(aa.columns[columns_to_drop], axis=1)

# Define the target variable 'y' and feature set 'x'
y = aa['preventive_visit_gap_ind']
x = aa.drop(columns=['preventive_visit_gap_ind'])

# Feature selection using feature importances from an initial XGBoost model
initial_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    verbosity=0
)
initial_model.fit(x, y)

# Get feature importances and select top 60 features
importances = initial_model.feature_importances_
feature_importances = pd.Series(importances, index=x.columns)
selected_features = feature_importances.sort_values(ascending=False).head(60).index.tolist()

# Initialize a list to record results
results = []

# Set the number of iterations (e.g., 20,000)
for i in range(20000):
    # Randomly select 30 features from the selected features
    selected_features_iter = random.sample(selected_features, 30)
    x_iter = aa[selected_features_iter]

    # Split the dataset into training and testing sets, maintaining class proportions
    x_train, x_test, y_train, y_test = train_test_split(
        x_iter, y, test_size=0.2, stratify=y
    )

    # Calculate scale_pos_weight to handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Initialize XGBClassifier with calculated scale_pos_weight
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        verbosity=0,
        scale_pos_weight=scale_pos_weight
    )

    # Train the model
    model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        verbose=False
    )

    # Predict probabilities and classes
    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Record the results of this iteration
    results.append({
        'iteration': i + 1,
        'auc': auc,
        'accuracy': accuracy,
        'selected_features': selected_features_iter
    })

    # Print the results of this iteration
    print(f"Iteration {i + 1}: AUC={auc:.4f}, Accuracy={accuracy:.4f}")

# Convert the results to a DataFrame for further analysis
results_df = pd.DataFrame(results)

# Find the iteration with the highest AUC
best_run = results_df.loc[results_df['auc'].idxmax()]

# Print the best results
print("\nBest Result:")
print(f"AUC: {best_run['auc']:.4f}")
print(f"Accuracy: {best_run['accuracy']:.4f}")
print(f"Selected Features: {best_run['selected_features']}")










# -------------------------------------
# Use the best selected features to build a final model
# -------------------------------------

# Define features and target using the best selected features
feature_columns = ['plan_benefit_package_id', 'rwjf_population', 'rwjf_disconnect_youth_pct', 'unattributed_provider_label', 'rwjf_insufficient_sleep_pct', 'riskarr_upside', 'rwjf_child_free_lunch_pct', 'total_cob_paid_pmpm_cost', 'veteran_ind_label', 'rx_tier_1_pmpm_ct', 'rwjf_adult_obesity_pct', 'age', 'riskarr_downside', 'cond_key_382', 'disabled_ind_label', 'rwjf_uninsured_child_pct', 'cond_key_329', 'days_since_last_login', 'cms_tot_ma_payment_amt', 'rwjf_mammography_pct', 'cond_key_37', 'rwjf_hawaiian_race_pct', 'rwjf_physical_distress_pct', 'riskarr_rewards', 'hcc_model_type_MEDICAL', 'oontwk_mbr_resp_pmpm_cost', 'total_net_paid_pmpm_cost', 'rwjf_mental_distress_pct', 'lis_ind_label', 'days_since_last_clm']
x = aa[feature_columns]
y = aa['preventive_visit_gap_ind']

# Split the data with stratification
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y
)

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Define hyperparameters
params = {
    'subsample': 0.9,
    'n_estimators': 250,
    'max_depth': 9,
    'learning_rate': 0.07,
    'colsample_bytree': 0.3,
    'gamma': 0.1,
    'min_child_weight': 4,
    'reg_alpha': 0.1,
    'reg_lambda': 3,
    'scale_pos_weight': scale_pos_weight
}

# Initialize XGBClassifier with hyperparameters
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    verbosity=0,
    **params
)

# Train the model
model.fit(
    x_train, y_train,
    eval_set=[(x_test, y_test)],
    verbose=False
)

# Predict probabilities and classes
y_prob = model.predict_proba(x_test)[:, 1]
y_pred = model.predict(x_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"AUC: {auc:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(class_report)
print("Confusion Matrix:")
print(conf_matrix)















# -------------------------------------
# Hyperparameter tuning using RandomizedSearchCV
# -------------------------------------

from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'subsample': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth':a
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.09],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 2, 3, 4, 5],
    'reg_alpha': [0, 0.005, 0.01, 0.05, 0.1, 0.5],
    'reg_lambda': [1, 2, 3]
}

# Set up RandomizedSearchCV to find the best model
grid_search = RandomizedSearchCV(
    xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        verbosity=0
    ),
    param_distributions=param_grid,
    n_iter=100,  # Number of parameter settings that are sampled
    scoring='roc_auc',
    cv=10,       # Number of cross-validation folds
    verbose=1,
    n_jobs=-1    # Use all available CPU cores
)

# Train the model using RandomizedSearchCV
grid_search.fit(x_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Predict probabilities using the best model
y_prob = best_model.predict_proba(x_test)[:, 1]  # Get probabilities for class 1

# Calculate AUC
auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.2f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# Calculate accuracy using a threshold of 0.5
y_pred = (y_prob >= 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Accuracy after tuning: {accuracy:.2f}")

# Print the best hyperparameters found
print("Best parameters found: ", grid_search.best_params_)










