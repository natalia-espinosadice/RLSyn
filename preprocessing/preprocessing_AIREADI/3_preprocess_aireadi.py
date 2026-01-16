import pandas as pd 
import os 
import numpy as np
import argparse 
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TEST_DATA_SIZES = [0.1] 
#all conditions except elevated A1C and prediabetes as that is too highly correlated with diabetes AND ekg categorical diagnoses 
AI_READI_FULL_CAT_COLS = ['Age-related macular degeneration', 'Arthritis', 'Cancer', 'Cataracts (1+ eyes)', 'Chronic pullmonary problems', 'Circulation problems', 'Diabetic retinopathy (1+)', 'Digestive problems', 'Marijuana user', 'Dry eye (1+)', 'Glaucoma (1+)', 'Hearing impairment',
'Heart attack', 'High blood cholesterol', 'High blood pressure', 'Kidney problems', 'Low blood pressure', 'Mild cognitive impairmen', 'Multiple sclerosis', 'Obesity', 'Osteoporosis','Other heart issues (pacemaker)', 'Other neurological conditions',
"Parkinson's disease", 'Retinal vascular occlusion', 'Stroke', 'Type 2 Diabetes', 'Urinary problems', 'ABNORMAL', 'BORDERLINE', 'NORMAL', 'OTHERWISE NORMAL', 'cube_visuospatial_executive_value', 'fluency_language_value', 'lettera_value', 'trails_visuospatial_executive_value']
#same wearables, NEW measurements and NEW ekgs 
AI_READI_FULL_CONTINUOUS_COLS = [
    #wearable (same as ai-readi-og)
    'heart_rate_mean', 'blood_glucose_mean', 'resp_rate_mean', 'stress_mean', 'blood_glucose_std', 'total_steps',  'resting_heart_rate', 'total_kcal', 'sleep_light_hrs', 'sleep_deep_hrs', 'sleep_rem_hrs', 'sleep_awake_hrs', 'act_generic_hrs', 'act_running_hrs','act_walking_hrs', 'act_sedentary_hrs', 
    #measurements: avoid a1c and the following columns with not enough data: [NT-proBNP (pg/mL)_value (pg/mL), Troponin-T (ng/L)_value (ng/L) ]
    'A/G Ratio_value', 'ALT (IU/L)_value (IU/L)', 'AST (IU/L)_value (IU/L)', 'Albumin (g/dL)_value (g/dL)', 'Alkaline Phosphatase (IU/L)_value (IU/L)', 'BUN (mg/dL)_value (mg/dL)', 'BUN/Creatinine ratio_value', "Bilirubin Total (mg/dL)_value (mg/dL)", 'C-Peptide (ng/mL)_value (ng/mL)', 'CRP - HS (mg/L)_value (mg/L)', 'Calcium (mg/dL)_value (mg/dL)', 'Carbon Dioxide, Total (mEq/L)_value (mEq/L)', 'Chloride (mEq/L)_value (mEq/L)', 'Creatinine (mg/dL)_value (md/dL)', 'Globulin, Total (g/dL)_value (g/dL)', 'Glucose (mg/dL)_value (mg/dL)', 'HDL Cholesterol (mg/dL)_value (mg/dL)',
    'INSULIN (ng/mL)_value (ng/mL)', 'LDL Cholesterol Calculation (mg/dL)_value (mg/dL)',  'Potassium (mEq/L)_value (mEq/L)', 'Protein, Total (g/dL)_value (g/dL)', 'Sodium (mEq/L)_value (mEq/L)', 'Total Cholesterol (mg/dL)_value (mg/dL)', 'Triglycerides (mg/dL)_value (mg/dL)',  'Urine Albumin (mg/dL)_value (mg/DL)', 'Urine Creatinine (mg/dL)_value (mg/DL)', 'clock_visuospatial_executive_time_value', 'delayed_recall_with_no_clue_time_value', 'digitspan_time_value', 'memory_trial1_time_value', 'memory_trial2_time_value', 'moca_abstraction_time_value', 'moca_orientation_time_value', 'moca_total_score_value', 'naming_time_value', 'repetition_time_value', 'subtraction_time_value',  'cube_visuospatial_executive_time_value', 'lettera_time_value', 'trails_visuospatial_executive_time_value',
    #multicategorical - try as numeric 
    'moca_combined_mis_score_value', #0 to 15
    'memory_trial1_value', #0 to 5 
    'memory_trial2_value', #2 to 5 
    'moca_abstraction_value', # 0 1 2 
    'moca_orientation_value', #3 to 6 
    'naming_value', # 0 to 3 
    'repetition_value', # 0 1 2 
    'subtraction_value', # 0 to 3 
    #ekgs 
    'Rate', 'PR', 'QRSD', 'QT', 'QTc', 'P', 'QRS', 'T',     
]

def main(): 
    cat_cols = AI_READI_FULL_CAT_COLS
    continuous_cols = AI_READI_FULL_CONTINUOUS_COLS
    save_folder_root = "../AI-READI-FULL"
    data_path = "ai_readi_full_data.csv" 
    stats = [] 
    print("Num continuous:", len(AI_READI_FULL_CONTINUOUS_COLS))
    print("Num categorical:", len(AI_READI_FULL_CAT_COLS))
    for seed in SEEDS: 
        for TEST_SIZE in TEST_DATA_SIZES: 
            save_folder = f"{save_folder_root}/preprocessed_data{round(1-TEST_SIZE, 1)}_seed{seed}"
            os.makedirs(save_folder, exist_ok=True)
            data = pd.read_csv(data_path)
            data = data.rename(columns={'Type II Diabetes': 'Type 2 Diabetes', 'Bilirubin, Total (mg/dL)_value (mg/dL)': 'Bilirubin Total (mg/dL)_value (mg/dL)'})
            
            #order columns and drop na
            data = data[cat_cols + continuous_cols + ["patient_id", "day"]]   
            data = data.dropna()
            print(len(data))
            na_counts = data.isna().sum()
            print(na_counts[na_counts > 0])

            #save unnormalized data 
            data3 = data.drop(columns=["day"])
            data3.to_csv(save_folder + '/original_data_with_patients.csv', index=False)
            
            #normalize columns 
            min_max_log = {}
            for col in continuous_cols:
                col_value = np.array(data[col])
                min_max_log[col] = [np.min(col_value), np.max(col_value)]
                norm_col_value = (col_value - min_max_log[col][0]) / (min_max_log[col][1] - min_max_log[col][0])
                data[col] = list(norm_col_value)
            
            #save min max log 
            np.save(save_folder+ '/min_max_log.npy', min_max_log)
            
            #save normalized data with patients 
            data_patient = data.drop(columns=["day"])
            data_patient = data_patient.to_csv(save_folder + '/preprocessed_data_with_patients.csv', index=False)
            
            #save normalized data without patients 
            data_patient = data.drop(columns=["patient_id"])
            data_patient.to_csv(save_folder + '/preprocessed_data_no_patients.csv', index=False)
            
            #split training testing by patient to avoid leakage 
            data_full = data.sample(frac=1, random_state=seed).reset_index(drop=True) 
            def split_by_group_and_label(df, test_size, seed):
                group = 'patient_id'
                label = 'Type 2 Diabetes'
                patients = (df.groupby(group)[label].max().rename('label').reset_index())
                test_patients = (patients.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=test_size, random_state=seed)))
                test_groups = set(test_patients[group])
                train_df = df[~df[group].isin(test_groups)].reset_index(drop=True)
                test_df  = df[df[group].isin(test_groups)].reset_index(drop=True)
                return train_df, test_df
            training_data_df, testing_data_df = split_by_group_and_label(data_full, TEST_SIZE, seed)
            
            #save normalized training and testing 
            training_data_df = training_data_df.drop(columns=["patient_id", "day"])
            testing_data_df = testing_data_df.drop(columns=["patient_id", "day"])
            training_data_df.to_csv(save_folder + '/normalized_training_data.csv', index=False)
            testing_data_df.to_csv(save_folder + '/normalized_testing_data.csv', index=False)
            
            #also save unnormalized training and testing  
            min_max_log = np.load(save_folder + '/min_max_log.npy', allow_pickle=True).item()
            for key, min_max in min_max_log.items():
                min_, max_ = min_max[0], min_max[1]
                col_values = np.array(training_data_df[key])
                training_data_df[key] = (1 - col_values)*min_ + col_values*max_
                col_values = np.array(testing_data_df[key])
                testing_data_df[key] = (1 - col_values)*min_ + col_values*max_
            training_data_df.to_csv(save_folder + '/original_training_data.csv', index=False)
            testing_data_df.to_csv(save_folder + '/original_testing_data.csv', index=False)

            stats.append([round(1-TEST_SIZE, 2), seed, round(len(testing_data_df)/len(data_full), 2), round(training_data_df["Type 2 Diabetes"].mean(), 2), round(testing_data_df["Type 2 Diabetes"].mean(), 2)])

    for row in stats: 
        print(row)

if __name__ == "__main__":
    main()