
import numpy as np
import pandas as pd
import wfdb
import matplotlib.pyplot as plt
import os 
import pandas as pd 
import numpy as np
import warnings
import re
import argparse 
from pathlib import Path
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 30)   
pd.set_option("display.max_rows",  50) 
pd.set_option("display.width",  None)  
PATIENT_IDS = [1023, 1024, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1079, 1080, 1081, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1103, 1104, 1105, 1106, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1128, 1129, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1143, 1144, 1145, 1146, 1148, 1149, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1163, 1164, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1188, 1189, 1192, 1193, 1194, 1195, 1196, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1280, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1359, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1372, 1373, 1374, 1376, 1377, 1378, 1379, 1380, 1381, 1383, 1384, 1385, 4009, 4019, 4022, 4026, 4028, 4030, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4048, 4049, 4051, 4052, 4054, 4055, 4056, 4057, 4058, 4059, 4060, 4061, 4062, 4064, 4065, 4066, 4067, 4068, 4072, 4073, 4074, 4075, 4076, 4077, 4078, 4082, 4087, 4088, 4089, 4091, 4095, 4101, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4127, 4128, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4138, 4139, 4140, 4141, 4142, 4143, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4153, 4154, 4155, 4156, 4157, 4158, 4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4175, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4196, 4200, 4201, 4202, 4203, 4205, 4206, 4207, 4208, 4210, 4211, 4212, 4215, 4216, 4219, 4220, 4221, 4222, 4224, 4225, 4226, 4227, 4228, 4229, 4230, 4231, 4232, 4234, 4235, 4236, 4237, 4239, 4240, 4241, 4244, 4245, 4246, 4247, 4248, 4249, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4257, 4261, 4263, 4264, 4265, 4266, 4267, 4268, 4269, 4270, 4271, 4273, 4274, 4275, 4278, 4279, 4281, 4282, 4283, 4284, 4285, 4286, 4287, 4289, 4290, 4291, 4292, 4294, 4296, 4297, 4298, 4299, 4301, 4302, 7025, 7037, 7038, 7039, 7040, 7041, 7043, 7044, 7045, 7047, 7048, 7049, 7051, 7052, 7053, 7056, 7058, 7059, 7061, 7062, 7063, 7064, 7065, 7066, 7067, 7068, 7069, 7070, 7071, 7072, 7073, 7074, 7076, 7077, 7078, 7079, 7080, 7081, 7082, 7084, 7086, 7087, 7089, 7090, 7092, 7093, 7096, 7097, 7098, 7099, 7100, 7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109, 7110, 7111, 7112, 7113, 7114, 7115, 7116, 7117, 7118, 7119, 7120, 7121, 7122, 7123, 7124, 7125, 7126, 7127, 7128, 7129, 7130, 7131, 7132, 7133, 7134, 7136, 7137, 7138, 7139, 7140, 7141, 7142, 7143, 7144, 7145, 7146, 7147, 7148, 7149, 7150, 7152, 7153, 7154, 7155, 7156, 7157, 7158, 7159, 7160, 7161, 7162, 7163, 7164, 7165, 7166, 7167, 7168, 7169, 7170, 7171, 7172, 7173, 7174, 7175, 7176, 7177, 7178, 7179, 7180, 7181, 7182, 7183, 7184, 7185, 7186, 7187, 7188, 7189, 7190, 7191, 7192, 7193, 7194, 7195, 7196, 7197, 7198, 7199, 7200, 7201, 7202, 7203, 7204, 7206, 7207, 7208, 7209, 7210, 7211, 7212, 7213, 7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7224, 7225, 7226, 7227, 7228, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7236, 7237, 7238, 7239, 7240, 7241, 7242, 7243, 7244, 7245, 7246, 7247, 7248, 7249, 7250, 7251, 7252, 7253, 7254, 7255, 7256, 7257, 7258, 7259, 7260, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 7268, 7269, 7270, 7271, 7272, 7273, 7274, 7275, 7276, 7277, 7278, 7279, 7280, 7281, 7282, 7283, 7284, 7285, 7286, 7287, 7288, 7290, 7291, 7292, 7293, 7294, 7295, 7296, 7297, 7298, 7299, 7300, 7301, 7302, 7303, 7304, 7305, 7306, 7307, 7308, 7309, 7310, 7311, 7312, 7313, 7314, 7315, 7316, 7317, 7318, 7319, 7320, 7322, 7323, 7325, 7326, 7327, 7328, 7329, 7330, 7332, 7333, 7334, 7335, 7336, 7337, 7338, 7339, 7340, 7341, 7343, 7344, 7345, 7346, 7347, 7348, 7349, 7350, 7351, 7352, 7354, 7355, 7356, 7357, 7358, 7359, 7360, 7361, 7362, 7363, 7364, 7365, 7366, 7367, 7368, 7369, 7371, 7372, 7373, 7374, 7375, 7376, 7377, 7378, 7379, 7381, 7382, 7383, 7384, 7385, 7386, 7387, 7388, 7389, 7390, 7391, 7392, 7393, 7394, 7395, 7396, 7397, 7398, 7399, 7401, 7402, 7403, 7404, 7405, 7406, 7407, 7408, 7409, 7411]

#--------------------------------------EKGS--------------------------------------#
def get_ekgs(path): 
    df = pd.read_csv(path + '/cardiac_ecg/manifest.tsv', delimiter='\t')
    cols = ['manufacturer', 'device_model', 'modality', 'header_version', 'dataset_information', 'dataset_usage_and_license', 'machine_text', 'machine_detail_description', 'interpretation_criteriaversion', 'patient_criteriaversion', 'internalmeasurements_version', 'Time_axis', 'Amplitude', 'device_documentation_type_and_version', 'participant_id', 'participant_position', 'Rate', 'PR', 'QRSD', 'QT', 'QTc', 'P', 'QRS', 'T', 'high_pass_filter_setting', 'low_pass_filter_setting', 'notch_filter_setting', 'notch_harmonic_setting', 'artifact_filter_flag', 'hysteresis_filter_flag', 'notch_filtered', 'ac_setting', 'report_description', 'interpretation_comment_1', 'interpretation_comment_2', 'validation_id', 'validation_date']
    for i in range(1,9):
        cols.extend(['comment_{}_key'.format(i), 'comment_{}_val'.format(i)])
    data = {col: [] for col in cols}
    #parse 
    for i in range(len(df)):
        _, header = wfdb.rdsamp(path+df.loc[i]['wfdb_hea_filepath'][:-4])
        for x in header['comments']:
            temp = x.split(':')
            if len(temp) == 2:
                data[temp[0]].append(temp[1].strip())
            else:
                data[temp[0]].append(np.nan)
        for j in range(1,9):
            if len(data['comment_{}_key'.format(j)]) != i+1:
                data['comment_{}_key'.format(j)].append(np.nan)

            if len(data['comment_{}_val'.format(j)]) != i+1:
                data['comment_{}_val'.format(j)].append(np.nan)
    #convert to df and drop uninformative columns 
    data = pd.DataFrame(data)
    data = data[[col for col in data.columns if len(data[col].value_counts()) != 1 or 'comment' in col]]
    #add binary columns for normal/abnormal
    data = pd.concat([data, pd.get_dummies(data['interpretation_comment_2'].apply(lambda x: x[2:-6])).astype(int)], axis=1).drop(columns=['interpretation_comment_2'])
    #Make numeric 
    data['participant_position'] = data['participant_position'].apply(lambda x: int(x.split(' ')[0]))
    #select relevant columns 
    data = data[['participant_id', 'participant_position', 'Rate', 'PR', 'QRSD', 'QT', 'QTc', 'P', 'QRS', 'T', 'ABNORMAL', 'BORDERLINE', 'NORMAL', 'OTHERWISE NORMAL']]
    data = data.rename(columns={"participant_id": "patient_id"})
    os.makedirs("ekgs", exist_ok=True)
    data.to_csv('ekgs/ekg_cleaned.csv')

#helper to resolve duplicate EKGS
def pick_best(g, prefer='normal', prefer_position=(0,30,60)):
    num_cols = ['Rate','PR','QRSD','QT','QTc']
    angle_cols = ['P','QRS','T']
    flag_cols  = ['ABNORMAL','BORDERLINE','NORMAL','OTHERWISE NORMAL']
    g = g.copy()
    #drop obviously broken ECGs
    g = g[~((g['QTc'] == 0) | (g['QTc'].isna() & g['QT'].isna()))]
    #prefer normal 
    sev_weights = {'ABNORMAL':3,'BORDERLINE':2,'NORMAL':1,'OTHERWISE NORMAL':0}
    g['sev_score'] = g[flag_cols].mul([sev_weights[c] for c in flag_cols]).max(axis=1)
    if prefer == 'normal':  # flip so lower is better
        g['sev_score'] = -g['sev_score']
    #prefer 0 over 30 over 60 
    pos_order = {p:i for i,p in enumerate(prefer_position)}
    g['pos_score'] = g['participant_position'].map(pos_order).fillna(len(pos_order))
    #completeness
    g['nonnull'] = g[num_cols + angle_cols].notna().sum(axis=1)
    #final tie breaker QtC
    g = g.sort_values(['pos_score','sev_score','nonnull','QTc'], ascending=[True, True, False, False])
    return g.iloc[0]
#--------------------------------------CONDITIONS--------------------------------------#
def extract_condition_name(condition_str):
    if ',' in condition_str:
        return condition_str.split(',')[1].strip()
    else:
        match = re.search(r'mhoccur_(\w+)|mh_(\w+)', condition_str)
        if match:
            return match.group(1) or match.group(2)
        return condition_str

def get_conditions(path): 
    csv_path = Path(path) 
    df = pd.read_csv(csv_path)
    #focus only on patients we also have wearable data for and on relevant columns from csv file 
    df = df[df['person_id'].isin(PATIENT_IDS)]
    col_to_keep = ["condition_occurrence_id", "person_id", "condition_concept_id", "condition_type_concept_id", "condition_status_concept_id", "condition_source_value"]
    df = df[col_to_keep].sort_values(by=["person_id", "condition_concept_id"]).reset_index(drop=True)
    #clean column names 
    conditions = df["condition_source_value"].unique()
    df['condition'] = df['condition_source_value'].apply(extract_condition_name)
    df['condition'] = df['condition'].replace({ 
        'Age-related macular degeneration (AM': "Age-related macular degeneration", 
        'Arthritis (joint pain)': 'Arthritis', 
        'Cancer (any type)': 'Cancer',
        'Cataracts (in one or both eyes)': 'Cataracts (1+ eyes)',
        'Chronic pulmonary (lung) problems (E': 'Chronic pullmonary problems',
        'Circulation problems (Examples: art': 'Circulation problems',
        "Dementia (Examples: Alzheimer's Disea": 'Dementia/Alzheimers',
        'Diabetic retinopathy (in one or both': 'Diabetic retinopathy (1+)',
        'Digestive problems (Examples: stomach': 'Digestive problems', 
        'Do you use marijuana now?': 'Marijuana user',
        'Dry eye (in one or both eyes)': 'Dry eye (1+)',
        'Elevated A1C levels (elevated blood sugar': 'Elevated A1C',
        'Glaucoma (in one or both eyes)': 'Glaucoma (1+)', 
        'Mild cognitive impairment (known as': 'Mild cognitive impairmen',
        'Other heart issues (Examples: pace': 'Other heart issues (pacemaker)',
        'Retinal vascular occlusion ("stroke': 'Retinal vascular occlusion', 
        'Urinary problems (Examples: urinary t': 'Urinary problems'})
    condition_matrix = df.pivot_table(index='person_id', columns='condition', values='condition_occurrence_id', aggfunc='count', fill_value=0)
    #convert to binary condition 
    condition_matrix = (condition_matrix > 0).astype(int)
    condition_matrix = condition_matrix.reset_index().rename(columns={'person_id': 'patient_id'})
    condition_matrix = condition_matrix.fillna(0).astype(int)
    #save 
    csv_path = Path("conditions/conditions.csv")
    condition_matrix.to_csv(csv_path)
    txt_path = csv_path.with_suffix(".txt")
    txt_path.write_text(condition_matrix.to_string(index=False, col_space=10, justify="right", float_format="%.3f", na_rep="NaN"))
    return condition_matrix 

#--------------------------------------MEASUREMENTS--------------------------------------#
def get_measurements(path): 
    csv_path = Path(path) 
    df = pd.read_csv(csv_path)
    #focus only on patients we also have wearable data for and on relevant columns from csv file 
    df = df[df['person_id'].isin(PATIENT_IDS)]
    col_to_keep = ["person_id", "measurement_date", "measurement_source_value", "unit_source_value", "value_source_value"]
    df = df[col_to_keep].sort_values(by=["person_id", "measurement_source_value"]).reset_index(drop=True)
    df = df.rename(columns={'person_id': 'patient_id'})
    #clean 
    df['value_source_value'] = pd.to_numeric(df['value_source_value'], errors='coerce')
    df['measurement_column'] = df['measurement_source_value'] + '_value'
    df.loc[df['unit_source_value'].notna(), 'measurement_column'] = (df.loc[df['unit_source_value'].notna(), 'measurement_source_value'] + '_value (' + df.loc[df['unit_source_value'].notna(), 'unit_source_value'] + ')')
    #take first if duplicates 
    pivoted_df = df.pivot_table(index=['patient_id', 'measurement_date'],  columns='measurement_column', values='value_source_value', aggfunc='first').reset_index()
    pivoted_df.columns.name = None
    print("Available measurements:")
    print(pivoted_df.columns)
    print([col for col in pivoted_df.columns if col not in ['patient_id', 'measurement_date']])
    print(f"\nShape: {pivoted_df.shape}")
    print(pivoted_df.head())
    #save 
    csv_path = Path("measurements/measurements.csv")
    pivoted_df.to_csv(csv_path, index=False)
    txt_path = csv_path.with_suffix(".txt")
    txt_path.write_text(pivoted_df.to_string(index=False, col_space=10, justify="right", float_format="%.3f", na_rep="NaN"))
    return pivoted_df


def main(): 
    parser = argparse.ArgumentParser()
    #select dataset (small or large AIREADI version)
    #absolute location to AI-READI data folder as downloaded 
    parser.add_argument("--AIREADI_ROOT", type=str, required=True)
    #location to preprocessed wearable data from previous processing step: ex: "scalar_data/all_patients_8_24.0h_3.0h_False_True.csv
    parser.add_argument("--PREPROCESSED_WEARABLE_CSV", type=str, required=True)
    args = parser.parse_args()
    ai_readi_path = args.AIREADI_ROOT
    preprocessed_wearable_csv = args.PREPROCESSED_WEARABLE_CSV
    
    get_ekgs(ai_readi_path)
    get_conditions(f"{ai_readi_path}/clinical_data/condition_occurrence.csv")
    get_measurements(f"{ai_readi_path}/clinical_data/measurement.csv")

    #load wearable (contains blood glucose too), conditions, measurements and ekg 
    conditions = pd.read_csv("conditions/conditions.csv")
    measurements = pd.read_csv("measurements/measurements.csv")
    scalar_wearable = pd.read_csv(preprocessed_wearable_csv)
    ekg = pd.read_csv("ekgs/ekg_cleaned.csv")

    #match patients 
    scalar_wearable = scalar_wearable[scalar_wearable['patient_id'].isin(PATIENT_IDS)]
    patients = list(scalar_wearable['patient_id'])
    conditions = conditions[conditions['patient_id'].isin(patients)]
    measurements = measurements[measurements['patient_id'].isin(patients)]
    ekg = ekg[ekg['patient_id'].isin(patients)]
    
    #collapse duplicate measurement rows 
    measurements = measurements.drop(columns=['measurement_date'])
    key = ['patient_id']
    df = measurements
    key_cols = [key] if isinstance(key, str) else list(key)
    dupes = df[df.duplicated(key_cols, keep=False)].copy()
    value_cols = [c for c in df.columns if c not in key_cols]
    dupes  = df[df.duplicated(key_cols, keep=False)].copy()
    unique = df[~df.duplicated(key_cols, keep=False)].copy()
    #collapse each duplicate group by coalescing non-nulls column-wise
    collapsed = (dupes.groupby(key_cols)[value_cols].apply(lambda g: g.bfill().ffill().iloc[0]).reset_index())
    df_merged = pd.concat([unique, collapsed], ignore_index=True)
    assert not df_merged.duplicated(key_cols, keep=False).any()
    print(f"Before: {len(df)} rows  | After: {len(df_merged)} rows")
    measurements = df_merged 

    #remove duplicates for ekg: prefer the Normal score, drop any rows with Nan measurements, prefer position 0 
    ekg = (ekg.groupby('patient_id', group_keys=False).apply(pick_best).reset_index(drop=True))
    ekg = ekg.drop(columns=['sev_score', 'pos_score', 'nonnull'])

    #merge wearables with conditions, measurements and ekgs 
    combined_all = scalar_wearable.merge(conditions, on='patient_id', how='left')
    combined_all = combined_all.merge(measurements, on='patient_id', how='left')
    combined_all = combined_all.merge(ekg, on='patient_id', how='left')
    combined_all = combined_all.drop(columns=["Unnamed: 0_x", "Unnamed: 0", "Unnamed: 0_y"])
    combined_all.to_csv("ai_readi_full_data_new.csv")
    print(len(combined_all))
    print(list(combined_all.columns))
    print(len(list(combined_all.columns)))



if __name__ == "__main__":
    main()