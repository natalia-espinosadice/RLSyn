import torch, torch.nn as nn, torch.nn.functional as F
import os 
import optuna
from optuna.visualization import (plot_optimization_history, plot_parallel_coordinate, plot_pareto_front, plot_param_importances)
from evaluation.evaluate_mimic import (get_column_wise_correlationsM, compute_dimension_wide_distribution, latent_cluster_analysisM, run_PCA, medical_concept_abundance, clinical_knowledge_violation, train_and_test_classification, log_result_RL_MIMIC, mem_risk_MIMIC)
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os 
import time 
from pathlib import Path
import numpy as np 
import sys, subprocess, csv, yaml
from pathlib import Path
import random
import argparse 

CAT_COLS = ['WHITE', 'BLACK', 'ASIAN', 'HISPANIC', 'UN', 'OTHER', 'DIE_1y', 'GENDER', '008', '010', '031', '038', '041', '053', '054', '070', '070.4', '070.9', '071', '078', '079', '080', '081', '090', '10', '1000', '1001', '1002', '1004', '1005', '1006', '1007', '1008', '1009', '1010', '1011', '1013', '1014', '1015', '1019', '110', '110.1', '110.11', 
              '110.12', '110.13', '110.2', '1100', '112', '112.3', '117', '117.4', '130', '130.1', '131', '132', '132.1', '133', '134', '134.1', '136', '145', '145.2', '145.3', '149', '149.1', '149.2', '149.3', '149.4', '149.9', '150', '151', '153', '153.2', '153.3', '155', '155.1', '157', '158', '159', '159.2', '159.3', '159.4', '164', '165', '165.1',
            '170', '170.1', '170.2', '172', '172.11', '172.2', '172.3', '174', '174.1', '174.11', '175', '180', '180.1', '180.3', '182', '184', '184.1', '184.2', '185', '187', '187.2', '189', '189.2', '189.21', '189.4', '190', '191', '191.1', '191.11', '193', '194', '195', '195.1', '195.3', '196', '197', '198', '198.1', '198.2', '198.3', '198.4', '198.5', '198.6', 
            '198.7', '199', '199.4', '200', '200.1', '201', '202', '202.2', '202.21', '202.24', '204', '204.1', '204.11', '204.12', '204.2', '204.21', '204.22', '204.3', '204.4', '208', '209', '210', '211', '212', '213', '214', '214.1', '215', '216', '217', '218', '218.1', '220', '221', '222', '223', '225', '225.1', '225.2', '226', '227', '227.1', '227.2', '227.3', 
            '228', '229', '230', '240', '241', '241.1', '241.2', '242', '242.1', '242.2', '242.3', '244', '244.1', '244.2', '244.4', '244.5', '245', '245.21', '246', '246.7', '249', '250', '250.1', '250.11', '250.12', '250.13', '250.14', '250.2', '250.21', '250.22', '250.23', '250.24', '250.3', '250.4', '250.41', '250.42', '250.5', '251', '251.1', '251.8', '252', '252.1', 
            '252.2', '253', '253.1', '253.11', '253.2', '253.3', '253.7', '254', '255', '255.11', '255.12', '255.21', '256', '256.4', '257', '257.1', '258', '259', '259.2', '260', '260.1', '260.6', '260.7', '261', '261.1', '261.2', '261.4', '261.41', '262', '263', '264', '269', '270', '270.1', '270.11', '270.2', '270.21', '270.31', '270.32', '270.33', '270.35', 
              '271', '271.3', '271.9', '272', '272.1', '272.11', '272.12', '272.13', '274', '274.1', '274.11', '274.2', '274.21', '275', '275.1', '275.2', '275.3', '275.5', '275.53', '276', '276.1', '276.11', '276.12', '276.13', '276.14', '276.4', '276.41', '276.42', '276.6', '276.8', '277', '277.1', '277.4', '277.5', '277.51', '277.6', '277.7', '278', '278.1', '278.3', '278.4', 
              '279', '279.1', '279.11', '279.7', '279.8', '280', '280.1', '280.2', '281', '281.1', '281.11', '281.12', '281.13', '281.9', '282', '282.5', '282.8', '282.9', '283', '283.1', '283.2', '283.21', '284', '285', '285.1', '285.2', '285.22', '285.8', '286', '286.1', '286.11', '286.12', '286.13', '286.2', '286.3', '286.4', '286.5', '286.6', '286.7', '287', '287.1', '287.2', 
              '287.3', '287.31', '287.32', '287.4', '288', '288.1', '288.3', '289', '289.4', '289.5', '289.8', '289.9', '290', '290.1', '290.11', '290.16', '290.3', '291', '291.1', '291.4', '291.8', '292', '292.1', '292.12', '292.3', '292.4', '292.6', '293', '293.1', '295', '295.1', '295.2', '295.3', '296', '296.1', '296.2', '296.22', '297', '300', '300.1', '300.11', '300.12', '300.13', 
              '300.3', '300.4', '300.8', '300.9', '301', '301.1', '301.2', '302', '303', '303.1', '303.3', '303.4', '304', '305.2', '305.21', '306', '306.1', '306.9', '31', '312', '312.3', '313', '313.1', '313.2', '313.3', '315', '315.1', '315.2', '315.3', '316', '316.1', '317', '317.1', '317.11', '318', '320', '323', '323.2', '323.8', '324', '325', '327', '327.3', '327.4', '327.41', '327.5', 
              '327.6', '331', '331.1', '331.9', '332', '333', '333.1', '333.2', '333.3', '333.4', '333.8', '334', '334.1', '334.2', '335', '337', '337.1', '338', '339', '340', '340.1', '341', '342', '343', '344', '345', '345.1', '345.11', '345.12', '345.3', '346', '346.1', '346.2', '346.3', '347', '348', '348.2', '348.4', '348.7', '348.8', '348.9', '349', '350', '350.1', '350.2', '350.3', '350.5', '350.6', '351', '352', '352.1', '352.2', '353', '353.1', '353.2', '355', '355.1', '356', '357', '358', '358.1', '359', '359.1', '359.2', '360', '361', '362', '362.29', '362.4', '362.7', '362.8', '363', '364', '364.5', '364.9', '365', '365.11', '365.2', '366', '366.2', '367', '367.1', '367.4', '367.9', '368', '368.1', '368.2', '368.4', '368.5', '368.9', '369', '369.2', '369.5', '370', '370.1', '371', '371.1', '371.3', '372', '374', '374.1', '374.3', '375', '375.1', '376', '377', '377.3', '378', '378.1', '378.2', '378.5', '379', '379.1', '379.2', '379.4', '379.9', '38', '38.1', '38.2', '380', '380.1', '380.4', '381', '381.11', '381.2', '381.3', '382', '383', '384', '384.4', '385', '386', '386.1', '386.2', '386.21', '386.3', '386.9', '388', '389', '389.1', '389.2', '389.3', '389.4', '389.5', '394', '394.2', '394.3', '394.4', '394.7', '395', '395.1', '395.3', '395.4', '395.6', '396', '401', '401.1', '401.2', '401.21', '401.22', '401.3', '402', '41', '41.1', '41.2', '41.4', '411', '411.1', '411.2', '411.3', '411.4', '411.41', '411.8', '411.9', '414', '415', '415.11', '415.2', '415.21', '416', '418', '418.1', '420', '420.1', '420.2', '420.21', '420.22', '420.3', '425', '425.1', '425.11', '425.12', '425.2', '425.8', '426', '426.2', '426.21', '426.23', '426.24', '426.25', '426.3', '426.31', '426.32', '426.4', '426.8', '426.9', '426.91', '427', '427.1', '427.11', '427.12', '427.2', '427.3', '427.41', '427.42', '427.5', '427.6', '427.61', '427.7', '427.8', '427.9', '428', '428.2', '429', '429.1', '429.2', 
              '429.3', '430', '430.1', '430.2', '430.3', '433', '433.1', '433.11', '433.12', '433.2', '433.21', '433.3', '433.31', '433.32', '433.5', '433.8', '440', '440.1', '440.2', '440.9', '441', '441.1', '441.2', '442', '442.1', '442.11', '442.2', '442.3', '442.8', '443', '443.1', '443.7', '443.8', '443.9', '444', '444.1', '444.2', '446', '446.3', '446.4', '446.5', '446.6', '446.7', '446.8', '446.9', '447', '447.1', '448', '450', '451', '451.2', '452', '452.1', '452.8', '454', '454.1', '454.11', '455', '456', '457', '458', '458.1', '458.2', '458.9', '459', '459.1', '459.9', '464', '465', '465.2', '465.4', '470', '471', '472', '473', '473.3', '473.4', '474', '474.1', '474.2', '475', '476', '477', '478', '479', '480', '480.1', '480.11', '480.12', '480.2', '480.3', '480.5', '481', '483', '495', '496', '496.1', '496.2', '496.21', '497', '498', '499', '500', '500.1', '500.2', '501', '502', '503', '504', '505', '506', '507', '508', '509', '509.1', '509.2', '509.3', '509.8', '510', '510.2', '512', '512.1', '512.2', '512.7', '512.8', '512.9', '513', '513.3', '513.4', '513.8', '514', '514.1', '514.2', '516', '516.1', '519', '519.1', '519.2', '519.8', '519.9', '520', '520.2', '521', '521.1', '522', '522.1', '522.5', '523', '523.1', '523.31', '523.32', '524', '525', '526', '527', '527.2', '527.7', '527.8', '528', '528.11', '528.12', '528.3', '528.5', '528.6', '528.7', '529', '529.1', '529.6', '53', '53.1', '530', '530.11', '530.12', '530.14', '530.2', '530.3', '530.5', '530.6', '530.7', '530.9', '531', '531.1', '531.2', '531.3', '531.4', '531.5', '532', '535', '535.1', '535.2', '535.6', '535.8', '536', '537', '539', '54', '540', '540.1', '540.11', '550', '550.1', '550.2', '550.3', '550.4', '550.5', '555', '555.1', '555.2', '555.21', '556', '556.1', '557', '557.1', '558', '559', '560', '560.1', '560.2', '560.3', '560.4', '561', '561.1', '562', '562.1', '563', '564', '564.1', '564.8', '564.9', '565', '565.1', '567', '568', '568.1', '569', '569.1', '569.2', '571', '571.5', '571.51', '571.6', '571.8', '571.81', '572', '573', '573.1', '573.2', '573.3', '573.4', '573.5', '573.6', '573.7', '573.9', '574', '574.1', '574.11', '574.12', '574.2', '574.3', '575', '575.1', '575.2', '575.6', '575.7', '575.8', '575.9', '577', '577.1', '577.2', '577.3', '578', '578.1', '578.2', '578.8', '578.9', '579', '579.2', '579.8', '580', '580.11', '580.12', '580.31', '580.32', '585', '585.1', '585.2', '585.3', '585.31', '586', '586.12', '586.2', '586.3', '586.4', '587', '588', '588.1', '589', '590', '591', '592', '592.1', '592.11', '592.12', '592.13', '592.2', '593', '593.2', '594', '594.1', '594.2', '594.3', '594.8', '595', '596', '596.1', '596.5', '597', '597.1', '597.2', '598', '598.9', '599', '599.1', '599.2', '599.3', '599.4', '599.6', '599.8', '599.9', '600', '601', '601.1', '601.11', '601.12', '601.4', '601.8', '602', '603', '603.1', '604', '604.1', '605', '608', '610', '610.1', '610.8', '611', '611.3', '612', '612.2', '613', '613.1', '613.5', '613.7', '613.8', '614', '614.1', '614.3', '614.31', '614.32', '614.33', '614.4', '614.5', '614.52', '614.53', '614.54', '615', '617', '618', '618.1', '618.2', '618.5', '618.6', '619', '619.1', '619.2', '619.3', '619.4', '619.5', '620', '621', '622', '622.1', '622.2', '623', '624', '624.1', '624.2', '624.9', '625', '625.1', '626', '626.1', '626.11', '626.12', '626.13', '626.14', '626.2', '626.8', '627', '627.1', '627.2', '627.3', '627.4', '627.5', '628', '634', '634.1', '634.3', '635', '635.2', '635.3', '636', '636.2', '636.3', '636.8', '639', '642', '642.1', '643', '643.1', '644', '645', '646', '647', '647.1', '647.3', '649', '649.1', '651', '652', '653', '654', '654.1', '654.2', '655', '656', '661', '663', '665', '668', '669', '671', '674', '676', '681', '681.1', '681.2', '681.3', '681.7', '686', '686.1', '686.2', '686.3', '686.4', '686.5', '687', '687.1', '687.2', '687.3', '687.4', '689', '690', '690.1', '691', '694', '694.1', '694.2', '694.3', '695', '695.1', '695.22', '695.3', '695.41', '695.42', '695.7', '695.8', '695.81', '695.9', '696', '696.41', '696.42', '697', '698', '70', '70.1', '70.2', '70.3', '70.4', '70.9', '700', '701', '701.2', '701.4', '701.5', '701.6', '702', '702.1', '703', '703.1', '704', '704.1', '704.11', '704.2', '704.8', '705', '705.3', '705.8', '706', '706.1', '706.2', '706.8', '707', '707.1', '709', '709.2', '709.3', '709.4', '709.5', '709.6', '709.7', '71', '710', '710.11', '710.12', '710.19', '711', '711.1', '711.2', '711.3', '712', '713', '713.5', '714', '714.1', '714.2', '715', '715.1', '716', '716.1', '716.2', '716.9', '717', '720', '721', '721.1', '721.2', '721.8', '722', '722.1', '722.6', '722.7', '722.8', '722.9', '723', '723.1', '724', '724.8', '724.9', '726', '726.1', '726.2', '726.3', '726.4', '727', '727.1', '727.2', '727.4', '727.5', '727.7', '728', '728.1', '728.2', '728.7', '728.71', '729', '729.3', '729.7', '731', '731.1', '732', '732.1', '733', '733.2', '733.4', '733.6', '733.8', '733.9', '735', '735.1', '735.2', '735.21', '735.3', '736', '736.2', '737', '737.1', '737.3', '738', '738.4', '739', '740', '740.1', '740.11', '740.12', '740.9', '741', '741.1', '741.2', '741.3', '741.4', '741.5', '742', '742.1', '742.2', '742.9', '743', '743.12', '743.13', '743.2', '743.21', '743.9', '745', '747', '747.1', '747.11', '747.12', '747.13', '747.2', '748', '749', '749.2', '750', '750.11', '750.14', '750.21', '750.22', '751', '751.11', '751.12', '751.2', '751.21', '751.22', '751.3', '752', '752.11', '752.2', '753', '754', '755', '755.1', '755.4', '755.6', '755.61', '756', '756.1', '756.21', '756.3', '756.5', '757', '758', '758.1', '759', '759.1', '760', '761', '763', '764', '765', '766', '770', '771', '771.1', '771.2', '772', '772.1', '772.2', '772.3', '772.4', '772.6', '773', '78', '780', '781', '782', '782.3', '782.6', '783', '785', '788', '789', '79', '79.1', '79.2', '790', '790.1', '790.6', '790.8', '790.9', '791', '792', '793', '793.2', '794', '795', '795.8', '796', '797', '797.1', '798', '8', '8.5', '8.52', '8.6', '8.7', '80', '800', '800.1', '800.2', '800.3', '800.4', '801', '801.1', '802', '803', '803.1', '803.2', '803.3', '804', '805', '807', '809', '81', '816', '817', '818', '819', '830', '835', '836', '840', '841', '842', '850', '851', '853', '854', '855', '856', '857', '858', '859', '860', '870', '870.1', '870.2', '870.3', '870.4', '870.5', '870.6', '870.8', '871', '871.3', '871.4', '872', '873', '874', '875', '876', '90', '90.2', '90.3', '907', '910', '911', '912', '913', '915', '916', '930', '931', '938', '938.1', '939', '941', '942', '946', '947', '949', '952', '958', '958.1', '958.2', '960', '960.1', '960.2', '961', '961.1', '962', '962.3', '963', '963.1', '964', '964.1', '965', '965.1', '965.3', '966', '967', '969', '971', '972', '973', '974', '975', '976', '977', 
              '979', '980', '981', '983', '985', '987', '988', '989', '990', '994', '994.2']
NUM_COLS = ['AGE', 'BMI', 'DIASTOLIC', 'SYSTOLIC']

REPO_ROOT = Path(__file__).resolve().parent  
CONFIG_TRAIN  = REPO_ROOT / "configs/our_mimic/train_edm.yaml"
CONFIG_SAMPLE = REPO_ROOT / "configs/our_mimic/sample_edm.yaml"
#-----config help-------#
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def dump_config(cfg, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def override_cfg(cfg, overrides: dict):
    #by nested dict keys. Ex: "data": {"path": "/data/EHR"},
    def rec(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = rec(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    return rec(cfg, overrides)

def log_result(RESULTS_CSV, tag, seed, batch, time_dim, learning_rate, s2h_auc, s2h_prauc, s2h_acc, r2s_auc, r2s_prauc, r2s_acc, elapsed_time):
    p = Path(RESULTS_CSV)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"tag": tag, "seed" : seed,
        "batch": batch, "time_dim": time_dim, "lr": learning_rate, 
        "s2h_auc" : s2h_auc, "s2h_prauc": s2h_prauc, "s2h_acc" : s2h_acc,
        "r2s_auc" : r2s_auc, "r2s_prauc": r2s_prauc, "r2s_acc" : r2s_acc,
        "elapsed_time" : elapsed_time,
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def run_to_logs(cmd, cwd, trial_dir, name, env=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    (trial_dir / f"{name}.stdout.txt").write_text(p.stdout)
    (trial_dir / f"{name}.stderr.txt").write_text(p.stderr)
    return p

def evaluate_model_MIMIC(df_train, df_real, df_syn, df_train_norm, df_hold_norm, df_real_norm, df_syn_norm, elapsed_time, OUT_DIR, SEED, RESULT_CSV, RUN_NAME, ITERS, DATA_SIZE): 
    #-----------------FIDELITY-----------------#
    #column wise correlations 
    cwc = get_column_wise_correlationsM(df_real, df_syn, f"{OUT_DIR}/correlations", True) 
    ad2d, continuous_w_d = compute_dimension_wide_distribution(df_real, df_syn, f"{OUT_DIR}/dimension_wide_distributions")
    latent_cluster_analysis = latent_cluster_analysisM(df_real, df_syn, f"{OUT_DIR}/PCA")
    run_PCA(df_real, [df_syn], f"{OUT_DIR}/PCA", SEED)
    mca_dist, mca_tvd_dist = medical_concept_abundance(df_real, df_syn, CAT_COLS, f"{OUT_DIR}/medical_abundance")
    combined_clinical_violations = clinical_knowledge_violation(df_train, df_syn, CAT_COLS, f"{OUT_DIR}/clinical_knowledge_violation")
    #-----------------UTILITY-----------------#
    EXCLUDE_COLS = ['WHITE', 'BLACK', 'ASIAN', 'HISPANIC', 'UN', 'OTHER', 'DIE_1y']
    cat_cols = [c for c in CAT_COLS if c not in EXCLUDE_COLS]
    r2s_results = train_and_test_classification(df_real_norm, df_syn_norm, f"{OUT_DIR}/real_to_synthetic", SEED, cat_cols)
    r2s_auc = r2s_results["test"]["auroc"]
    r2s_prauc = r2s_results["test"]["prauc"]
    r2s_acc  = r2s_results["test"]["acc"]
    s2h_results = train_and_test_classification(df_syn_norm, df_hold_norm, f"{OUT_DIR}/synthetic_to_hold", SEED,  cat_cols)
    s2h_auc = s2h_results["test"]["auroc"]
    s2h_prauc = s2h_results["test"]["prauc"]
    s2h_acc  = s2h_results["test"]["acc"]
    r2r_results = train_and_test_classification(df_train_norm, df_hold_norm, f"{OUT_DIR}/real_to_real", SEED, cat_cols)
    r2r_auc = r2r_results["test"]["auroc"]
    r2r_prauc = r2r_results["test"]["prauc"]
    r2r_acc  = r2r_results["test"]["acc"]
    #-----PRIVAVY------#
    df_train_norm_bal = df_train_norm.sample(n=21000, random_state=SEED)[NUM_COLS + CAT_COLS]
    df_hold_norm_bal = df_hold_norm.sample(n=9000, random_state=SEED)[NUM_COLS + CAT_COLS]
    mem_aucs = mem_risk_MIMIC(df_train_norm_bal, df_hold_norm_bal, df_syn_norm, CAT_COLS, NUM_COLS, f"{OUT_DIR}/mem_risk", SEED)
    #-----------------LOG RESULTS-----------------#
    with open(f"{OUT_DIR}/eval.txt", "a") as f:
        f.write(f"CWC: {cwc}\nr2s_auc: {r2s_auc}\nr2s_prauc: {r2s_prauc}\nr2s_acc: {r2s_acc}\ns2h_auc: {s2h_auc}\ns2h_prauc: {s2h_prauc}\ns2h_acc: {s2h_acc}\n"
                f"r2r_auc: {r2r_auc}\nr2r_prauc: {r2r_prauc}\nr2r_acc: {r2r_acc}\nmem_auc_real: {mem_aucs['real_roc_auc']}\nmem_auc_synth: {mem_aucs['synth_roc_auc']}\n"
                f"ad2d: {ad2d}\ncontinuous_w_d: {continuous_w_d}\nlatent_cluster_analysis: {latent_cluster_analysis}\nmca_dist: {mca_dist}\nmca_tvd_dist: {mca_tvd_dist}\ncombined_clinical_violations: {combined_clinical_violations}\nelapsed_time: {elapsed_time}\n"
        )
    log_result_RL_MIMIC(RESULT_CSV, RUN_NAME, ITERS, DATA_SIZE, SEED, cwc, ad2d, continuous_w_d, latent_cluster_analysis, mca_dist, mca_tvd_dist, combined_clinical_violations, s2h_auc, s2h_prauc, s2h_acc, r2s_auc, r2s_prauc, r2s_acc, r2r_auc, r2r_prauc, r2r_acc, mem_aucs, elapsed_time)
    return s2h_auc, r2s_auc 

def train(data_path, seed, base_dir, npy_path, run_name): 
    trial_dir = base_dir / run_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    train_cfg_out = trial_dir / "train_edm.yaml"
    sample_cfg_out =  trial_dir / "sample_edm.yaml"
    ckpt_path = trial_dir / "checkpoints" / "final_checkpoint.pth"
    synth_path = trial_dir / "samples" / "all_x.npy"
    save_syn = trial_dir / "synthetic.csv"
    save_syn_rescaled = trial_dir / "synthetic_rescaled.csv"
    result_csv = trial_dir / "results.csv"
    train_data = f"{data_path}/normalized_training_data_{seed}.csv"
    train_array_npy = Path(data_path) / f"normalized_training_data_{seed}.array.npy"

    df = pd.read_csv(train_data)[NUM_COLS + CAT_COLS]
    print(len(df.columns))
    arr = df.values.astype(np.float32)
    print(arr.shape[0], arr.shape[1])
    np.save(train_array_npy, arr)
    raw_data = np.load(train_array_npy)
    print(raw_data.shape[1])
    
    #load override and save config file 
    train_cfg  = load_config(CONFIG_TRAIN)
    sample_cfg = load_config(CONFIG_SAMPLE)
    train_overrides = {"data": {"path": str(train_array_npy)}, "train": {"seed": seed}}
    sample_overrides = {"model": {"ckpt": str(ckpt_path)}, "test": {"seed": seed}}
    train_cfg  = override_cfg(train_cfg,  train_overrides)
    sample_cfg = override_cfg(sample_cfg, sample_overrides)
    dump_config(train_cfg,  train_cfg_out)
    dump_config(sample_cfg, sample_cfg_out)
    sample_npy_path = trial_dir / "samples" / "sample.npy"

    #train 
    start_time = time.time() 
    cmd = [sys.executable, "main.py", "--mode", "train",  "--workdir", str(trial_dir), "--config", str(train_cfg_out)]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH','')}"
    if seed !=5: 
        proc  = run_to_logs(cmd,      cwd=REPO_ROOT, trial_dir=trial_dir, name="train", env=env)
        elapsed_time = (time.time() - start_time) / 60 
    else: 
        elapsed_time = np.nan 
    #sample 
    cmd_eval = [sys.executable, "main.py", "--mode", "eval", "--workdir", str(trial_dir), "--config", str(sample_cfg_out)]
    proc2 = run_to_logs(cmd_eval, cwd=REPO_ROOT, trial_dir=trial_dir, name="eval",  env=env)

    #get synthetic 
    synthetic = np.load(synth_path, allow_pickle=True)
    df_syn = pd.DataFrame(synthetic, columns=NUM_COLS + CAT_COLS)
    df_syn.to_csv(save_syn, index=False)
    feature_range = np.load(npy_path, allow_pickle=True).item()
    for col in NUM_COLS:
        xmin, xmax = feature_range[col]
        df_syn[col] = (1.0 - df_syn[col]) * xmin + df_syn[col] * xmax
    df_syn.to_csv(save_syn_rescaled)

    #evaluation 
    df_train_norm = pd.read_csv(f"{data_path}/normalized_training_data_{seed}.csv")[NUM_COLS + CAT_COLS]
    df_hold_norm =  pd.read_csv(f"{data_path}/normalized_testing_data_{seed}.csv")[NUM_COLS + CAT_COLS]
    df_syn_norm = pd.read_csv(save_syn)[NUM_COLS + CAT_COLS]
    #unnormalize (save space by not saving)
    feature_range = np.load(npy_path, allow_pickle=True).item()
    for col in NUM_COLS:
        xmin, xmax = feature_range[col]
        df_train_norm[col] = (1.0 - df_train_norm[col]) * xmin + df_train_norm[col] * xmax
        df_hold_norm[col] = (1.0 - df_hold_norm[col]) * xmin + df_hold_norm[col] * xmax
    df_train = df_train_norm 
    df_hold = df_hold_norm 
    df_real = pd.concat([df_train, df_hold])[NUM_COLS + CAT_COLS]
    df_train_norm = pd.read_csv(f"{data_path}/normalized_training_data_{seed}.csv")[NUM_COLS + CAT_COLS]
    df_hold_norm =  pd.read_csv(f"{data_path}/normalized_testing_data_{seed}.csv")[NUM_COLS + CAT_COLS]
    df_real_norm = pd.concat([df_train_norm, df_hold_norm])[NUM_COLS + CAT_COLS]
    evaluate_model_MIMIC(df_train, df_real, df_syn, df_train_norm, df_hold_norm, df_real_norm, df_syn_norm, elapsed_time, trial_dir, seed, result_csv, f"ehr_seed{seed}", 5000, 0.7)          
    if sample_npy_path.exists():
        os.remove(sample_npy_path)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--SEED", type=str, required=True )
    p.add_argument("--DATA_PATH", type=str, required=True ) #"/PATH/TO/DATA/"
    p.add_argument("--NPY_PATH", type=str, required=True ) #"/PATH/TO/DATA/min_max_log.npy"
    p.add_argument("--BASE_DIR", type=str, required=True )
    p.add_argument("--RUN_NAME", type=str, required=True ) 
    return vars(p.parse_args())

def main(): 
    args = parse_args() 
    seed = args["SEED"]
    #MODIFY
    data_path = f"{args['DATA_PATH']}/seed{seed}"  
    npy_path = args['NPY_PATH']
    run_name = args['RUN_NAME']
    base_dir = (REPO_ROOT / f"{args['BASE_DIR']}")   
    train(data_path, seed, base_dir, npy_path, run_name)
   
if __name__ == '__main__':
    main() 

