# RLSyn


**Preprocessing AI-READI** 

To preprocess AI-READI, first download the data to a folder (ex: ../dataset). Then data preprocessing proceeds in 3 steps. 

*1_get_wearable_data.py*: Aligns wearable data in regular intervals and then gets averaged scalar values for each valid day. Output for next step gets saved in scalar_data/all_patients*.csv. 

    To run: python 1_get_wearable_data.py --DATA_PATH /absolute/path/to/dataset/folder

*2_aggregate_data_modalities.py*: Preprocesses other data modalities (ekgs, conditions, measurements) and aggregates with preprocessed wearable data. Output for next step gets saved to ai_readi_*_data.csv. 
    
    To run: python 2_aggregate_data_modalities.py  --AIREADI_ROOT /absolute/path/to/dataset/folder --PREPROCESSED_WEARABLE_CSV relative/path/to/output/of/last/step 

*3_preprocess_aireadi.py* preprocesses the AI-READI data. Specifically, it drops na rows, selects features and creates 10 training and testing splits (10 seeds) for training data sizes from 0.1 to 0.9 (ie, 10-90 training testing split... 90-10 training testing split). You can modify the variables at the top to change these seeds/data splits as needed. 

    To run: python 3_preprocess_aireadi.py 

Data is saved under /AI-READI-OG or /AI-READI-FULL depending on what you select Each subfolder is in the format /preprocessed_data{DATA_SIZE}_seed{SEED}. Inside each subfolder, you'll find: 
- *min_max_log.npy*: normalization stats based on training data 
- *original_data_with_patients.csv*: unnormalized data with patient ids (na rows still dropped)
- *preprocessed_data_with_patients.csv*: normalized data with patient ids 
- *preprocessed_data_no_patients.csv*: normalized data without patient ids 
- *original_training_data.csv*: unnormalized training data (no patient ids)
- *original_testing_data.csv*: unnormalized testing data (no patient ids)
- *normalized_training_data.csv*: normalized training data (no patient ids)
- *normalized_testing_data.csv*: normalized testing data (no patient ids)


**Preprocessing MIMIC** 

We follow the preprocessing steps from (Yan et al., 2024), available [here](https://github.com/yanchao0222/tutorial_data_synthesis_and_evaluation). The preprocessing code can be run from the Jupyter notebook included. 

**EMR_WGAN** 

The EMR_WGAN code is located inside /EMR_WGAN/. Our code is functionally identical to the original code from (Yan et al., 2024). 
- *models.py*: Generator and Discriminator models
- *hyperparams.py*: provides appropriate hyperparameters for AIREADI/MIMIC. Note that MIMIC follows (Yan et al., 2024). 
- *train.py*: the core training loop. automatically evaluates after runs using *evaluation/evaluate_aireadi.py*. 
- *optuna_search_aireadi_full.py*: optuna search code for AIREADI. 

    To run: python -m EMR_WGAN.train --OUT_DIR "test" --RESULT_CSV "results.csv" --RUN_NAME "hi" --DATASET [AIREADI or MIMIC] --SEED 0 
 
*Checkpointing Logic*: The code automatically checks the best checkpoint (lowest loss), assuming that at least 1/3 of the total epochs have passed (to avoid getting stuck on early losses that might be a result of the discriminator not beign strong enough yet), AND the last checkpoint (last epoch of training). Training checkpoints get saved under the model name (EMR_{RUN_NAME}) under EMR_WGAN_training_checkpoints. 


**EHR_DIFF** 

We include the original code from [EHR_DIFF](https://github.com/sczzz3/EHRDiff). Our optuna search files can be found under optuna_search_aireadi.py and optuna_search_mimic.py, which run an Optuna search over the following parameters: time dimension (time_dim), learning rate (lr), number of epochs (n_epochs) and batch size (batch_size). To run either of these, use 
    
    python EHR_DIFF/optuna_search_aireadi.py --DATA_PATH /YOUR/PATH/TO/AIREADI/DATA --STUDY_NAME /YOUR/SAVE/DIR. 

The best trials from these searches are used in our code, with the configs updated in configs/our_aireadi and configs/our_mimic. To rerun our code, use main_aireadi.py and main_mimic.py. 
main_aireadi.py and main_mimic.py. To run, 
    
    python main_mimic.py --SEED SEED --DATA_PATH /PATH/TO/DATA --NPY_PATH /PATH/TO/DATA/min_max_log.npy --BASE_DIR /YOUR/SAVE/DIR --RUN_NAME /YOUR/RUN/NAME


**RL** 

The RLSyn code is located inside /RL/. 

- *models.py*: Generator (Policy) and Discriminator models
- *hyperparams.py*: provides appropriate hyperparameters for AIREADI and MIMIC. 
- *train.py*: the core training loop. automatically evaluates after runs (using *evaluation/evaluate_aireadi.py*)
- *optuna_search_aireadi_full.py: optuna search code for aireadi. 
- *optuna_search_mimic.py: optuna search code for mimic. 

    To run: python -m RL.train --OUT_DIR "test" --RESULT_CSV "results.csv" --RUN_NAME "hi" --DATASET [AIREADI or MIMIC] --DATA_PATH /path/to/data --SEED 0 

You can add flags as desired to override other hyperparameters (in parse_args())



**Evaluation** 

- *evaluate_aireadi.py*: evaluation methods for AI-READI, automatically called from train.py for all models. 
    - Utility evaluation: Type 2 Diabetes classifications.
    - Fidelity evaluation: correlations, value statistics, histograms, distributional score (from value statistics), PCA 
    - Privacy evaluation: membership inference risk 
    - Plotting losses (RL and EMR_WGAN specific)
- *evaluate_mimic.py*: evaluation methods for MIMIC, automatically called from train.py for all models. 
    - Utility evaluation: 'DIE_1y' target variable classification. We use a LightGB (following Yan et al., 2024). 
    - Fidelity evaluation: correlations, value statistics, histograms, dimension wide distribution, latent cluster analysis, medical concept abundance, clinical knowledge violations. 
    - Privacy evaluation: membership inference risk 
    - Plotting losses (RL and EMR_WGAN specific)


**Results**

Inside *results*, you'll find our results for each model and dataset. 


