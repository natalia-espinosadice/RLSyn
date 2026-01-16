import tensorflow as tf
import pandas as pd
import numpy as np
import time
import os, re
from pathlib import Path
import random
import argparse 
from evaluation.evaluate_mimic import evaluate_model_MIMIC #(make_logger, log_result_EMR, compute_value_stats, log_result_EMR_search, print_stats, latent_cluster_analysis, train_on_real_test_on_real, train_on_synth_test_on_hold, train_on_hold_test_on_synth, get_histograms, get_column_wise_correlations, train_on_real_test_on_synth, train_on_synth_test_on_real, classify_real_vs_syn, plot_emr_losses, mem_risk)
from evaluation.evaluate_aireadi import (make_logger, log_result_EMR, compute_value_stats, log_result_EMR_search, print_stats, latent_cluster_analysis, train_on_real_test_on_real, train_on_synth_test_on_hold, train_on_hold_test_on_synth, get_histograms, get_column_wise_correlations, train_on_real_test_on_synth, train_on_synth_test_on_real, classify_real_vs_syn, plot_emr_losses, mem_risk)
from EMR_WGAN.hyperparams import HyperParams_AIREADI, HyperParams_MIMIC
from EMR_WGAN.models import Generator, Discriminator 


def train(H, log):
    start_time = time.time() 
    os.makedirs(H.CHECKPOINT_DIRECTORY, exist_ok=True)
    data = np.array(pd.read_csv(f"{H.DATA_PATH}/normalized_training_data.csv").values).astype('float32')
    dataset_train = tf.data.Dataset.from_tensor_slices(data).shuffle(10000,seed=H.SEED, reshuffle_each_iteration=True).batch(H.BATCH, drop_remainder=True)
    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=H.G_LR) #1e-5
    discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=H.D_LR) #2e-5
    generator = Generator(H)
    discriminator = Discriminator(H)
    checkpoint = tf.train.Checkpoint(generator=generator)
    manager = tf.train.CheckpointManager(checkpoint, directory=H.CHECKPOINT_DIRECTORY, max_to_keep=50)

    @tf.function
    def d_step(real):
        z = tf.random.normal(shape=[H.BATCH, H.NOISE_DIM])
        epsilon = tf.random.uniform(shape=[H.BATCH, 1], minval=0., maxval=1.)
        with tf.GradientTape() as disc_tape:
            synthetic = generator(z, training = False) #adding training = 
            interpolate = real + epsilon * (synthetic - real)
            real_output = discriminator(real)
            fake_output = discriminator(synthetic)
            w_distance = (-tf.reduce_mean(real_output) + tf.reduce_mean(fake_output))
            with tf.GradientTape() as t:
                t.watch(interpolate)
                interpolate_output = discriminator(interpolate)
            w_grad = t.gradient(interpolate_output, interpolate)
            slopes = tf.sqrt(tf.reduce_sum(tf.square(w_grad), 1))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            disc_loss = H.GRADIENT_PENALTY * gradient_penalty + w_distance
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        return disc_loss, w_distance

    @tf.function
    def g_step():
        z = tf.random.normal(shape=[H.BATCH, H.NOISE_DIM])
        with tf.GradientTape() as gen_tape:
            synthetic = generator(z,training = True) #added training = 
            fake_output = discriminator(synthetic)
            gen_loss = -tf.reduce_mean(fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    @tf.function
    def train_step(batch):
        #note: could pull a fresh batch of data in?? 
        for _ in tf.range(H.DISC_STEPS):
            disc_loss, w_distance = d_step(batch)
        g_step()
        return disc_loss, w_distance

    print('training start', flush=True)
    best_loss = 1000000.0
    best_epoch = 0 
    for epoch in range(H.EPOCHS):
        start_time = time.time()
        total_loss = 0.0
        total_w = 0.0
        step = 0.0
        for args in dataset_train:
            loss, w = train_step(args)
            total_loss += loss
            total_w += w
            step += 1
        duration_epoch = time.time() - start_time
        format_str = 'epoch: %d, loss = %f, w = %f, (%.2f)'
        if epoch % 10 == 0:
            log(format_str % (epoch, -total_loss / step, -total_w / step, duration_epoch))
            if ((epoch > (H.EPOCHS/3)) and epoch % 50 == 0 and -total_loss / step <= best_loss and -total_loss / step > 0):
                best_loss = -total_loss / step
                best_epoch = epoch 
                manager.save(checkpoint_number=epoch)
                log('ckpt %d saved with loss %.6f' % (epoch, best_loss))
            elif(epoch == (H.EPOCHS - 10)): 
                best_loss = -total_loss / step 
                manager.save(checkpoint_number=epoch)
                log('final ckpt %d saved with loss %.6f' % (epoch, best_loss))
    elapsed_time = (time.time() - start_time) / 60 
    return elapsed_time, best_epoch 


def gen(H, ckpt):
    generator = Generator(H)
    checkpoint = tf.train.Checkpoint(generator=generator)
    manager = tf.train.CheckpointManager(checkpoint, directory=H.CHECKPOINT_DIRECTORY, max_to_keep=50)
    #we do this manually in train: save the best checkpoint and the last checkpoint! 
    ''' 
    if H.LOAD_CHECKPOINT_NUMBER == 'best':
        status = checkpoint.restore(manager.latest_checkpoint)
    else:
    '''
    checkpoint.restore(H.CHECKPOINT_DIRECTORY + '/ckpt-' + ckpt).expect_partial()
    @tf.function
    def g_step():
        z = tf.random.normal(shape=[100, H.NOISE_DIM])
        synthetic = generator.test(z)
        return synthetic
    if H.DATASET == "MIMIC": 
        max_attempts = 100000
        data_df = pd.read_csv(f"{H.DATA_PATH}/preprocessed_training_data.csv")[H.CAT_COLS+H.NUM_COLS]
        data = np.array(data_df.values).astype('float32')
        pos_target = int(int(np.sum(data[:,H.OUTCOME_DIMENSION] == 1)) / 6)
        neg_target = int(int(np.sum(data[:,H.OUTCOME_DIMENSION] == 0)) / 6)
    else:
        max_attempts = 100000
        data_df = pd.read_csv(f"{H.DATA_PATH}/preprocessed_data_no_patients.csv")[H.CAT_COLS+H.NUM_COLS]
        data = np.array(data_df.values).astype('float32')
        pos_target = int(np.sum(data[:,H.OUTCOME_DIMENSION] == 1)) 
        neg_target = int(np.sum(data[:,H.OUTCOME_DIMENSION] == 0))
    
    syn_pos = []
    syn_neg = []
    attempts = 0 
    #modified to be more efficient, same idea 
    while (len(syn_pos) < pos_target or len(syn_neg) < neg_target) and attempts < max_attempts:
        batch = g_step().numpy()
        mask_pos = batch[:, H.OUTCOME_DIMENSION] >= 0.5 
        syn_pos.extend(batch[mask_pos])
        if len(syn_neg) < neg_target:
            syn_neg.extend(batch[~mask_pos])
        attempts += 1
        if attempts %1000 == 0: 
            print(attempts, len(syn_pos))
    if attempts == max_attempts:
        print(f"[WARN] Reached {max_attempts} generator calls with only {len(syn_pos)}/{pos_target} positives and {len(syn_neg)}/{neg_target} negatives")
    
    os.makedirs(f"{H.OUT_DIR}/ckpt_{ckpt}", exist_ok=True)
    syn = np.array(syn_pos[:pos_target] +syn_neg[:neg_target])
    df_syn_norm = pd.DataFrame(syn, columns=H.CAT_COLS+H.NUM_COLS)
    df_syn_norm.to_csv(f"{H.OUT_DIR}/ckpt_{ckpt}/synthetic.csv")
    col_list = list(data_df.columns)
    continuous_col_name_list = [col_list[col_ind] for col_ind in H.CONTINUOUS_FEATURE_COL_IND]
    feature_range = np.load(H.NPY_PATH, allow_pickle=True).item()
    for col_name, i in zip(continuous_col_name_list, H.CONTINUOUS_FEATURE_COL_IND):
        xmin, xmax = feature_range[col_name][0], feature_range[col_name][1]
        syn[:, i] = (1 - syn[:, i])*xmin + syn[:,i]*xmax
    df_syn = pd.DataFrame(syn, columns=H.CAT_COLS+H.NUM_COLS)
    df_syn.to_csv(f"{H.OUT_DIR}/ckpt_{ckpt}/synthetic_rescaled.csv")
    if attempts == max_attempts: 
        return -1 
    return 0 


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def evaluate_model_AIREADI(df_real, df_hold, df_train, df_syn, df_hold_norm, df_train_norm, df_syn_norm, df_real_with_patients_norm, H, elapsed_time, checkpoint): 
    #plot losses
    plot_emr_losses(f"{H.OUT_DIR}/{H.MODEL_ID}_outs.txt", f"{H.OUT_DIR}")
    #-----------------FIDELITY-----------------#
    #examine value statistics per feature
    print_stats(f"{H.OUT_DIR}/ckpt_{checkpoint}/eval.txt", df_real, df_syn, df_hold, df_train, H.NUM_COLS) 
    #get single feature histograms
    get_histograms(df_real, df_syn, f"{H.OUT_DIR}/ckpt_{checkpoint}/histograms")
    #PCA analysis
    latent_cluster_analysis(df_real_with_patients_norm.drop(columns=['patient_id']), [df_syn_norm], f"{H.OUT_DIR}/ckpt_{checkpoint}/PCA", H.SEED)
    #column wise correlations 
    cwc = get_column_wise_correlations(df_real, df_syn, f"{H.OUT_DIR}/ckpt_{checkpoint}/correlations") 
    #distributional scores by value stats 
    overall_score_real_num, overall_score_hold_num, overall_score_real_cat, overall_score_hold_cat, overall_score_real, overall_score_hold = compute_value_stats(df_real, df_hold, df_syn, f"{H.OUT_DIR}/ckpt_{checkpoint}/value_stat_analysis.txt", H.NUM_COLS, H.CAT_COLS)  
    #-----------------UTILITY-----------------#
    #get hold out (keep same size no matter the train - test split)
    ten_percent = 490
    if len(df_hold_norm) > ten_percent: 
        hold_fraction = ten_percent / len(df_hold_norm)   
        df_hold_norm = df_hold_norm.sample(frac=hold_fraction, random_state=H.SEED).reset_index(drop=True)
    #synthetic to hold out 
    s2h_auc, s2h_acc = train_on_synth_test_on_hold(df_syn_norm,  df_hold_norm, f"{H.OUT_DIR}/ckpt_{checkpoint}/synth_to_hold", H.SEED)
    #real to synthetic 
    r2s_auc, r2s_acc = train_on_real_test_on_synth(df_real_with_patients_norm, df_syn_norm, f"{H.OUT_DIR}/ckpt_{checkpoint}/real_to_synth", H.SEED)
    #synthetic to real 
    s2r_auc, s2r_acc = train_on_synth_test_on_real(df_syn_norm, df_real_with_patients_norm, f"{H.OUT_DIR}/ckpt_{checkpoint}/synth_to_real", H.SEED)
    #real to real 
    r2r_auc, r2r_acc = train_on_real_test_on_real(f"{H.OUT_DIR}/ckpt_{checkpoint}/real_to_real", df_real_with_patients_norm, H.SEED)
    #classify real vs syn 
    df_real = df_real_with_patients_norm.copy().drop(columns=["patient_id"])
    df_real["is_syn"] = 0
    df_syn = df_syn_norm.copy()
    df_syn["is_syn"] = 1 
    fraction = len(df_real) / len(df_syn)
    if fraction < 1: 
        df_syn = df_syn.sample(frac=fraction, random_state=H.SEED).reset_index(drop=True)
    elif fraction > 1: 
        new_frac = len(df_syn) / len(df_real) 
        df_real =  df_real.sample(frac=new_frac, random_state=H.SEED).reset_index(drop=True)
    df_all = pd.concat([df_real, df_syn], ignore_index=True)
    rvs_auc, rvs_acc = classify_real_vs_syn(df_all, f"{H.OUT_DIR}/ckpt_{checkpoint}/real_vs_syn", H.SEED)
    #-----------------PRIVACY---------------------#
    mem_aucs = mem_risk(df_train_norm, df_hold_norm, df_syn_norm, H.CAT_COLS, H.NUM_COLS, f"{H.OUT_DIR}/ckpt_{checkpoint}/mem_risk", H.SEED) 
    #-----------------LOG RESULTS-----------------#
    with open(f"{H.OUT_DIR}/ckpt_{checkpoint}/eval.txt", "a") as f:
        f.write(f"CWC: {cwc}\noverall_score_hold : {overall_score_hold}\noverall_score_real: {overall_score_real}\noverall_score_real_num: {overall_score_real_num}\noverall_score_hold_num: {overall_score_hold_num}\noverall_score_real_cat: {overall_score_real_cat}\noverall_score_hold_cat:{overall_score_hold_cat}\n"
            f"r2r_auc: {r2r_auc}\nr2r_acc: {r2r_acc}\nr2s_auc: {r2s_auc}\nr2s_acc: {r2s_acc}\ns2r_auc: {s2r_auc}\ns2r_acc: {s2r_acc}\ns2h_auc: {s2h_auc}\ns2h_acc: {s2h_acc}\nrvs_auc: {rvs_auc}\nrvs_acc: {rvs_acc}\n"
            f"real_mem_auc_: {mem_aucs['real_roc_auc']}\nreal_mem_auc_bal: {mem_aucs['real_roc_auc_bal']}\nsynth_mem_auc: {mem_aucs['synth_roc_auc']}\nsynth_mem_auc_bal: {mem_aucs['synth_roc_auc_bal']}\nelapsed_time: {elapsed_time}")
    log_result_EMR(H.RESULT_CSV, H.RUN_NAME, H.EPOCHS, H.DATA_SIZE, H.SEED, checkpoint, cwc, overall_score_real_num, overall_score_hold_num,  overall_score_real_cat, overall_score_hold_cat, overall_score_real, overall_score_hold, 
            r2r_auc, r2r_acc, s2h_auc, s2h_acc, s2r_auc, s2r_acc, r2s_auc, r2s_acc, rvs_auc, rvs_acc, mem_aucs["real_roc_auc"], mem_aucs["real_roc_auc_bal"], mem_aucs["synth_roc_auc"], mem_aucs["synth_roc_auc_bal"], elapsed_time)

def list_checkpoints(folder):
    files = os.listdir(folder)
    ckpts = []
    for f in files:
        match = re.match(r"ckpt-(\d+)\.", f)
        if match:
            ckpts.append(match.group(1))
    # deduplicate + sort numerically
    ckpts = sorted(set(ckpts), key=int)
    return [str(c) for c in ckpts]



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--OUT_DIR", type=str, required=True )
    p.add_argument("--RESULT_CSV", type=str, required=True)
    p.add_argument("--RUN_NAME", type=str, required=True)
    p.add_argument("--DATASET", choices=["AIREADI", "MIMIC"], required=True)
    p.add_argument("--DATA_PATH", type=str, required=True)
    p.add_argument("--SEED", type=int, required=True)
    p.add_argument("--eval", action='store_true')
    p.add_argument("--ckpt", ,type=str, default=None)
    
    #add more here if you want to override other hps. 
    return vars(p.parse_args())


def main(): 
    #hyperparameter set up and arg parsing 
    global H 
    args = parse_args() 
    if args["DATASET"] == "AIREADI": 
        H = HyperParams_AIREADI().override(**args)
        H = H.override(
            NPY_PATH = f"{H.DATA_PATH}/min_max_log.npy",
            CHECKPOINT_DIRECTORY = f"EMR_WGAN_training_checkpoints_AIREADIFULL_BEST/{H.RUN_NAME}", 
            MODEL_ID = f"EMR_AIREADIFULL_{H.RUN_NAME}", 
        )
    elif args["DATASET"] == "MIMIC": 
        H = HyperParams_MIMIC().override(**args)
        H = H.override(
            NPY_PATH = f"{H.DATA_PATH}/min_max_log.npy",
            CHECKPOINT_DIRECTORY = f"EMR_WGAN_training_checkpoints_MIMIC/{H.RUN_NAME}", 
            MODEL_ID = f"EMR_MIMIC_{H.RUN_NAME}", 
        )
    #set seeds and directory
    set_global_seed(H.SEED)
    os.makedirs(H.OUT_DIR, exist_ok=True)
    #train and log 
    if not args.eval:
        log, fh = make_logger(H.MODEL_ID, H.OUT_DIR)
        try: 
            elapsed_time, best_epoch = train(H, log)
            print("training complete")
        finally: 
            fh.close() 
        print('train complete')
    else: 
        if args.ckpt is None: 
            checkpoints = [x.split('ckpt-')[1].split('.data')[0] for x in os.listdir(H.CHECKPOINT_DIRECTORY) if '.data' in x] #[str(best_epoch), str(H.EPOCHS-10)]
        else: 
            checkpoints = [int(ckpt)]
        print("Evaluating checkpoints", checkpoints)
        elapsed_time=0
    
    ##generate/evaluate from the 'best' checkpoint (lowest loss) AND the last checkpoint (last epoch)
    for checkpoint in checkpoints: 
        if f'ckpt_{checkpoint}' in os.listdir(H.OUT_DIR):
            gen_success = 1
            print(checkpoint)
        else:
            print("NEED TO GENERATE")
            gen_success = gen(H, checkpoint)
        if gen_success != -1: 
            #evaluate 
            if H.DATASET == "AI-READI-OG" or H.DATASET == "AI-READI-FULL": 
                #get raw to use for cwc, value stat analysis, histograms etc. 
                df_train = pd.read_csv(f"{H.DATA_PATH}/original_training_data.csv")[H.NUM_COLS+ H.CAT_COLS]
                df_hold = pd.read_csv(f"{H.DATA_PATH}/original_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
                df_real = pd.read_csv(f"{H.DATA_PATH}/original_data_with_patients.csv").drop(columns=['patient_id'])[H.NUM_COLS+H.CAT_COLS]
                df_syn = pd.read_csv(f"{H.OUT_DIR}/ckpt_{checkpoint}/synthetic_rescaled.csv")[H.NUM_COLS+H.CAT_COLS]
                #get normalized data to use for classifications (need patients to split real data without leakage)
                df_hold_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_testing_data.csv")[H.NUM_COLS+ H.CAT_COLS]
                df_real_with_patients_norm =  pd.read_csv(f"{H.DATA_PATH}/preprocessed_data_with_patients.csv")[H.NUM_COLS + H.CAT_COLS +["patient_id"]]
                df_syn_norm = pd.read_csv(f"{H.OUT_DIR}/ckpt_{checkpoint}/synthetic.csv")[H.NUM_COLS+H.CAT_COLS]
                df_train_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_training_data.csv")[H.NUM_COLS+ H.CAT_COLS]
                evaluate_model_AIREADI(df_real, df_hold, df_train, df_syn, df_hold_norm, df_train_norm, df_syn_norm, df_real_with_patients_norm, H, elapsed_time, checkpoint) 
            else: 
                df_real = pd.read_csv(f"{H.DATA_PATH}/original_testing_data_0.csv")[H.NUM_COLS+ H.CAT_COLS]
                df_train = pd.read_csv(f"{H.DATA_PATH}/original_training_data_0.csv")[H.NUM_COLS+H.CAT_COLS]
                df_syn = pd.read_csv(f"{H.OUT_DIR}/ckpt_{checkpoint}/synthetic_rescaled.csv")[H.NUM_COLS+H.CAT_COLS]
                #get normalized data to use for classifications (need patients to split real data without leakage)
                df_hold_norm = pd.read_csv(f"{H.DATA_PATH}/normalized_testing_data_0.csv")[H.NUM_COLS+ H.CAT_COLS]
                df_train_norm =  pd.read_csv(f"{H.DATA_PATH}/normalized_training_data_0.csv")[H.NUM_COLS + H.CAT_COLS] 
                df_syn_norm = pd.read_csv(f"{H.OUT_DIR}/ckpt_{checkpoint}/synthetic.csv")[H.NUM_COLS+H.CAT_COLS]
                #evaluate 
                s2h_auc, r2s_auc = evaluate_model_MIMIC(df_train, df_real, df_syn, df_train_norm, df_hold_norm, df_train_norm, df_syn_norm, elapsed_time, H.OUT_DIR, H.CAT_COLS, H.SEED, H.RESULT_CSV, H.MODEL_ID) 
    

            
if __name__ == '__main__':
    main() 
   