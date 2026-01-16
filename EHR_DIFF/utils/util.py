import torch
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
# import PIL
# from torchvision.utils import make_grid
from scipy import linalg
from scipy.stats import pearsonr
from pathlib import Path

# from dataset_tool import is_image_ext


def average_tensor(t):
    size = float(dist.get_world_size())
    dist.all_reduce(t.data, op=dist.ReduceOp.SUM)
    t.data /= size


def set_seeds(rank, seed):
    random.seed(rank + seed)
    torch.manual_seed(rank + seed)
    np.random.seed(rank + seed)
    torch.cuda.manual_seed(rank + seed)
    torch.cuda.manual_seed_all(rank + seed)
    torch.backends.cudnn.benchmark = True


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    # else:
    #     raise ValueError('Directory already exists.')


def add_dimensions(x, n_additional_dims):
    for _ in range(n_additional_dims):
        x = x.unsqueeze(-1)

    return x


def save_checkpoint(ckpt_path, state):
    saved_state = {'model': state['model'].state_dict(),
                   'ema': state['ema'].state_dict(),
                   'optimizer': state['optimizer'].state_dict(),
                   'step': state['step']}
    torch.save(saved_state, ckpt_path)


#added function 
def postprocess_columns(X, binary_cols=None, continuous_cols=None):
    X = np.asarray(X, dtype=np.float32, order='C')
    N, D = X.shape
    # binary mask
    bin_mask = (np.array(binary_cols, dtype=bool)
                if (binary_cols is not None and np.asarray(binary_cols).dtype == bool and len(binary_cols) == D)
                else np.zeros(D, dtype=bool))
    if bin_mask.sum() == 0 and binary_cols is not None and not bin_mask.any():
        bin_mask = np.zeros(D, dtype=bool)
        bin_mask[np.array(binary_cols, dtype=int)] = True
    # continuous mask
    cont_mask = (np.array(continuous_cols, dtype=bool)
                 if (continuous_cols is not None and np.asarray(continuous_cols).dtype == bool and len(continuous_cols) == D)
                 else np.zeros(D, dtype=bool))
    if cont_mask.sum() == 0 and continuous_cols is not None and not cont_mask.any():
        cont_mask = np.zeros(D, dtype=bool)
        cont_mask[np.array(continuous_cols, dtype=int)] = True
    # apply EHR_DIFF-style postprocess
    if bin_mask.any():
        X[:, bin_mask] = np.rint(np.clip(X[:, bin_mask], 0.0, 1.0))
    if cont_mask.any():
        X[:, cont_mask] = np.clip(X[:, cont_mask], 0.0, 1.0)
    return X

    
def sample_random_batch(EHR_task, sampling_shape, sampler, path, device, n_classes=None, name='sample', config=None):
    # make_dir(path)

    x = torch.randn(sampling_shape, device=device)
    if n_classes is not None:
        y = torch.randint(n_classes, size=(
            sampling_shape[0],), dtype=torch.int32, device=device)
    else:
        y = None
    x = sampler(x, y).cpu()

    # x[np.isnan(x)] = 0

    #removed
    ''' 
    if EHR_task == 'binary':
        x = np.rint(np.clip(x, 0, 1))
    elif EHR_task == 'continuous':
        x = np.clip(x, 0, 1)
    '''
    #added code 
    x = x.detach().numpy()
    BINARY_COL = getattr(config.data, 'binary_cols', None)       
    CONTINUOUS_COL = getattr(config.data, 'continuous_cols', None) 
    x = postprocess_columns(x, binary_cols=BINARY_COL, continuous_cols=CONTINUOUS_COL)
    np.save(os.path.join(path, name), x)


# ------------------------------------------------------------------------------------
def plot_dim_dist(train_data, syn_data, save_dir):

    train_data_mean = np.mean(train_data, axis = 0)
    temp_data_mean = np.mean(syn_data, axis = 0)

    corr = pearsonr(temp_data_mean, train_data_mean)
    nzc = sum(temp_data_mean[i] > 0 for i in range(temp_data_mean.shape[0]))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    slope, intercept = np.polyfit(train_data_mean, temp_data_mean, 1)
    fitted_values = [slope * i + intercept for i in train_data_mean]
    identity_values = [1 * i + 0 for i in train_data_mean]

    ax.plot(train_data_mean, fitted_values, 'b', alpha=0.5)
    ax.plot(train_data_mean, identity_values, 'r', alpha=0.5)
    ax.scatter(train_data_mean, temp_data_mean, alpha=0.3)
    ax.set_title('corr: %.4f, none-zero columns: %d, slope: %.4f'%(corr[0], nzc, slope))
    ax.set_xlabel('Feature prevalence of real data')
    ax.set_ylabel('Feature prevalence of synthetic data')

    fig.savefig(save_dir + '/{}.png'.format('Dimension-Wise Distribution'))
    plt.close(fig)

    return corr[0], nzc
