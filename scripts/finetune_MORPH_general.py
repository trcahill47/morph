from fileinput import filename

from json import load

import os

import sys

import csv

import argparse

import torch

import torch.nn as nn

import torch.nn.functional as F

import matplotlib

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

import numpy as np

import pandas as pd

from tqdm import tqdm

from huggingface_hub import hf_hub_download

from skimage.metrics import structural_similarity as ssim

from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ["MPLBACKEND"] = "Agg"

matplotlib.use("Agg")



# Add project root to Python path

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(project_root)



from src.utils.device_manager import DeviceManager

from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression

from src.utils.metrics_3d import Metrics3DCalculator

from src.utils.visualize_predictions_3d_full import Visualize3DPredictions

from src.utils.visualize_rollouts_3d_full import Visualize3DRolloutPredictions

from src.utils.data_preparation_fast import FastARDataPreparer

from config.data_config import DataConfig

from src.utils.normalization import RevIN

from src.utils.uptf7 import UPTF7

from pathlib import Path

from sklearn.model_selection import train_test_split



#%% Argument parser

MORPH_MODELS = {

    'Ti': [8, 256,  4,  4, 1024],

    'S' : [8, 512,  8,  4, 2048],

    'M' : [8, 768, 12,  8, 3072],

    'L' : [8, 1024,16, 16, 4096]

    }



patch_size = 8

DATA_CONFIG = DataConfig(project_root, patch_size)



parser = argparse.ArgumentParser(description="Run inference on trained ViT3D model")

parser.add_argument('--dataset', type=str, help="Path to saved .npy/.pth", required=True)

parser.add_argument('--dataset_name', type=str, help="Name of the dataset", default=None)

parser.add_argument('--dataset_specs', nargs=5, type=int, metavar=('F','C','D','H','W'),

                    default=[2,1,1,128,128], help="Dataset specs")

parser.add_argument('--model_choice', type=str, default='FM', help="Model to finetune")

parser.add_argument('--download_model', action='store_true', help="Download model weights")

parser.add_argument('--model_size', type=str, choices=list(MORPH_MODELS.keys()),

                    default='Ti', help='choose from Ti, S, M, L')

parser.add_argument('--ckpt_from', type=str, choices=['FM','FT'], default='FM',

                    help="Checkpoint information from FM or previous FT", required=True)

parser.add_argument('--checkpoint', type=str, default=None,

                    help="Path to saved .pth state dict if download_model is False")



# Finetune levels

parser.add_argument('--ft_level1', action='store_true', help="Level-1 finetuning (LoRA, PE, LN)")

parser.add_argument('--ft_level2', action='store_true', help="Level-2 finetuning (Encoder)")

parser.add_argument('--ft_level3', action='store_true', help="Level-3 finetuning (Decoder)")

parser.add_argument('--ft_level4', action='store_true', help="All model parameters")

parser.add_argument('--lr_level4', type=float, default=1e-4,

                    help="Learning rate for level-4 finetuning")

parser.add_argument('--wd_level4', type=float, default=0.0,

                    help="Weight decay for level-4 finetuning")



# Parallelization

parser.add_argument('--parallel', type=str, choices=['dp','no'], default='dp',

                    help="DataParallel vs No parallelization")



# LoRA parameters

parser.add_argument('--rank_lora_attn', type=int, default=16)

parser.add_argument('--rank_lora_mlp', type=int, default=12)

parser.add_argument('--lora_p', type=float, default=0.05)



# Data and compute

parser.add_argument('--n_epochs', type=int, default=150, help="Fine-tuning epochs")

parser.add_argument('--n_traj', type=int, help="Fine-tuning trajectories")

parser.add_argument('--rollout_horizon', type=int, default=10,

                    help="Visualization: single step & rollouts")



# Model default hyperparameters

parser.add_argument('--tf_reg', nargs=2, type=float, metavar=('dropout','emb_dropout'),

                    default=[0.1,0.1])

parser.add_argument('--heads_xa', type=int, default=32)

parser.add_argument('--ar_order', type=int, default=1)

parser.add_argument('--max_ar_order', type=int, default=1)

parser.add_argument('--test_sample', type=int, default=0)

parser.add_argument('--device_idx', type=int, default=0)

parser.add_argument('--patience', type=int, default=10)



# Optimizer hyperparameters

parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--lr', type=float, default=1e-4)

parser.add_argument('--weight_decay', type=float, default=0.0)

parser.add_argument('--lr_scheduler', action='store_true')



# Save related

parser.add_argument('--overwrite_weights', action='store_true')

parser.add_argument('--save_every', type=int, default=1)

parser.add_argument('--save_batch_ckpt', action='store_true')

parser.add_argument('--save_batch_freq', type=int, default=1000)



# ============================================================

# NEW ARGUMENTS FOR SWEEP SYSTEM

# ============================================================

parser.add_argument('--sweep_mode', action='store_true',

                    help="Suppress all PNG/visual output. Only save CSV metrics and .pth files.")

parser.add_argument('--checkpoint_every', type=int, default=0,

                    help="If >0, evaluate and record metrics every N epochs during training. "

                         "Used for epoch-axis sweep. Saves a .pth per checkpoint.")

parser.add_argument('--train_pct', type=int, default=100,

                    help="Percentage of training data used (for logging to master CSV only).")

parser.add_argument('--master_csv_path', type=str, default=None,

                    help="Path to the master metrics CSV. If provided, appends a row after "

                         "each checkpoint evaluation.")

parser.add_argument('--best_model_dir', type=str, default=None,

                    help="Directory to save the best .pth. In fraction mode, one file per "

                         "data pct. In epoch mode, one file per checkpoint epoch.")

# ============================================================



args = parser.parse_args()

devices = DeviceManager.list_devices()

filters, dim, heads, depth, mlp_dim = MORPH_MODELS[args.model_size]

dropout, emb_dropout = args.tf_reg

device = devices[args.device_idx] if devices else 'cpu'



# Default paths

savepath_results = os.path.join(project_root, "experiments", "results", "test")

os.makedirs(savepath_results, exist_ok=True)

savepath_model = os.path.join(project_root, "models")

os.makedirs(savepath_model, exist_ok=True)

loadpath_muvar = os.path.join(project_root, 'data')

os.makedirs(loadpath_muvar, exist_ok=True)

loadpath_datasets = os.path.join(project_root, 'datasets')

os.makedirs(loadpath_datasets, exist_ok=True)



# Load dataset

print(f'→ Loading dataset ...')

ext = Path(args.dataset).suffix.lower()

print(f'→ Dataset format: {ext}')



if ext == ".npy":

    print(f'→ Loading .npy dataset from {args.dataset}')

    dataset = np.load(os.path.join(loadpath_datasets, args.dataset))

    print("Original dataset shape", dataset.shape)

else:

    raise ValueError("Unsupported dataset format. Need a .npy file.")



# Set fname unconditionally so it's always defined when loading weights below

# regardless of whether --download_model is passed

if args.model_size == 'Ti':

    fname = 'morph-Ti-FM-max_ar1_ep225.pth'

elif args.model_size == 'S':

    fname = 'morph-S-FM-max_ar1_ep225.pth'

elif args.model_size == 'M':

    fname = 'morph-M-FM-max_ar1_ep290_latestbatch.pth'

elif args.model_size == 'L':

    fname = 'morph-L-FM-max_ar16_ep189_latestbatch.pth'



# Download model if needed

if args.download_model and args.ckpt_from == 'FM' and args.checkpoint == None:

    print(f'→ Downloading model weights from hugging face...')



    weights_path = hf_hub_download(

        repo_id="mahindrautela/MORPH",

        filename=fname,

        subfolder="models/FM",

        repo_type="model",

        resume_download=True,

        local_dir=".",

        local_dir_use_symlinks=False

    )

    print(f'MORPH-FM-{args.model_size} weights saved to {weights_path}')



num_samples = dataset.shape[0]

traj_len = dataset.shape[1]



# Convert to UPTF7 format

dataset_uptf7 = UPTF7(

    dataset=dataset,

    num_samples=num_samples,

    traj_len=traj_len,

    fields=args.dataset_specs[0],

    components=args.dataset_specs[1],

    image_depth=args.dataset_specs[2],

    image_height=args.dataset_specs[3],

    image_width=args.dataset_specs[4]

).transform()

print("UPTF7 dataset shape", dataset_uptf7.shape)



# RevIN normalization

dataset_name = 'name' if args.dataset_name is None else args.dataset_name

revin = RevIN(loadpath_muvar)

norm_prefix = f'norm_{dataset_name}'



revin.compute_stats(dataset_uptf7, prefix=norm_prefix)

dataset_norm = revin.normalize(dataset_uptf7, prefix=norm_prefix)

print("Normalize dataset shape", dataset_norm.shape)



recovered = revin.denormalize(dataset_norm, prefix=norm_prefix)

tol_6 = 1e-2

max_error = 0.0

for i in range(recovered.shape[0]):

    maxerror_i = np.max(np.abs(recovered[i] - dataset_uptf7[i]))

    max_error = max(maxerror_i, max_error)

assert max_error < tol_6, "Denormalization did not perfectly recover original!"

print("RevIN round-trip OK")

del recovered



# Finetune level

if args.ft_level4:

    lev = 4

    args.ft_level1 = args.ft_level2 = args.ft_level3 = None

    args.rank_lora_attn = 0

    args.rank_lora_mlp = 0

    args.lora_p = 0

elif args.ft_level1 and args.ft_level2 and args.ft_level3:

    lev = 3

elif args.ft_level1 and args.ft_level2:

    lev = 2

elif args.ft_level1:

    lev = 1

else:

    raise ValueError("Select a fine-tuning level: --ft_level1/2/3 or --ft_level4")

print(f"→ Set Level-{lev} fine-tuning")



# Create dataloaders

class DatasetforDataloader(Dataset):

    def __init__(self, X, y):

        self.X, self.y = X, y

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]



print("→ Data splitting...")

dataset_norm_rs = dataset_norm.transpose(0,1,4,5,6,3,2)

train_data, tmp = train_test_split(dataset_norm_rs, test_size=0.2, random_state=42, shuffle=True)

val_data, test_data = train_test_split(tmp, test_size=0.5, random_state=42, shuffle=True)



n_traj = train_data.shape[0] if args.n_traj is None else args.n_traj

print(f'→ [{dataset_name}] Number of finetuning trajectories: {n_traj}')



print(f'→ [{dataset_name}] Dataset preparation...')

preparer = FastARDataPreparer(ar_order=args.ar_order)

X_tr, y_tr = preparer.prepare(train_data[0:n_traj])

X_va, y_va = preparer.prepare(val_data)

X_te, y_te = preparer.prepare(test_data)

print(f'→ Training Inputs: {X_tr.shape} and Targets: {y_tr.shape}')

assert X_tr.shape[0] == n_traj * (train_data.shape[1] - 1), "Shape mismatch !!"



del train_data, val_data



# Model architecture

patch_size  = 8

max_patches = 4096

max_fields  = 3

max_components = 3



model_name = (f'ft_morph-{args.model_size}-{dataset_name}-max_ar{args.max_ar_order}_'

              f'rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}')



ft_model = ViT3DRegression(

    patch_size=patch_size, dim=dim, depth=depth,

    heads=heads, heads_xa=args.heads_xa, mlp_dim=mlp_dim,

    max_components=max_components, conv_filter=filters,

    max_ar=args.max_ar_order,

    max_patches=max_patches, max_fields=max_fields,

    dropout=dropout, emb_dropout=emb_dropout,

    lora_r_attn=args.rank_lora_attn,

    lora_r_mlp=args.rank_lora_mlp,

    lora_alpha=None,

    lora_p=args.lora_p

).to(device)



num_params_model = sum(p.numel() for p in ft_model.parameters()) / 1e6

print(f"→ NUMBER OF PARAMETERS OF THE MODEL (in M): {num_params_model:.3g}")



n_gpus = torch.cuda.device_count()

print(f'→ Finetuning on {n_gpus} GPUs')

batch_size = args.batch_size

if args.parallel == 'dp' and n_gpus > 1:

    ft_model = nn.DataParallel(ft_model)

    batch_size = n_gpus * args.batch_size

print(f'→ Selected (Overall) Batch size for {dataset_name} is {batch_size}')



ft_tr = DatasetforDataloader(X_tr, y_tr)

ft_va = DatasetforDataloader(X_va, y_va)

ft_te = DatasetforDataloader(X_te, y_te)



ft_tr_loader = DataLoader(ft_tr, batch_size=batch_size, shuffle=True)

ft_va_loader = DataLoader(ft_va, batch_size=batch_size, shuffle=False)

ft_te_loader = DataLoader(ft_te, batch_size=batch_size, shuffle=False)

print(f'→ Length dataloader: Tr {len(ft_tr_loader)}, Val {len(ft_va_loader)}, Te {len(ft_te_loader)}')



# Fine-tuning setup

from src.utils.select_fine_tuning_parameters import SelectFineTuningParameters

selector = SelectFineTuningParameters(ft_model, args)

optimizer = selector.configure_levels()

ft_model.train().to(device)

start_lr = optimizer.param_groups[0]['lr']

start_wd = optimizer.param_groups[0]['weight_decay']

print(f'→ Starting from LR: {start_lr} & Weight Decay: {start_wd}')



criterion = nn.MSELoss()

scheduler = None

if args.lr_scheduler:

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',

                                                            factor=0.5, patience=5)



# Load weights

start_epoch = 0

if args.ckpt_from == 'FM':

    print(f"→ Loading checkpoints from {args.ckpt_from}")

    if args.checkpoint is None:

        checkpoint = fname

    else:

        checkpoint = args.checkpoint

    checkpoint_path = os.path.join(savepath_model, f'{args.model_choice}', checkpoint)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    state_dict = ckpt["model_state_dict"]



    target = ft_model.module if isinstance(ft_model, nn.DataParallel) else ft_model



    if state_dict and next(iter(state_dict)).startswith("module.") and args.parallel == 'no':

        print("→ Stripping 'module.' from checkpoints")

        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}



    flag = True if args.ft_level4 else False

    missing, unexpected = target.load_state_dict(state_dict, strict=flag)



    print("Missing keys (expected: LoRA A/B etc.):",

          [k for k in missing if k.endswith((".A", ".B")) or ".lora" in k])

    print("Unexpected keys:", unexpected)

    print(f"→ Resumed from {checkpoint_path}, starting at epoch {start_epoch}")



elif args.ckpt_from == 'FT':

    print(f"→ Loading checkpoints from {args.ckpt_from}")

    resume_path = os.path.join(savepath_model, f'{dataset_name}', args.checkpoint)

    ckpt = torch.load(resume_path, map_location=device, weights_only=True)

    state_dict = ckpt["model_state_dict"]



    target = ft_model.module if isinstance(ft_model, nn.DataParallel) else ft_model



    if any(k.startswith("module.") for k in state_dict.keys()):

        print("→ Stripping 'module.' from checkpoints")

        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}



    target.load_state_dict(state_dict, strict=True)

    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = ckpt["epoch"]

    print(f"→ Resumed from {resume_path}, starting at epoch {start_epoch}")

else:

    print('No model is loaded')





# ============================================================

# EVALUATION HELPER — called at checkpoints and end of training

# Computes all metrics + rollout, appends to master CSV.

# In sweep_mode: skips all PNG generation.

# In normal mode: generates full visual outputs as before.

# ============================================================

def run_evaluation(epoch_num, current_best_eval_mse=float('inf')):

    """

    Run full evaluation at a given epoch checkpoint.

    Computes MSE, RMSE, NRMSE, per-step rollout MSE and SSIM.

    Writes to master CSV. Saves .pth if best_model_dir is set.

    Generates visuals only when sweep_mode is False.

    """

    print(f"\n→ [{dataset_name}] Running evaluation at epoch {epoch_num}...")

    ft_model.eval()



    # ----- Aggregate test metrics -----

    out_all, tar_all = [], []

    with torch.no_grad():

        for inp, tar in tqdm(ft_te_loader):

            inp = inp.to(device)

            _, _, out = ft_model(inp)

            out_all.append(out.detach().cpu())

            tar_all.append(tar)

    out_all = torch.concat(out_all, dim=0)

    tar_all = torch.concat(tar_all, dim=0)



    mse  = F.mse_loss(out_all, tar_all, reduction='mean')

    mae  = F.l1_loss(out_all, tar_all, reduction='mean')

    rmse = mse ** 0.5



    td_out = torch.from_numpy(test_data[:, 1:])

    td_out = td_out.permute(0, 1, 6, 5, 2, 3, 4)

    out_all_rs = out_all.reshape(td_out.shape)

    tar_all_rs = tar_all.reshape(td_out.shape)



    outputs_denorm = RevIN.denormalize_testeval(loadpath_muvar, norm_prefix, out_all_rs,

                                                dataset=dataset_name)

    targets_denorm = RevIN.denormalize_testeval(loadpath_muvar, norm_prefix, tar_all_rs,

                                                dataset=dataset_name)



    vrmse = Metrics3DCalculator.calculate_VRMSE(outputs_denorm, targets_denorm).mean()

    nrmse = Metrics3DCalculator.calculate_NRMSE(outputs_denorm, targets_denorm).mean()



    print(f'→ Epoch {epoch_num} | MSE:{mse:.5f} RMSE:{rmse:.5f} MAE:{mae:.5f} '

          f'VRMSE:{vrmse:.5f} NRMSE:{nrmse:.5f}')



    # ----- Per-timestep rollout (MSE + SSIM) -----

    sim = test_data[args.test_sample]

    sim_rs = np.transpose(sim, (0, 5, 4, 1, 2, 3)).astype(np.float32)

    sim_tensor = torch.from_numpy(sim_rs).unsqueeze(0)



    per_step_mse  = []

    per_step_ssim = []



    with torch.no_grad():

        sim_input = sim_tensor[:, 0:1].to(device)

        for t in range(args.rollout_horizon):

            _, _, pred = ft_model(sim_input)

            target_t = sim_tensor[:, t + 1].to(device)



            step_mse = F.mse_loss(pred.squeeze(1), target_t).item()

            per_step_mse.append(step_mse)



            pred_np = pred.squeeze().cpu().numpy()

            targ_np = target_t.squeeze().cpu().numpy()

            drange  = targ_np.max() - targ_np.min()

            try:

                step_ssim = ssim(targ_np, pred_np, data_range=drange, channel_axis=0)

            except ValueError:

                step_ssim = ssim(targ_np, pred_np, data_range=drange)

            per_step_ssim.append(step_ssim)



            sim_input = pred.unsqueeze(1)



    mean_rollout_mse  = float(np.mean(per_step_mse))

    mean_rollout_ssim = float(np.mean(per_step_ssim))

    final_rollout_mse = per_step_mse[-1]

    final_rollout_ssim = per_step_ssim[-1]



    # ----- Append to master CSV -----

    if args.master_csv_path:

        master_csv = args.master_csv_path

    else:

        master_csv = os.path.join(project_root, "sweep_results",

                                  dataset_name, "master_metrics.csv")

    os.makedirs(os.path.dirname(master_csv), exist_ok=True)



    file_exists = os.path.isfile(master_csv)

    with open(master_csv, mode='a', newline='') as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow([

                'dataset', 'model_size', 'train_pct', 'n_traj',

                'epoch', 'mse', 'rmse', 'nrmse', 'vrmse', 'mae',

                'mean_rollout_mse', 'mean_rollout_ssim',

                'final_rollout_mse', 'final_rollout_ssim',

                'per_step_mse', 'per_step_ssim'

            ])

        writer.writerow([

            dataset_name, args.model_size, args.train_pct, n_traj,

            epoch_num,

            mse.item(), rmse.item(), nrmse.item(), vrmse.item(), mae.item(),

            mean_rollout_mse, mean_rollout_ssim,

            final_rollout_mse, final_rollout_ssim,

            str(per_step_mse), str(per_step_ssim)

        ])

    print(f"→ Appended metrics row to {master_csv}")



    # ----- Save .pth to best_model_dir if specified -----

    if args.best_model_dir:

        os.makedirs(args.best_model_dir, exist_ok=True)

        if args.checkpoint_every > 0:

            # Epoch-axis mode: save one .pth per checkpoint unconditionally

            pth_name = f"{args.model_size}_{args.train_pct}pct_ep{epoch_num}.pth"

            pth_path = os.path.join(args.best_model_dir, pth_name)

            torch.save({

                "epoch": epoch_num,

                "model_state_dict": ft_model.state_dict(),

                "optimizer_state_dict": optimizer.state_dict(),

                "mse": mse.item(),

                "nrmse": nrmse.item()

            }, pth_path)

            print(f"→ Saved epoch checkpoint: {pth_path}")

        else:

            # Fraction mode: only overwrite if this is the best MSE seen so far

            if mse.item() < current_best_eval_mse:

                pth_name = f"best_{args.model_size}_{args.train_pct}pct.pth"

                pth_path = os.path.join(args.best_model_dir, pth_name)

                torch.save({

                    "epoch": epoch_num,

                    "model_state_dict": ft_model.state_dict(),

                    "optimizer_state_dict": optimizer.state_dict(),

                    "mse": mse.item(),

                    "nrmse": nrmse.item()

                }, pth_path)

                print(f"→ New best model saved (MSE {mse.item():.5f} < {current_best_eval_mse:.5f}): {pth_path}")

            else:

                print(f"→ Model NOT saved — MSE {mse.item():.5f} did not beat current best {current_best_eval_mse:.5f}")



    # ----- Visuals (skipped in sweep_mode) -----

    if not args.sweep_mode:

        savepath_results_ = os.path.join(savepath_results, f'{dataset_name}')

        os.makedirs(savepath_results_, exist_ok=True)



        field_names = [f'field-{fi}' for fi in range(1, sim_tensor.shape[2] + 1)]

        figurename = (f'ft_st_MORPH-{args.model_size}_{args.model_choice}_ar{args.max_ar_order}_chAll_'

                      f'samp{args.test_sample}_tot-trajs{n_traj}_tot-eps{epoch_num}_'

                      f'rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}_t')



        viz = Visualize3DPredictions(ft_model, sim_tensor, device)

        for t in range(args.rollout_horizon):

            viz.visualize_predictions(time_step=t, component=0, slice_dim='d',

                                      save_path=savepath_results_,

                                      figname=f'{figurename}{t}.png')



        viz_roll = Visualize3DRolloutPredictions(model=ft_model, test_dataset=sim_tensor,

                                                 device=device, field_names=field_names,

                                                 component_names=["d","h","w"])

        figurename_ro = (f'ft_ro_MORPH-{args.model_size}_{args.model_choice}_ar{args.max_ar_order}_tAll_'

                         f'samp{args.test_sample}_tot-trajs{n_traj}_tot-eps{epoch_num}_'

                         f'rank-lora{args.rank_lora_attn}_ftlevel{lev}_lr{args.lr_level4}_wd{args.wd_level4}_field')

        for fi in range(sim_tensor.shape[2]):

            viz_roll.visualize_rollout(start_step=0, num_steps=args.rollout_horizon, field=fi,

                                       component=0, slice_dim='d',

                                       save_path=savepath_results_,

                                       figname=f'{figurename_ro}{fi}.png')



        # Dual-axis MSE/SSIM plot

        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_mse = 'tab:red'

        ax1.set_xlabel('Rollout Timestep')

        ax1.set_ylabel('MSE', color=color_mse, fontsize=12)

        ax1.plot(range(1, args.rollout_horizon + 1), per_step_mse,

                 marker='o', color=color_mse, lw=2, label='MSE')

        ax1.tick_params(axis='y', labelcolor=color_mse)

        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()

        color_ssim = 'tab:blue'

        ax2.set_ylabel('SSIM', color=color_ssim, fontsize=12)

        ax2.plot(range(1, args.rollout_horizon + 1), per_step_ssim,

                 marker='s', color=color_ssim, lw=2, label='SSIM')

        ax2.tick_params(axis='y', labelcolor=color_ssim)

        ax2.set_ylim(0, 1.05)

        plt.title(f'Rollout: {dataset_name} | {args.model_size} | ep{epoch_num}', fontsize=14)

        fig.tight_layout()

        fig.savefig(os.path.join(savepath_results_,

                                 f'dual_metrics_{args.model_size}_{dataset_name}_ep{epoch_num}.png'))

        plt.close(fig)



        # Loss curve

        fig, ax = plt.subplots()

        ax.plot(train_losses, label='Train')

        ax.plot(val_losses, label='Val')

        ax.set_xlabel('Epoch')

        ax.set_ylabel('Loss')

        ax.legend()

        fig.savefig(os.path.join(savepath_results_,

                                 f'loss_{args.model_size}_{dataset_name}_ep{epoch_num}.png'))

        plt.close(fig)



    ft_model.train()

    return mse.item()

# ============================================================





# Fine-tuning loop

import time

from src.utils.trainers import Trainer



savepath_model_folder = os.path.join(savepath_model, f'{dataset_name}')

os.makedirs(savepath_model_folder, exist_ok=True)

model_path = os.path.join(savepath_model_folder, model_name)



train_losses, val_losses = [], []

best_val_loss = float('inf')

best_eval_mse = float('inf')

epochs_no_improve = 0

last_evaluated_epoch = -1

ep_st = time.time()



for epoch in range(start_epoch, args.n_epochs):

    tr_loss = Trainer.train_singlestep(ft_model, ft_tr_loader, criterion, optimizer, device,

                                       epoch, scheduler, model_path,

                                       args.save_batch_ckpt, args.save_batch_freq)

    vl_loss = Trainer.validate_singlestep(ft_model, ft_va_loader, criterion, device)



    train_losses.append(tr_loss)

    val_losses.append(vl_loss)



    if args.lr_scheduler:

        scheduler.step(vl_loss)



    current_lr = optimizer.param_groups[0]['lr']

    print(f"Time = {(time.time()-ep_st)/60:.2f} min., LR:{current_lr:.6f}, "

          f"Epoch {epoch+1}/{args.n_epochs} | Train:{tr_loss:.5f}, Val:{vl_loss:.5f}")



    # Early stopping tracking (val loss based)

    if vl_loss < best_val_loss:

        best_val_loss = vl_loss

        epochs_no_improve = 0

        print(f" New best validation loss: {best_val_loss:.5f}")

    else:

        epochs_no_improve += 1

        print(f" No improvement for {epochs_no_improve}/{args.patience} epochs")



    # ------------------------------------------------------------------

    # CHECKPOINT EVALUATION

    # Epoch-axis mode: evaluate every N epochs

    # Fraction mode: evaluate only at the final epoch

    # ------------------------------------------------------------------

    is_final_epoch = (epoch + 1 == args.n_epochs) or (epochs_no_improve >= args.patience)

    is_checkpoint  = (args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0)



    if is_checkpoint or is_final_epoch:

        eval_mse = run_evaluation(epoch + 1, current_best_eval_mse=best_eval_mse)

        if eval_mse < best_eval_mse:

            best_eval_mse = eval_mse

        last_evaluated_epoch = epoch + 1



    # ---- #7 FIX: Save loss curve data every epoch in sweep mode ----

    if args.sweep_mode and args.master_csv_path:

        loss_csv_dir = os.path.dirname(args.master_csv_path)

        loss_csv_path = os.path.join(loss_csv_dir,

                                     f'loss_curve_{args.model_size}_{args.train_pct}pct.csv')

        write_header = not os.path.exists(loss_csv_path)

        with open(loss_csv_path, 'a', newline='') as _f:

            _w = csv.writer(_f)

            if write_header:

                _w.writerow(['epoch', 'train_loss', 'val_loss'])

            _w.writerow([epoch + 1, tr_loss, vl_loss])



    # Loss curve saved every epoch only in non-sweep mode

    if not args.sweep_mode:

        savepath_results_ = os.path.join(savepath_results, f'{dataset_name}')

        os.makedirs(savepath_results_, exist_ok=True)

        fig, ax = plt.subplots()

        ax.plot(train_losses, label='Train')

        ax.plot(val_losses, label='Val')

        ax.set_xlabel('Epoch')

        ax.set_ylabel('Loss')

        ax.legend()

        fig.savefig(os.path.join(savepath_results_,

                                 (f'loss_{args.model_choice}_{dataset_name}_'

                                  f'max_ar_{args.max_ar_order}_rank-lora{args.rank_lora_attn}_ftlevel{lev}'

                                  f'_lr{args.lr_level4}_wd{args.wd_level4}.png')))

        plt.close(fig)



    if epochs_no_improve >= args.patience:

        print(f"Early stopping triggered after {args.patience} epochs without improvement.")

        break



# Final evaluation only if the last epoch hasn't already been evaluated

last_epoch = len(train_losses)

if last_epoch != last_evaluated_epoch:

    run_evaluation(last_epoch, current_best_eval_mse=best_eval_mse)



del X_tr, y_tr, X_va, y_va

print(f"\n→ [{dataset_name}] Training complete.")

