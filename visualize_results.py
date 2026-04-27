"""
visualize_results.py
====================
Two independent modes:

  python visualize_results.py plots \
      --csv sweep_results/fns-kf/master_metrics.csv \
      --dataset fns-kf \
      --out_dir sweep_results/fns-kf/plots

  python visualize_results.py visuals \
      --pth path/to/model.pth \
      --dataset datasets/fns-kf/solution_0.npy \
      --dataset_name fns-kf \
      --model_size M \
      --rollout_horizon 20 \
      --out_dir sweep_results/fns-kf/final_visuals

Part A (plots): reads master_metrics.csv and generates
  - Line plot: MSE vs epochs (per model size)
  - Line plot: SSIM vs epochs (per model size)
  - Line plot: MSE vs data % (per model size)
  - Line plot: SSIM vs data % (per model size)
  - Heatmap: MSE over (data %, epoch) grid
  - Heatmap: SSIM over (data %, epoch) grid
  - Pareto scatter: MSE vs n_traj colored by epoch

Part B (visuals): loads a .pth and generates
  - Side-by-side single-step PNGs (input/target/pred/residual)
  - Rollout PNGs
  - Dual-axis MSE+SSIM plot
  - MP4 animation (target | prediction | residual)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


# ============================================================
# PART A — PLOTS FROM MASTER CSV
# ============================================================

def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Parse per_step lists stored as strings
    import ast
    if 'per_step_mse' in df.columns:
        df['per_step_mse']  = df['per_step_mse'].apply(ast.literal_eval)
    if 'per_step_ssim' in df.columns:
        df['per_step_ssim'] = df['per_step_ssim'].apply(ast.literal_eval)
    return df


def plot_metric_vs_epoch(df, metric, out_dir, dataset_filter=None, model_sizes=None):
    """Line plot of `metric` vs epoch, one line per model size."""
    sub = df.copy()
    if dataset_filter:
        sub = sub[sub['dataset'] == dataset_filter]
    if model_sizes:
        sub = sub[sub['model_size'].isin(model_sizes)]

    fig, ax = plt.subplots(figsize=(9, 6))
    for ms, grp in sub.groupby('model_size'):
        grp_sorted = grp.sort_values('epoch')
        ax.plot(grp_sorted['epoch'], grp_sorted[metric], marker='o', label=f'Model {ms}')

    ax.set_xlabel('Training Epochs', fontsize=13)
    ax.set_ylabel(metric.upper(), fontsize=13)
    ax.set_title(f'{metric.upper()} vs Epochs — {dataset_filter or "all datasets"}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f'{metric}_vs_epochs.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_metric_vs_datapct(df, metric, out_dir, dataset_filter=None, model_sizes=None,
                            fixed_epoch=None):
    """
    Line plot of `metric` vs train_pct.
    If fixed_epoch is given, filter to that epoch only.
    Otherwise uses the max epoch per (model_size, train_pct) group.
    """
    sub = df.copy()
    if dataset_filter:
        sub = sub[sub['dataset'] == dataset_filter]
    if model_sizes:
        sub = sub[sub['model_size'].isin(model_sizes)]

    if fixed_epoch is not None:
        sub = sub[sub['epoch'] == fixed_epoch]
    else:
        # Take the best (lowest MSE) row per group as the "final" result
        idx = sub.groupby(['model_size', 'train_pct'])['mse'].idxmin()
        sub = sub.loc[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    for ms, grp in sub.groupby('model_size'):
        grp_sorted = grp.sort_values('train_pct')
        ax.plot(grp_sorted['train_pct'], grp_sorted[metric], marker='o', label=f'Model {ms}')

    ax.set_xlabel('Training Data %', fontsize=13)
    ax.set_ylabel(metric.upper(), fontsize=13)
    ax.set_title(f'{metric.upper()} vs Data % — {dataset_filter or "all datasets"}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f'{metric}_vs_datapct.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_heatmap(df, metric, out_dir, dataset_filter=None, model_size=None):
    """
    Heatmap of `metric` over (train_pct x epoch).
    One heatmap per model size if model_size is None.
    """
    sub = df.copy()
    if dataset_filter:
        sub = sub[sub['dataset'] == dataset_filter]

    sizes = [model_size] if model_size else sub['model_size'].unique()

    for ms in sizes:
        grp = sub[sub['model_size'] == ms]
        if grp.empty:
            continue

        pivot = grp.pivot_table(index='train_pct', columns='epoch',
                                values=metric, aggfunc='mean')
        pivot = pivot.sort_index(ascending=False)  # high data % at top

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.9),
                                        max(5, len(pivot.index) * 0.8)))
        sns.heatmap(pivot, ax=ax, annot=True, fmt='.4f',
                    cmap='RdYlGn_r' if metric in ('mse','rmse','nrmse') else 'RdYlGn',
                    linewidths=0.5, cbar_kws={'label': metric.upper()})
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Train Data %', fontsize=12)
        ax.set_title(f'{metric.upper()} Heatmap — {ms} | {dataset_filter or "all"}', fontsize=13)
        fig.tight_layout()
        path = os.path.join(out_dir, f'heatmap_{metric}_{ms}.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_pareto(df, out_dir, dataset_filter=None, model_sizes=None):
    """
    Scatter of MSE vs n_traj, colored by epoch.
    Shows the accuracy vs data-size tradeoff (pareto-style).
    """
    sub = df.copy()
    if dataset_filter:
        sub = sub[sub['dataset'] == dataset_filter]
    if model_sizes:
        sub = sub[sub['model_size'].isin(model_sizes)]

    fig, ax = plt.subplots(figsize=(10, 7))
    for ms, grp in sub.groupby('model_size'):
        sc = ax.scatter(grp['n_traj'], grp['mse'],
                        c=grp['epoch'], cmap='plasma',
                        s=80, alpha=0.8, label=f'Model {ms}',
                        marker='o' if ms == 'M' else '^')
    plt.colorbar(sc, ax=ax, label='Epoch')
    ax.set_xlabel('Number of Training Trajectories (n_traj)', fontsize=13)
    ax.set_ylabel('MSE', fontsize=13)
    ax.set_title(f'Accuracy vs Data Size — {dataset_filter or "all datasets"}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, 'pareto_mse_vs_ntraj.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_rollout_curves(df, out_dir, dataset_filter=None, model_sizes=None,
                         train_pct=None, epoch=None):
    """
    Per-timestep MSE and SSIM curves for selected runs.
    Each row in the CSV stores the full list.
    """
    import ast
    sub = df.copy()
    if dataset_filter:
        sub = sub[sub['dataset'] == dataset_filter]
    if model_sizes:
        sub = sub[sub['model_size'].isin(model_sizes)]
    if train_pct is not None:
        sub = sub[sub['train_pct'] == train_pct]
    if epoch is not None:
        sub = sub[sub['epoch'] == epoch]

    if sub.empty:
        print("  No rows matched rollout_curves filter — skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    for _, row in sub.iterrows():
        label = f"{row['model_size']} | {row['train_pct']}% | ep{row['epoch']}"
        steps = range(1, len(row['per_step_mse']) + 1)
        ax1.plot(steps, row['per_step_mse'], marker='o', label=label)
        ax2.plot(steps, row['per_step_ssim'], marker='s', label=label)

    ax1.set_ylabel('MSE per Rollout Step', fontsize=12)
    ax1.set_title('Rollout MSE over Time', fontsize=13)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Rollout Timestep', fontsize=12)
    ax2.set_ylabel('SSIM per Rollout Step', fontsize=12)
    ax2.set_title('Rollout SSIM over Time', fontsize=13)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'Per-Timestep Rollout — {dataset_filter or "all"}', fontsize=14)
    fig.tight_layout()
    path = os.path.join(out_dir, 'rollout_curves.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_cross_dataset_comparison(csv_paths, dataset_names, metric, out_dir, model_sizes=None):
    """
    Overlay metric curves from two datasets on the same plot.
    csv_paths: list of two CSV file paths [fns-kf csv, swe csv]
    dataset_names: list of two short names ['fns-kf', 'swe']
    """
    fig, (ax_epoch, ax_pct) = plt.subplots(1, 2, figsize=(16, 6))

    linestyles = ['-', '--']
    for csv_path, ds_name, ls in zip(csv_paths, dataset_names, linestyles):
        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found, skipping {ds_name}")
            continue
        df = load_csv(csv_path)
        if model_sizes:
            df = df[df['model_size'].isin(model_sizes)]

        for ms, grp in df.groupby('model_size'):
            grp_sorted = grp.sort_values('epoch')
            ax_epoch.plot(grp_sorted['epoch'], grp_sorted[metric],
                          marker='o', linestyle=ls,
                          label=f'{ds_name} | {ms}')

        idx = df.groupby(['model_size', 'train_pct'])['mse'].idxmin()
        best = df.loc[idx]
        for ms, grp in best.groupby('model_size'):
            grp_sorted = grp.sort_values('train_pct')
            ax_pct.plot(grp_sorted['train_pct'], grp_sorted[metric],
                        marker='o', linestyle=ls,
                        label=f'{ds_name} | {ms}')

    ax_epoch.set_xlabel('Training Epochs', fontsize=12)
    ax_epoch.set_ylabel(metric.upper(), fontsize=12)
    ax_epoch.set_title(f'{metric.upper()} vs Epochs — Cross-Dataset', fontsize=13)
    ax_epoch.legend(fontsize=8)
    ax_epoch.grid(True, alpha=0.3)

    ax_pct.set_xlabel('Training Data %', fontsize=12)
    ax_pct.set_ylabel(metric.upper(), fontsize=12)
    ax_pct.set_title(f'{metric.upper()} vs Data % — Cross-Dataset', fontsize=13)
    ax_pct.legend(fontsize=8)
    ax_pct.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f'cross_dataset_{metric}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")



def run_plots(args):
    print(f"\n=== PART A: Generating plots from {args.csv} ===\n")
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_csv(args.csv)
    print(f"  Loaded {len(df)} rows. Columns: {list(df.columns)}")

    model_sizes = args.model_sizes if args.model_sizes else None
    ds = args.dataset if args.dataset else None

    plot_metric_vs_epoch(df,    'mse',            args.out_dir, ds, model_sizes)
    plot_metric_vs_epoch(df,    'mean_rollout_ssim', args.out_dir, ds, model_sizes)
    plot_metric_vs_epoch(df,    'nrmse',           args.out_dir, ds, model_sizes)
    plot_metric_vs_datapct(df,  'mse',             args.out_dir, ds, model_sizes)
    plot_metric_vs_datapct(df,  'mean_rollout_ssim', args.out_dir, ds, model_sizes)
    plot_heatmap(df,            'mse',             args.out_dir, ds)
    plot_heatmap(df,            'mean_rollout_ssim', args.out_dir, ds)
    plot_pareto(df,                                args.out_dir, ds, model_sizes)
    plot_rollout_curves(df,                        args.out_dir, ds, model_sizes)

    print(f"\nAll plots saved to {args.out_dir}/")


# ============================================================
# PART B — VISUALS FROM A .PTH FILE
# Generates the same visual output the finetune script produces,
# but on demand, for any saved checkpoint.
# ============================================================

def run_visuals(vis_args):
    print(f"\n=== PART B: Generating visuals from {vis_args.pth} ===\n")

    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    from skimage.metrics import structural_similarity as ssim
    import imageio.v2 as imageio
    from PIL import Image

    # ---- Add project root to path ----
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    sys.path.append(project_root)

    from src.utils.device_manager import DeviceManager
    from src.utils.vit_conv_xatt_axialatt2 import ViT3DRegression
    from src.utils.metrics_3d import Metrics3DCalculator
    from src.utils.visualize_predictions_3d_full import Visualize3DPredictions
    from src.utils.visualize_rollouts_3d_full import Visualize3DRolloutPredictions
    from src.utils.data_preparation_fast import FastARDataPreparer
    from src.utils.normalization import RevIN
    from src.utils.uptf7 import UPTF7

    MORPH_MODELS = {
        'Ti': [8, 256,  4,  4, 1024],
        'S' : [8, 512,  8,  4, 2048],
        'M' : [8, 768, 12,  8, 3072],
        'L' : [8, 1024,16, 16, 4096]
    }

    os.makedirs(vis_args.out_dir, exist_ok=True)
    devices = DeviceManager.list_devices()
    device  = devices[0] if devices else 'cpu'

    ms = vis_args.model_size
    filters, dim, heads, depth, mlp_dim = MORPH_MODELS[ms]

    # Infer max_ar_order from checkpoint filename if not specified
    max_ar = vis_args.max_ar_order
    if max_ar is None:
        # Try to parse from filename e.g. "morph-L-FM-max_ar16_..."
        import re
        m = re.search(r'max_ar(\d+)', os.path.basename(vis_args.pth))
        max_ar = int(m.group(1)) if m else 1
    print(f"  Using max_ar_order={max_ar}")

    ft_model = ViT3DRegression(
        patch_size=8, dim=dim, depth=depth,
        heads=heads, heads_xa=32, mlp_dim=mlp_dim,
        max_components=3, conv_filter=filters,
        max_ar=max_ar,
        max_patches=4096, max_fields=3,
        dropout=0.1, emb_dropout=0.1,
        lora_r_attn=16, lora_r_mlp=12,
        lora_alpha=None, lora_p=0.05
    ).to(device)

    ckpt = torch.load(vis_args.pth, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]
    target = ft_model.module if isinstance(ft_model, nn.DataParallel) else ft_model
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    target.load_state_dict(state_dict, strict=False)
    ft_model.eval()
    print(f"  Loaded checkpoint from {vis_args.pth}")

    epoch_label = ckpt.get("epoch", "unk")

    # ---- Load and prep dataset ----
    loadpath_muvar = os.path.join(project_root, 'data')
    dataset = np.load(vis_args.dataset)
    num_samples = dataset.shape[0]
    traj_len    = dataset.shape[1]

    dataset_uptf7 = UPTF7(
        dataset=dataset, num_samples=num_samples, traj_len=traj_len,
        fields=vis_args.dataset_specs[0],
        components=vis_args.dataset_specs[1],
        image_depth=vis_args.dataset_specs[2],
        image_height=vis_args.dataset_specs[3],
        image_width=vis_args.dataset_specs[4]
    ).transform()

    dataset_name = vis_args.dataset_name
    norm_prefix  = f'norm_{dataset_name}'
    revin = RevIN(loadpath_muvar)
    revin.compute_stats(dataset_uptf7, prefix=norm_prefix)
    dataset_norm = revin.normalize(dataset_uptf7, prefix=norm_prefix)
    dataset_norm_rs = dataset_norm.transpose(0,1,4,5,6,3,2)

    from sklearn.model_selection import train_test_split
    _, tmp        = train_test_split(dataset_norm_rs, test_size=0.2, random_state=42, shuffle=True)
    _, test_data  = train_test_split(tmp, test_size=0.5, random_state=42, shuffle=True)

    sim     = test_data[0]
    sim_rs  = np.transpose(sim, (0, 5, 4, 1, 2, 3)).astype(np.float32)
    sim_tensor = torch.from_numpy(sim_rs).unsqueeze(0)
    field_names = [f'field-{fi}' for fi in range(1, sim_tensor.shape[2] + 1)]

    rollout = vis_args.rollout_horizon
    prefix  = f'{ms}_{dataset_name}_ep{epoch_label}'

    # ---- Single-step PNGs ----
    print(f"  Generating single-step PNGs (0 to {rollout-1})...")
    viz = Visualize3DPredictions(ft_model, sim_tensor, device)
    for t in range(rollout):
        viz.visualize_predictions(
            time_step=t, component=0, slice_dim='d',
            save_path=vis_args.out_dir,
            figname=f'single_step_{prefix}_t{t}.png'
        )

    # ---- Rollout PNGs ----
    print(f"  Generating rollout PNGs...")
    viz_roll = Visualize3DRolloutPredictions(
        model=ft_model, test_dataset=sim_tensor,
        device=device, field_names=field_names,
        component_names=["d","h","w"]
    )
    for fi in range(sim_tensor.shape[2]):
        viz_roll.visualize_rollout(
            start_step=0, num_steps=rollout, field=fi,
            component=0, slice_dim='d',
            save_path=vis_args.out_dir,
            figname=f'rollout_{prefix}_field{fi}.png'
        )

    # ---- Per-timestep MSE + SSIM ----
    print(f"  Computing per-timestep rollout metrics...")
    per_step_mse  = []
    per_step_ssim = []

    with torch.no_grad():
        sim_input = sim_tensor[:, 0:1].to(device)
        for t in range(rollout):
            _, _, pred = ft_model(sim_input)
            target_t   = sim_tensor[:, t + 1].to(device)
            step_mse   = F.mse_loss(pred.squeeze(1), target_t).item()
            per_step_mse.append(step_mse)

            pred_np  = pred.squeeze().cpu().numpy()
            targ_np  = target_t.squeeze().cpu().numpy()
            drange   = targ_np.max() - targ_np.min()
            try:
                step_ssim = ssim(targ_np, pred_np, data_range=drange, channel_axis=0)
            except ValueError:
                step_ssim = ssim(targ_np, pred_np, data_range=drange)
            per_step_ssim.append(step_ssim)

            sim_input = pred.unsqueeze(1)

    # ---- Dual-axis plot ----
    fig, ax1 = plt.subplots(figsize=(10, 6))
    c_mse  = 'tab:red'
    c_ssim = 'tab:blue'
    ax1.set_xlabel('Rollout Timestep')
    ax1.set_ylabel('MSE', color=c_mse, fontsize=12)
    ax1.plot(range(1, rollout + 1), per_step_mse, marker='o', color=c_mse, lw=2)
    ax1.tick_params(axis='y', labelcolor=c_mse)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.set_ylabel('SSIM', color=c_ssim, fontsize=12)
    ax2.plot(range(1, rollout + 1), per_step_ssim, marker='s', color=c_ssim, lw=2)
    ax2.tick_params(axis='y', labelcolor=c_ssim)
    ax2.set_ylim(0, 1.05)
    plt.title(f'Rollout Stability: {dataset_name} | {ms} | ep{epoch_label}', fontsize=14)
    fig.tight_layout()
    dual_path = os.path.join(vis_args.out_dir, f'dual_metrics_{prefix}.png')
    fig.savefig(dual_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {dual_path}")

    # ---- Animation ----
    # Reads the single-step PNGs and tiles target | prediction | residual
    print(f"  Building rollout animation...")
    frames = []
    missing_frames = []
    for t in range(rollout):
        png_path = os.path.join(vis_args.out_dir, f'single_step_{prefix}_t{t}.png')
        if not os.path.exists(png_path):
            missing_frames.append(t)
            continue
        img = Image.open(png_path)
        w, h   = img.size
        col_w  = w // 5   # 5 columns: input, target, pred, mse(tgt), mse(inp)
        # Use full height — single-field datasets only have one row
        # For multi-field datasets the title bar takes ~15% so use 85% of height
        row_h  = h

        target_crop = img.crop((1 * col_w, 0, 2 * col_w, row_h))
        pred_crop   = img.crop((2 * col_w, 0, 3 * col_w, row_h))
        mse_crop    = img.crop((3 * col_w, 0, 4 * col_w, row_h))

        frame = Image.new("RGB", (col_w * 3, row_h))
        frame.paste(target_crop, (0,       0))
        frame.paste(pred_crop,   (col_w,   0))
        frame.paste(mse_crop,    (2*col_w, 0))
        frames.append(frame)

    if missing_frames:
        raise FileNotFoundError(
            f"Animation aborted: {len(missing_frames)} single-step PNG(s) missing "
            f"for timesteps {missing_frames}. "
            f"Re-run the visuals command to regenerate all PNGs first."
        )

    if frames:
        anim_path = os.path.join(vis_args.out_dir, f'animation_{prefix}.mp4')
        imageio.mimsave(anim_path, [np.array(f) for f in frames], format='ffmpeg', fps=vis_args.fps)
        print(f"  Saved animation: {anim_path}")
    else:
        print("  No frames found for animation.")

    print(f"\nAll visuals saved to {vis_args.out_dir}/")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualization toolkit for MORPH sweep results.")
    subparsers = parser.add_subparsers(dest='command')

    # ---- plots subcommand ----
    p_plots = subparsers.add_parser('plots', help='Generate plots from master_metrics.csv')
    p_plots.add_argument('--csv',          type=str, required=True,
                         help='Path to master_metrics.csv')
    p_plots.add_argument('--out_dir',      type=str, required=True,
                         help='Output directory for plots')
    p_plots.add_argument('--dataset',      type=str, default=None,
                         help='Filter to a specific dataset name (e.g. fns-kf)')
    p_plots.add_argument('--model_sizes',  nargs='*', default=None,
                         help='Filter to specific model sizes (e.g. M L)')

    # ---- visuals subcommand ----
    p_vis = subparsers.add_parser('visuals', help='Generate visuals from a saved .pth checkpoint')
    p_vis.add_argument('--pth',            type=str, required=True,
                       help='Path to the .pth checkpoint')
    p_vis.add_argument('--dataset',        type=str, required=True,
                       help='Path to the dataset .npy file')
    p_vis.add_argument('--dataset_name',   type=str, required=True,
                       help='Short name for the dataset (e.g. fns-kf)')
    p_vis.add_argument('--model_size',     type=str, required=True,
                       choices=['Ti', 'S', 'M', 'L'])
    p_vis.add_argument('--out_dir',        type=str, required=True,
                       help='Output directory for visual files')
    p_vis.add_argument('--rollout_horizon',type=int, default=20,
                       help='Number of rollout steps to visualize')
    p_vis.add_argument('--max_ar_order',   type=int, default=None,
                       help='Max AR order (auto-detected from filename if omitted)')
    p_vis.add_argument('--fps',            type=int, default=3,
                       help='Frames per second for the animation')
    p_vis.add_argument('--dataset_specs',  nargs=5, type=int, metavar='N',
                       default=[2, 1, 1, 128, 128],
                       help='Dataset specs: fields components depth height width. '
                            'Default 2 1 1 128 128 for fns-kf. Use 1 1 1 128 128 for SWE.')

    # ---- compare subcommand ----
    p_compare = subparsers.add_parser('compare', help='Cross-dataset comparison plots')
    p_compare.add_argument('--csv1',          type=str, required=True,
                           help='Path to master_metrics.csv for dataset 1 (e.g. fns-kf)')
    p_compare.add_argument('--csv2',          type=str, required=True,
                           help='Path to master_metrics.csv for dataset 2 (e.g. swe)')
    p_compare.add_argument('--name1',         type=str, required=True,
                           help='Short name for dataset 1 (e.g. fns-kf)')
    p_compare.add_argument('--name2',         type=str, required=True,
                           help='Short name for dataset 2 (e.g. swe)')
    p_compare.add_argument('--out_dir',       type=str, required=True,
                           help='Output directory for comparison plots')
    p_compare.add_argument('--model_sizes',   nargs='*', default=None,
                           help='Filter to specific model sizes (e.g. M L)')
    p_compare.add_argument('--metrics',       nargs='*',
                           default=['mse', 'mean_rollout_ssim', 'nrmse'],
                           help='Metrics to compare')

    args = parser.parse_args()

    if args.command == 'plots':
        run_plots(args)
    elif args.command == 'visuals':
        run_visuals(args)
    elif args.command == 'compare':
        os.makedirs(args.out_dir, exist_ok=True)
        for metric in args.metrics:
            plot_cross_dataset_comparison(
                csv_paths=[args.csv1, args.csv2],
                dataset_names=[args.name1, args.name2],
                metric=metric,
                out_dir=args.out_dir,
                model_sizes=args.model_sizes
            )
        print(f"\nAll comparison plots saved to {args.out_dir}/")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
