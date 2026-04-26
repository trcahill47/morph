# ==== Packages ====

import argparse

import subprocess

import numpy as np

import os



# ============================================================

# sweep_script.py

#

# Two modes:

#   --sweep_mode fraction  : outer loop over data %, fixed epochs.

#                            One .pth saved per data % (best val).

#   --sweep_mode epochs    : single data %, checkpoints every N epochs.

#                            One .pth saved per checkpoint.

#

# In both modes:

#   - All PNG generation is suppressed inside finetune (--sweep_mode flag)

#   - One master_metrics.csv per dataset accumulates all rows

#   - Models saved to sweep_results/{dataset}/models/{model_size}/{fraction|epoch}_sweep/

#   - Full dataset passed to finetune each run; --n_traj slices in memory (no subset files)

# ============================================================



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_size", type=str, required=True,

                        help="MORPH model size: Ti, S, M, L")

    parser.add_argument("--dataset", type=str, required=True,

                        help="Dataset filename relative to datasets/. e.g. fns-kf/solution_0.npy")

    parser.add_argument("--dataset_name", type=str, required=True,

                        help="Short name for the dataset. e.g. fns-kf")

    parser.add_argument("--sweep_mode", type=str, required=True,

                        choices=["fraction", "epochs"],

                        help="fraction: vary data pct, fixed epochs. "

                             "epochs: fixed data pct, vary checkpoint intervals.")



    # Fraction-mode arguments

    parser.add_argument("--train_sizes", nargs="*", type=int, default=[],

                        help="[fraction mode] List of training data percentages. e.g. 5 10 15 20")

    parser.add_argument("--n_epochs", type=int, default=20,

                        help="[fraction mode] Total epochs to train per data-pct run.")



    # Epoch-mode arguments

    parser.add_argument("--train_pct", type=int, default=15,

                        help="[epoch mode] Fixed data percentage to use.")

    parser.add_argument("--max_epochs", type=int, default=25,

                        help="[epoch mode] Max epochs to train.")

    parser.add_argument("--checkpoint_every", type=int, default=5,

                        help="[epoch mode] Evaluate and save .pth every N epochs.")



    # Shared arguments

    parser.add_argument("--rollout", type=int, required=True,

                        help="Rollout horizon for evaluation.")

    parser.add_argument("--max_ar_order", type=int, default=1,

                        help="Max autoregressive order. Use 16 for model size L.")

    parser.add_argument("--patience", type=int, default=10,

                        help="Early stopping patience (passed to finetune).")

    parser.add_argument("--dataset_specs", nargs=5, type=int,

                        metavar='N',

                        default=[2, 1, 1, 128, 128],

                        help="Dataset specs: fields components depth height width. "

                             "Default 2 1 1 128 128 for fns-kf. Override for SWE.")



    args = parser.parse_args()



    # ----- Paths -----

    dataset_full_path = os.path.join("datasets", args.dataset)

    print(f"Loading dataset from {dataset_full_path}...")

    dataset = np.load(dataset_full_path, mmap_mode="r")

    total_samples = dataset.shape[0]

    print(f"Total samples in dataset: {total_samples}")



    sweep_root     = os.path.join("sweep_results", args.dataset_name)

    master_csv     = os.path.join(sweep_root, "master_metrics.csv")

    models_root    = os.path.join(sweep_root, "models")

    os.makedirs(sweep_root,  exist_ok=True)

    os.makedirs(models_root, exist_ok=True)

    print(f"Master CSV will be written to: {master_csv}")



    # ----- Helper: compute n_traj from percentage -----

    def traj_for_pct(pct):

        n = max(1, int(pct / 100 * total_samples))

        print(f"  Data {pct}% → {n} trajectories")

        return n



    # ----- Helper: build the base finetune command -----

    def base_command(n_traj, pct, n_ep, checkpoint_every=0):

        cmd = [

            "python", "scripts/finetune_MORPH_general.py",

            "--dataset",       args.dataset,

            "--dataset_name",  args.dataset_name,

            "--dataset_specs", str(args.dataset_specs[0]), str(args.dataset_specs[1]),

                               str(args.dataset_specs[2]), str(args.dataset_specs[3]),

                               str(args.dataset_specs[4]),

            "--model_choice",  "FM",

            "--model_size",    args.model_size,

            "--ckpt_from",     "FM",

            "--download_model",

            "--ft_level1",

            "--parallel",      "no",

            "--n_epochs",      str(n_ep),

            "--n_traj",        str(n_traj),

            "--patience",      str(args.patience),

            "--lr_scheduler",

            "--rollout_horizon", str(args.rollout),

            "--max_ar_order",  str(args.max_ar_order),

            "--sweep_mode",                        # suppress all PNGs

            "--train_pct",     str(pct),           # for CSV logging

            "--master_csv_path", master_csv,       # single master CSV

        ]

        if checkpoint_every > 0:

            cmd += ["--checkpoint_every", str(checkpoint_every)]

        return cmd



    # ==================================================================

    # FRACTION MODE

    # Outer loop: data percentages

    # Each run trains to n_epochs, saves best .pth to models/fraction_sweep/

    # ==================================================================

    if args.sweep_mode == "fraction":

        if not args.train_sizes:

            raise ValueError("--train_sizes is required for fraction mode")



        best_model_dir = os.path.join(models_root, args.model_size, "fraction_sweep")

        os.makedirs(best_model_dir, exist_ok=True)



        print(f"\n{'='*60}")

        print(f"FRACTION SWEEP | model={args.model_size} | "

              f"epochs={args.n_epochs} | sizes={args.train_sizes}")

        print(f"{'='*60}\n")



        for i, pct in enumerate(args.train_sizes):

            print(f"\n[{i+1}/{len(args.train_sizes)}] Data={pct}% | Epochs={args.n_epochs}")



            n_traj = traj_for_pct(pct)

            cmd = base_command(n_traj, pct, args.n_epochs, checkpoint_every=0)

            cmd += ["--best_model_dir", best_model_dir]



            print(f"  Running: {' '.join(cmd)}\n")

            subprocess.run(cmd, check=True)



            print(f"  Done: {pct}% data run complete.")



        print(f"\nFraction sweep complete. {len(args.train_sizes)} runs finished.")

        print(f"Models saved to: {best_model_dir}/")

        print(f"Metrics in:      {master_csv}")



    # ==================================================================

    # EPOCH MODE

    # Single data %, train to max_epochs, checkpoint every N epochs

    # Saves one .pth per checkpoint to models/epoch_sweep/

    # ==================================================================

    elif args.sweep_mode == "epochs":

        best_model_dir = os.path.join(models_root, args.model_size, "epoch_sweep")

        os.makedirs(best_model_dir, exist_ok=True)



        pct = args.train_pct

        print(f"\n{'='*60}")

        print(f"EPOCH SWEEP | model={args.model_size} | data={pct}% | "

              f"max_epochs={args.max_epochs} | checkpoint_every={args.checkpoint_every}")

        print(f"{'='*60}\n")



        n_traj = traj_for_pct(pct)

        cmd = base_command(n_traj, pct, args.max_epochs,

                           checkpoint_every=args.checkpoint_every)

        cmd += ["--best_model_dir", best_model_dir]



        print(f"Running: {' '.join(cmd)}\n")

        subprocess.run(cmd, check=True)



        n_checkpoints = args.max_epochs // args.checkpoint_every

        print(f"\nEpoch sweep complete. Up to {n_checkpoints} checkpoints saved.")

        print(f"Models saved to: {best_model_dir}/")

        print(f"Metrics in:      {master_csv}")





if __name__ == '__main__':

    main()
