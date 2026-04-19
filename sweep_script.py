# ==== Packages ====
import argparse
import subprocess
import numpy as np
import os
import shutil

def main():
    # ==== Command-line Arguments ====
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, required=True,
                        help="MORPH model size being used. (Ex: tiny, small, medium, large)")
    parser.add_argument("--dataset", type=str, required=True, help="PDE dataset path. (Ex: ./place_holder)")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Short name for the dataset. (Ex: fns-kf)")
    parser.add_argument("--train_size", nargs="*", type=int, required=True,
                        help="Training dataset percentage iteration range. (Ex: 25 50 75 100)")
    parser.add_argument("--epoch_range", nargs="*", type=int, required=True,
                        help="Epoch iteration range. (Ex: 10 50 100 200)")
    parser.add_argument("--rollout", type=int, required=True,
                        help="Rollout Horizon")
    args = parser.parse_args()

    # ==== Load dataset ====
    dataset_full_path = os.path.join("datasets", args.dataset)
    print(f"Loading dataset from {dataset_full_path}...")
    dataset = np.load(dataset_full_path, mmap_mode="r")
    total = dataset.shape[0]
    print(f"Total samples: {total}")

    # ==== Create top-level results folder ====
    sweep_results_dir = os.path.join("sweep_results", args.dataset_name, args.model_size)
    os.makedirs(sweep_results_dir, exist_ok=True)
    print(f"Sweep results will be saved to: {sweep_results_dir}/")

    # ==== Directories the finetune script writes to ====
    watched_dirs = [
        os.path.join("experiments", "results", "test"),
        os.path.join("models", args.dataset_name)
    ]

    # ==== Sweep ====
    dataset_dir = os.path.dirname(dataset_full_path)
    total_runs = len(args.train_size) * len(args.epoch_range)
    run = 0

    i = 0
    while i < len(args.train_size):
        pct = args.train_size[i]

        # ==== Slice and save subset ====
        n = max(1, int(pct / 100 * total))
        subset_filename = f"subset_{pct}pct.npy"
        subset_path = os.path.join("datasets", subset_filename)

        if not os.path.exists(subset_path):
            print(f"Creating subset: {subset_filename} (n={n})")
            np.save(subset_path, dataset[:n])
        else:
            print(f"Subset already exists: {subset_filename}")

        j = 0
        while j < len(args.epoch_range):
            epochs = args.epoch_range[j]
            run += 1
            print(f"\n{'='*60}")
            print(f"[Run {run}/{total_runs}] model={args.model_size} | train={pct}% (n={n}) | epochs={epochs}")
            print(f"{'='*60}\n")

            models_dataset_dir = os.path.join("models", args.dataset_name)

            if os.path.exists(models_dataset_dir):
                shutil.rmtree(models_dataset_dir)
            os.makedirs(models_dataset_dir, exist_ok=True)

            command = [
                "python", "scripts/finetune_MORPH_general.py",
                "--dataset", subset_filename,
                "--dataset_name", args.dataset_name,
                "--dataset_specs", "2", "1", "1", "128", "128",
                "--model_choice", "FM",
                "--model_size", args.model_size,
                "--ckpt_from", "FM",
                "--checkpoint", f"morph-{args.model_size}-FM-max_ar1_ep225.pth",
                "--ft_level1",
                "--parallel", "no",
                "--n_epochs", str(epochs),
                "--patience", "10",
                "--lr_scheduler",
                "--rollout_horizon", str(args.rollout)
            ]

            subprocess.run(command)

            # ==== Create iteration results folder ====
            run_folder = os.path.join(sweep_results_dir, f"{pct}percent",
                                      f"{args.model_size}_{pct}percent_{epochs}epochs")
            os.makedirs(run_folder, exist_ok=True)

            for d in watched_dirs:
                if os.path.exists(d):
                    # ==== Filter for the best model only ====
                    all_files = []
                    for root, _, filenames in os.walk(d):
                        for f in filenames:
                            all_files.append(os.path.join(root, f))

                    # ==== Isolate .pth files to find the "best" one ====
                    pth_files = [f for f in all_files if f.endswith('.pth')]

                    if pth_files:
                        best_model = sorted(pth_files, key=lambda x: os.path.getmtime(x))[-1]

                        # ==== Remove all other .pth files so they aren't copied ====
                        for pth in pth_files:
                            if pth != best_model:
                                os.remove(pth)

                    # ==== Copy the remaining files (Best Model + Metrics + Plots) ====
                    for root, _, filenames in os.walk(d):
                        for f in filenames:
                            src_path = os.path.join(root, f)
                            shutil.copy2(src_path, run_folder)

                    shutil.rmtree(d)
                    os.makedirs(d, exist_ok=True)
            j += 1
        i += 1

    print(f"\nSweep complete. {total_runs} runs finished. Results in {sweep_results_dir}/")

if __name__ == '__main__':
    main()
