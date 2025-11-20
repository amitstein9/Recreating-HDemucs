import re
import ast
import os
import matplotlib.pyplot as plt

# ---------------------------
# Part 1: Parse the Training Summary File
# ---------------------------
def parse_summary_file(filename):
    """
    Parses the merged training summary file to extract:
      - Train epochs, training loss, Reco (if present), Rrepo (if present)
      - Validation epochs, validation loss, and validation NSDR.
    Returns:
       train_epochs, train_loss, train_reco, train_rrepo,
       valid_epochs, valid_loss, valid_reco, valid_nsdr.
    """
    train_epochs, train_loss, train_reco, train_rrepo = [], [], [], []
    valid_epochs, valid_loss, valid_reco, valid_nsdr, valid_sdr = [], [], [], [], []
    
    # Regex patterns for Train Summary
    train_epoch_loss_pattern = re.compile(r"Train Summary \| Epoch (\d+) \| Loss=([\d\.]+)")
    train_reco_pattern = re.compile(r"Reco=([\d\.]+)")
    train_rrepo_pattern = re.compile(r"Rrepo=([\d\.]+)")
    
    # Regex patterns for Valid Summary
    valid_epoch_loss_nsdr_pattern = re.compile(r"Valid Summary \| Epoch (\d+) \| Loss=([\d\.]+).*Nsdr=([\d\.]+)")
    valid_reco_pattern = re.compile(r"Reco=([\d\.]+)")
    
    with open(filename, "r") as f:
        for line in f:
            if "Train Summary" in line:
                m = train_epoch_loss_pattern.search(line)
                if m:
                    epoch = int(m.group(1))
                    loss_val = float(m.group(2))
                    train_epochs.append(epoch)
                    train_loss.append(loss_val)
                    
                    reco_m = train_reco_pattern.search(line)
                    train_reco.append(float(reco_m.group(1)) if reco_m else None)
                    
                    rrepo_m = train_rrepo_pattern.search(line)
                    train_rrepo.append(float(rrepo_m.group(1)) if rrepo_m else None)
            elif "Valid Summary" in line:
                m = valid_epoch_loss_nsdr_pattern.search(line)
                if m:
                    valid_epochs.append(int(m.group(1)))
                    valid_loss.append(float(m.group(2)))
                    valid_nsdr.append(float(m.group(3)))
                    
                    reco_m = valid_reco_pattern.search(line)
                    valid_reco.append(float(reco_m.group(1)) if reco_m else None)
    return (train_epochs, train_loss, train_reco, train_rrepo,
            valid_epochs, valid_loss, valid_reco, valid_nsdr)

# ---------------------------
# Part 2: Parse Test Files
# ---------------------------
def remove_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def parse_test_file(filename):
    """
    Parses a test result file which is expected to contain a Python dictionary.
    Returns a dict of metrics.
    """
    with open(filename, "r") as f:
        content = f.read().strip()
    content = remove_ansi_codes(content)
    try:
        metrics = ast.literal_eval(content)
        if isinstance(metrics, dict):
            return metrics
        else:
            return {}
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return {}

# ---------------------------
# Part 3: Plot Training Curves and Save to Files
# ---------------------------
def plot_training_curves(train_epochs, train_loss, train_reco, train_rrepo,
                         valid_epochs, valid_loss, valid_reco, valid_nsdr):
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1) Plot Loss vs. Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, train_loss, label="Train Loss", color="blue", marker="o", markersize=4)
    plt.plot(valid_epochs, valid_loss, label="Validation Loss", color="orange", marker="o", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch (Training/Validation)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "loss.png"))
    plt.close()
    
    # 2) Plot Validation NSDR vs. Epoch
    plt.figure(figsize=(10, 6))
    plt.plot(valid_epochs, valid_nsdr, label="Validation NSDR", color="purple", marker="o", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("NSDR")
    plt.title("Validation NSDR vs. Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "nsdr.png"))
    plt.close()
    
    # 3) Plot Reco vs. Epoch (if available)
    if any(r is not None for r in train_reco):
        tr_data = [(ep, r) for ep, r in zip(train_epochs, train_reco) if r is not None]
        if tr_data:
            tr_ep, tr_reco = zip(*tr_data)
            plt.figure(figsize=(10, 6))
            plt.plot(tr_ep, tr_reco, label="Train Reco", color="green", marker="o", markersize=4)
            v_data = [(ep, r) for ep, r in zip(valid_epochs, valid_reco) if r is not None]
            if v_data:
                v_ep, v_reco = zip(*v_data)
                plt.plot(v_ep, v_reco, label="Validation Reco", color="red", marker="o", markersize=4)
            plt.xlabel("Epoch")
            plt.ylabel("Reco")
            plt.title("Reco vs. Epoch (Training/Validation)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, "reco.png"))
            plt.close()
    
    # 4) Plot Rrepo vs. Epoch (if available)
    if any(r is not None for r in train_rrepo):
        tr_data = [(ep, r) for ep, r in zip(train_epochs, train_rrepo) if r is not None]
        if tr_data:
            tr_ep, tr_rrepo = zip(*tr_data)
            plt.figure(figsize=(10, 6))
            plt.plot(tr_ep, tr_rrepo, label="Rrepo (Train)", color="brown", marker="o", markersize=4)
            plt.xlabel("Epoch")
            plt.ylabel("Rrepo")
            plt.title("Rrepo vs. Epoch (Training)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, "rrepo.png"))
            plt.close()

# ---------------------------
# Part 4: Plot Test Metrics and Save to Files
# ---------------------------
def plot_test_metrics(test_files):
    """
    For each instrument (drums, bass, other, vocals), plot test metrics.
    Expects test files in the given list (with path relative to "test_logs").
    Each test file is assumed to contain a Python dictionary with keys like:
      nsdr_drums, nsdr_med_drums, sdr_drums, sdr_med_drums, etc.
    Each instrument gets its own plot.
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    instruments = ["drums", "bass", "other", "vocals"]
    metric_types = ["nsdr", "sdr", "sir", "isr", "sar"]
    
    # Predefined colors and markers for overall metrics
    overall_colors = {
        "nsdr": "red",
        "sdr": "blue",
        "sir": "green",
        "isr": "purple",
        "sar": "orange"
    }
    overall_marker = "o"
    med_marker = "s"
    
    # Parse each test file (assume files are in "test_logs" folder)
    test_data = []  # list of (epoch, metrics_dict)
    for fname, epoch in sorted(test_files, key=lambda x: x[1]):
        full_path = os.path.join("test_logs", fname)
        metrics = parse_test_file(full_path)
        if metrics:
            test_data.append((epoch, metrics))
    
    if not test_data:
        print("No test data found. Ensure your test logs contain a Python dictionary representation of metrics.")
        return
    
    # For each instrument, create a plot.
    for inst in instruments:
        plt.figure(figsize=(10, 6))
        for metric in metric_types:
            overall_key = f"{metric}_{inst}"
            med_key = f"{metric}_med_{inst}"
            overall_vals = []
            med_vals = []
            epochs = []
            for epoch, m_dict in sorted(test_data, key=lambda x: x[0]):
                epochs.append(epoch)
                overall_vals.append(m_dict.get(overall_key, None))
                med_vals.append(m_dict.get(med_key, None))
            if any(v is not None for v in overall_vals):
                color = overall_colors.get(metric, None)
                plt.plot(epochs, overall_vals, label=f"{metric} (overall)", 
                         color=color, marker=overall_marker, linestyle="--", linewidth=2, markersize=4)
            if any(v is not None for v in med_vals):
                color = overall_colors.get(metric, None)
                plt.plot(epochs, med_vals, label=f"{metric} (med)", 
                         color=color, marker=med_marker, linestyle="-.", linewidth=2, markersize=4)
        plt.xlabel("Epoch (Test Checkpoint)")
        plt.ylabel("Metric Value")
        plt.title(f"Test Metrics for {inst.capitalize()}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        # Save the figure to the results folder.
        plt.savefig(os.path.join(results_dir, f"test_metrics_{inst}.png"))
        plt.close()

# ---------------------------
# Main Function
# ---------------------------
def main():
    # Define folder paths for input files.
    summary_file = os.path.join("train_logs", "hdemucs_summary.log")
    
    # Parse the training summary file.
    (train_epochs, train_loss, train_reco, train_rrepo,
     valid_epochs, valid_loss, valid_reco, valid_nsdr) = parse_summary_file(summary_file)
    
    # Plot training curves and save them to the results folder.
    plot_training_curves(train_epochs, train_loss, train_reco, train_rrepo,
                           valid_epochs, valid_loss, valid_reco, valid_nsdr)
    
    # Define test files along with their corresponding checkpoint epoch.
    test_files = [
        ("hdemucs_test1_759c9a58_127_epochs.out", 127),
        ("hdemucs_test2_44f697b5_218_epochs.out", 218),
        ("hdemucs_test3_53288a8c_after_282_epoches.out", 282),
        ("hdemucs_test4_53288a8c_after_390_epoches.out", 390)
    ]
    
    # Plot test metrics (each instrument gets its own graph) and save to results.
    plot_test_metrics(test_files)
    

if __name__ == "__main__":
    main()
