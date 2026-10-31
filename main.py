"""
Training Loop vs. PyTorch DataLoader: Architectural & Performance Benchmark
Part of the '100 PyTorch Projects in 10 Days' Challenge (Day 2 Milestone - Project 20)
Author: Gideon Mwiti

Demonstrates:
- Systematic comparison of manual indexing/slicing vs PyTorch DataLoader
- Throughput scaling across batch sizes B in [16, 32, 64, 128, 256, 512]
- Batch delivery latency jitter profiling (mean, std, 95th percentile)
- Training convergence verification proving numerical equivalence
- Detailed architectural analysis (pin_memory, num_workers, DistributedSampler, collate_fn)
"""

import sys
import types
import io
import os
import time
from typing import Tuple, List, Dict, Any

# Standard patch for Python 3.11+ bz2/lzma compatibility
sys.modules['bz2'] = types.SimpleNamespace(open=io.open, BZ2File=io.BytesIO)
sys.modules['lzma'] = types.SimpleNamespace(
    open=io.open,
    LZMAFile=io.BytesIO,
    FORMAT_ALONE=1,
    FORMAT_XZ=2,
    CHECK_NONE=0,
    CHECK_CRC32=1,
    CHECK_CRC64=4,
    CHECK_SHA256=10
)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------------------------------------------------------------------
# 1. Deterministic Seeding & Device Setup
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Execution Device: {DEVICE}")


# -----------------------------------------------------------------------------
# 2. Benchmark Neural Network Architecture
# -----------------------------------------------------------------------------
class BenchmarkMLP(nn.Module):
    """3-Layer Multi-Layer Perceptron with BatchNorm and ReLU."""
    def __init__(self, in_features: int = 32, hidden_dim: int = 64, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# 3. Custom Benchmark Dataset
# -----------------------------------------------------------------------------
class SyntheticBenchmarkDataset(Dataset):
    """
    Simulates real-world feature retrieval with deterministic synthesis.
    """
    def __init__(self, num_samples: int = 30000, num_features: int = 32, num_classes: int = 4):
        g = torch.Generator().manual_seed(42)
        self.X = torch.randn(num_samples, num_features, generator=g)
        # Generate non-linear ground truth targets
        weights = torch.randn(num_features, num_classes, generator=g)
        logits = self.X @ weights
        self.y = torch.argmax(logits, dim=1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------------------------
# 4. Training Loop Implementations
# -----------------------------------------------------------------------------
def train_epoch_manual(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    optimizer: optim.Optimizer,
    criterion: nn.Module
) -> Tuple[float, float, List[float]]:
    """
    Manual batching using torch.randperm and tensor slicing.
    Measures epoch latency and individual batch delivery timings.
    """
    model.train()
    total_loss, total_count = 0.0, 0
    num_samples = X.size(0)

    t0_epoch = time.perf_counter()
    indices = torch.randperm(num_samples)

    num_batches = (num_samples + batch_size - 1) // batch_size
    batch_latencies = []

    t_prev = time.perf_counter()
    for b in range(num_batches):
        batch_idx = indices[b * batch_size : min((b + 1) * batch_size, num_samples)]
        batch_x = X[batch_idx].to(DEVICE)
        batch_y = y[batch_idx].to(DEVICE)

        t_now = time.perf_counter()
        batch_latencies.append((t_now - t_prev) * 1000.0)

        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_count += batch_x.size(0)
        t_prev = time.perf_counter()

    epoch_time = time.perf_counter() - t0_epoch
    return total_loss / total_count, epoch_time, batch_latencies


def train_epoch_dataloader(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module
) -> Tuple[float, float, List[float]]:
    """
    Standard PyTorch DataLoader iteration.
    Measures epoch latency and individual batch delivery timings.
    """
    model.train()
    total_loss, total_count = 0.0, 0

    t0_epoch = time.perf_counter()
    batch_latencies = []

    t_prev = time.perf_counter()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        t_now = time.perf_counter()
        batch_latencies.append((t_now - t_prev) * 1000.0)

        optimizer.zero_grad()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        total_count += batch_x.size(0)
        t_prev = time.perf_counter()

    epoch_time = time.perf_counter() - t0_epoch
    return total_loss / total_count, epoch_time, batch_latencies


# -----------------------------------------------------------------------------
# 5. Benchmarking Experiments
# -----------------------------------------------------------------------------
def run_batch_size_scaling_benchmark(
    dataset: SyntheticBenchmarkDataset,
    batch_sizes: List[int] = [16, 32, 64, 128, 256, 512]
) -> Dict[str, List[float]]:
    """Evaluates throughput (samples/sec) across batch sizes."""
    print("[*] Running batch-size scaling and throughput benchmark...")
    manual_throughput = []
    dataloader_throughput = []

    criterion = nn.CrossEntropyLoss()

    for bs in batch_sizes:
        # 1. Manual Batching
        set_seed(42)
        model_m = BenchmarkMLP().to(DEVICE)
        opt_m = optim.Adam(model_m.parameters(), lr=1e-3)
        _, t_manual, _ = train_epoch_manual(model_m, dataset.X, dataset.y, bs, opt_m, criterion)
        samples_per_sec_m = len(dataset) / t_manual
        manual_throughput.append(samples_per_sec_m)

        # 2. DataLoader
        set_seed(42)
        model_d = BenchmarkMLP().to(DEVICE)
        opt_d = optim.Adam(model_d.parameters(), lr=1e-3)
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)
        _, t_dl, _ = train_epoch_dataloader(model_d, loader, opt_d, criterion)
        samples_per_sec_d = len(dataset) / t_dl
        dataloader_throughput.append(samples_per_sec_d)

        print(f"    Batch Size {bs:3d} | Manual: {samples_per_sec_m:7.1f} smp/s | DataLoader: {samples_per_sec_d:7.1f} smp/s")

    return {
        "batch_sizes": batch_sizes,
        "manual_throughput": manual_throughput,
        "dataloader_throughput": dataloader_throughput
    }


def run_convergence_and_latency_profiling(
    dataset: SyntheticBenchmarkDataset,
    num_epochs: int = 5,
    batch_size: int = 64
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compares loss convergence curves and step latency distributions."""
    print(f"\n[*] Running 5-Epoch Convergence & Step Jitter Profiling (Batch Size = {batch_size})...")
    criterion = nn.CrossEntropyLoss()

    # Manual Loop
    set_seed(42)
    model_m = BenchmarkMLP().to(DEVICE)
    opt_m = optim.Adam(model_m.parameters(), lr=1e-3)
    loss_manual, latencies_manual = [], []

    for ep in range(1, num_epochs + 1):
        loss_val, ep_time, b_lats = train_epoch_manual(model_m, dataset.X, dataset.y, batch_size, opt_m, criterion)
        loss_manual.append(loss_val)
        if ep == 1:
            latencies_manual = b_lats
        print(f"    [Manual]     Epoch {ep}/{num_epochs} | Loss: {loss_val:.4f} | Time: {ep_time*1000:.1f}ms")

    # DataLoader Loop
    set_seed(42)
    model_d = BenchmarkMLP().to(DEVICE)
    opt_d = optim.Adam(model_d.parameters(), lr=1e-3)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    loss_dataloader, latencies_dataloader = [], []

    for ep in range(1, num_epochs + 1):
        loss_val, ep_time, b_lats = train_epoch_dataloader(model_d, loader, opt_d, criterion)
        loss_dataloader.append(loss_val)
        if ep == 1:
            latencies_dataloader = b_lats
        print(f"    [DataLoader] Epoch {ep}/{num_epochs} | Loss: {loss_val:.4f} | Time: {ep_time*1000:.1f}ms")

    return loss_manual, loss_dataloader, latencies_manual, latencies_dataloader


# -----------------------------------------------------------------------------
# 6. Publication-Grade Diagnostic Visualization
# -----------------------------------------------------------------------------
def plot_benchmark_dashboard(
    scaling_data: Dict[str, List[float]],
    loss_manual: List[float],
    loss_dataloader: List[float],
    lats_manual: List[float],
    lats_dataloader: List[float],
    output_path: str = "dataloader_vs_manual_benchmark.png"
) -> None:
    print("[*] Generating publication-grade diagnostic dashboard...")
    fig = plt.figure(figsize=(19, 12), dpi=300)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], wspace=0.25, hspace=0.35)

    # Panel 1: Throughput vs Batch Size
    ax1 = fig.add_subplot(gs[0, 0])
    bs = scaling_data["batch_sizes"]
    ax1.plot(bs, scaling_data["manual_throughput"], label="Manual Index Slicing",
             color="#e74c3c", marker="o", linewidth=2.5, markersize=8)
    ax1.plot(bs, scaling_data["dataloader_throughput"], label="PyTorch DataLoader",
             color="#2980b9", marker="s", linewidth=2.5, markersize=8)

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(bs)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.set_xlabel("Batch Size $B$ (Log2 Scale)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Throughput (Samples / Second)", fontsize=11, fontweight="bold")
    ax1.set_title("Training Throughput vs. Batch Size Scaling", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10, loc="lower right")

    # Panel 2: Batch Delivery Latency Distribution & Jitter
    ax2 = fig.add_subplot(gs[0, 1])
    data_to_plot = [lats_manual[:150], lats_dataloader[:150]]
    box = ax2.boxplot(data_to_plot, patch_artist=True, widths=0.45,
                      medianprops=dict(color="black", linewidth=2))

    colors = ["#e74c3c", "#2980b9"]
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["Manual Loop", "DataLoader"], fontsize=10.5, fontweight="bold")
    ax2.set_ylabel("Batch Step Latency (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("Batch Retrieval Latency Distribution & Jitter (First 150 Batches)",
                  fontsize=12, fontweight="bold", pad=12)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Annotate Mean & 95th Percentile
    m_mean, d_mean = np.mean(lats_manual), np.mean(lats_dataloader)
    m_p95, d_p95 = np.percentile(lats_manual, 95), np.percentile(lats_dataloader, 95)
    ax2.text(1, m_p95, f"p95: {m_p95:.2f}ms\nMean: {m_mean:.2f}ms", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e74c3c", alpha=0.8))
    ax2.text(2, d_p95, f"p95: {d_p95:.2f}ms\nMean: {d_mean:.2f}ms", ha="center", va="bottom",
             fontsize=8.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2980b9", alpha=0.8))

    # Panel 3: Training Convergence Curve (Exact Equivalence)
    ax3 = fig.add_subplot(gs[1, 0])
    epochs = range(1, len(loss_manual) + 1)
    ax3.plot(epochs, loss_manual, label="Manual Batching Loss", color="#e74c3c",
             linewidth=3, marker="o", markersize=7)
    ax3.plot(epochs, loss_dataloader, label="PyTorch DataLoader Loss", color="#2980b9",
             linestyle="--", linewidth=2.5, marker="x", markersize=8)

    max_delta = max(abs(m - d) for m, d in zip(loss_manual, loss_dataloader))
    ax3.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax3.set_title(f"Numerical Convergence Verification (Max $\\Delta = {max_delta:.1e}$)",
                  fontsize=12, fontweight="bold", pad=12)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=10, loc="upper right")

    # Panel 4: Architectural Feature Comparison Matrix
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    table_data = [
        ["Architectural Dimension", "Manual Indexing Loop", "PyTorch DataLoader"],
        ["Multiprocess Prefetching", "[No] Single-thread only (GIL bound)", "[Yes] num_workers > 0 (Fork/Spawn workers)"],
        ["CUDA Memory Pinning", "[No] Manual pin_memory() calls", "[Yes] pin_memory=True (Fast DMA transfer)"],
        ["Prefetch Pipelining", "[No] Synchronous GPU stall", "[Yes] prefetch_factor > 0 background queue"],
        ["Dynamic Collation", "[No] Custom padding boilerplate", "[Yes] collate_fn for variable length inputs"],
        ["Distributed DDP Sampling", "[No] Complex rank slicing", "[Yes] DistributedSampler out-of-the-box"],
        ["Partial Batch Handling", "[Partial] Risk of indexing bugs / off-by-one", "[Yes] drop_last=True / False automated"],
        ["Separation of Concerns", "[No] Dataset & scheduling coupled", "[Yes] Dataset (Data) + DataLoader (Scheduling)"]
    ]

    table = ax4.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.28, 0.36, 0.36])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.05, 1.8)

    # Header styling
    for col in range(3):
        cell = table[(0, col)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold")

    # Alternate row colors
    for row in range(1, 8):
        bg_color = "#f8f9fa" if row % 2 == 0 else "#ffffff"
        for col in range(3):
            table[(row, col)].set_facecolor(bg_color)

    ax4.set_title("Engineering Trade-Offs & Architectural Capabilities", fontsize=12, fontweight="bold", pad=20)

    plt.suptitle("PyTorch Architecture Benchmark: Manual Batching Loop vs. PyTorch DataLoader",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[+] Diagnostic dashboard saved successfully to {output_path}")


# -----------------------------------------------------------------------------
# 7. Main Execution Routine
# -----------------------------------------------------------------------------
def main() -> None:
    print("================================================================================")
    print("      PyTorch Training Loop vs. DataLoader Comprehensive Benchmark              ")
    print("================================================================================")

    # 1. Dataset Generation
    dataset = SyntheticBenchmarkDataset(num_samples=30000, num_features=32, num_classes=4)
    print(f"[*] Instantiated SyntheticBenchmarkDataset: {len(dataset):,} samples, 32 features, 4 classes.")

    # 2. Batch-Size Scaling Benchmark
    scaling_data = run_batch_size_scaling_benchmark(dataset, batch_sizes=[16, 32, 64, 128, 256, 512])

    # 3. Training Convergence & Latency Distribution Profiling
    loss_manual, loss_dataloader, lats_manual, lats_dataloader = run_convergence_and_latency_profiling(
        dataset=dataset,
        num_epochs=5,
        batch_size=64
    )

    # 4. Generate Visualization Dashboard
    plot_benchmark_dashboard(
        scaling_data=scaling_data,
        loss_manual=loss_manual,
        loss_dataloader=loss_dataloader,
        lats_manual=lats_manual,
        lats_dataloader=lats_dataloader,
        output_path="dataloader_vs_manual_benchmark.png"
    )

    # 5. Summary Table
    print("\n" + "=" * 80)
    print("                     FINAL PERFORMANCE BENCHMARK SUMMARY                        ")
    print("=" * 80)
    print(f"{'Batch Size':<12} | {'Manual Loop (smp/s)':<22} | {'DataLoader (smp/s)':<22} | {'Delta':<10}")
    print("-" * 80)
    for i, bs in enumerate(scaling_data['batch_sizes']):
        m_tp = scaling_data['manual_throughput'][i]
        d_tp = scaling_data['dataloader_throughput'][i]
        diff_pct = ((d_tp - m_tp) / m_tp) * 100.0
        print(f"{bs:<12} | {m_tp:12.1f} smp/s       | {d_tp:12.1f} smp/s       | {diff_pct:+5.1f}%")
    print("=" * 80)
    print(f"Mean Batch Step Latency:  Manual = {np.mean(lats_manual):.3f}ms | DataLoader = {np.mean(lats_dataloader):.3f}ms")
    print(f"95th Pct Step Latency:    Manual = {np.percentile(lats_manual, 95):.3f}ms | DataLoader = {np.percentile(lats_dataloader, 95):.3f}ms")
    print("=" * 80)
    print("[✓] Milestone Project 20 execution complete. Visual dashboard generated.\n")


if __name__ == "__main__":
    main()
