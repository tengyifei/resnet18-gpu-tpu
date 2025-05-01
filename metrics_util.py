# Some utility functions to save and load metrics
import os
import json
from dataclasses import dataclass

def save_metrics(gpu_losses, tpu_losses, gpu_validation_metrics, tpu_validation_metrics, run_name):
  # Save the gpu and tpu losses to `outputs/` in JSON format.
  os.makedirs(f"outputs/convergence/{run_name}", exist_ok=True)

  with open(f"outputs/convergence/{run_name}/gpu_losses.json", "w") as f:
    json.dump(gpu_losses, f)

  with open(f"outputs/convergence/{run_name}/tpu_losses.json", "w") as f:
    json.dump(tpu_losses, f)

  with open(f"outputs/convergence/{run_name}/gpu_validation_metrics.json", "w") as f:
    json.dump(gpu_validation_metrics, f)

  with open(f"outputs/convergence/{run_name}/tpu_validation_metrics.json", "w") as f:
    json.dump(tpu_validation_metrics, f)
    

@dataclass
class Metrics:
  gpu_losses: list
  tpu_losses: list
  gpu_validation_metrics: list
  tpu_validation_metrics: list

def load_metrics(run_name) -> Metrics:
  # Load the gpu and tpu losses from `outputs/` in JSON format.
  with open(f"outputs/convergence/{run_name}/gpu_losses.json", "r") as f:
    gpu_losses = json.load(f)

  with open(f"outputs/convergence/{run_name}/tpu_losses.json", "r") as f:
    tpu_losses = json.load(f)

  with open(f"outputs/convergence/{run_name}/gpu_validation_metrics.json", "r") as f:
    gpu_validation_metrics = json.load(f)

  with open(f"outputs/convergence/{run_name}/tpu_validation_metrics.json", "r") as f:
    tpu_validation_metrics = json.load(f)

  return Metrics(gpu_losses, tpu_losses, gpu_validation_metrics, tpu_validation_metrics)


# Create figure with 2x2 subplots
def plot_metrics(metrics: Metrics, run_name: str):
  import matplotlib.pyplot as plt
  import numpy as np

  # Check if the metrics are loaded correctly
  if not isinstance(metrics, Metrics):
    raise ValueError("Metrics should be an instance of the Metrics dataclass.")

  # Create a figure with 2x2 subplots
  plt.rcParams['axes.titlesize'] = 14
  plt.rcParams['axes.labelsize'] = 12
  plt.rcParams['xtick.labelsize'] = 10
  plt.rcParams['ytick.labelsize'] = 10
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
  fig.suptitle(f'Training and Validation Metrics Comparison between GPU and TPU ({run_name})', fontsize=14, y=0.95)

  # 1. Training Loss Plot
  ax1.plot(metrics.gpu_losses, label='GPU Loss', alpha=0.7)
  ax1.plot(metrics.tpu_losses, label='TPU Loss', alpha=0.7)
  ax1.set_xlabel('Iteration')
  ax1.set_ylabel('Training Loss')
  ax1.set_title('Training Loss Curves')
  ax1.legend()

  # 2. Validation Loss Plot
  val_loss_gpu = [m['avg_valid_loss'] for m in metrics.gpu_validation_metrics]
  val_loss_tpu = [m['avg_valid_loss'] for m in metrics.tpu_validation_metrics]
  epochs = range(len(val_loss_gpu))

  ax2.plot(epochs, val_loss_gpu, label='GPU Val Loss', marker='o')
  ax2.plot(epochs, val_loss_tpu, label='TPU Val Loss', marker='o')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Validation Loss')
  ax2.set_title('Validation Loss Curves')

  # Add best validation loss markers with stars and connecting lines
  best_gpu_val_idx = np.argmin(val_loss_gpu)
  best_tpu_val_idx = np.argmin(val_loss_tpu)
  ax2.plot(best_gpu_val_idx, val_loss_gpu[best_gpu_val_idx], marker='*', color='blue', markersize=15)
  ax2.plot(best_tpu_val_idx, val_loss_tpu[best_tpu_val_idx], marker='*', color='orange', markersize=15)
  
  # Annotate with connecting lines
  ax2.annotate(f'Best GPU: {val_loss_gpu[best_gpu_val_idx]:.4f}',
               xy=(best_gpu_val_idx, val_loss_gpu[best_gpu_val_idx]),
               xytext=(20, 60), textcoords='offset points',
               arrowprops=dict(arrowstyle='-', color='darkgray', lw=1))
  ax2.annotate(f'Best TPU: {val_loss_tpu[best_tpu_val_idx]:.4f}',
               xy=(best_tpu_val_idx, val_loss_tpu[best_tpu_val_idx]),
               xytext=(0, 30), textcoords='offset points',
               arrowprops=dict(arrowstyle='-', color='darkgray', lw=1))
  ax2.legend()

  # 3. Top-1 Accuracy Plot
  top1_gpu = [m['top1_accuracy'] for m in metrics.gpu_validation_metrics]
  top1_tpu = [m['top1_accuracy'] for m in metrics.tpu_validation_metrics]

  ax3.plot(epochs, top1_gpu, label='GPU Top-1', marker='o')
  ax3.plot(epochs, top1_tpu, label='TPU Top-1', marker='o')
  ax3.set_xlabel('Epoch')
  ax3.set_ylabel('Top-1 Accuracy')
  ax3.set_title('Validation Top-1 Accuracy')

  # Add best top-1 accuracy markers with stars
  best_gpu_top1_idx = np.argmax(top1_gpu)
  best_tpu_top1_idx = np.argmax(top1_tpu)
  ax3.plot(best_gpu_top1_idx, top1_gpu[best_gpu_top1_idx], marker='*', color='blue', markersize=15)
  ax3.plot(best_tpu_top1_idx, top1_tpu[best_tpu_top1_idx], marker='*', color='orange', markersize=15)
  
  # Annotate with connecting lines
  ax3.annotate(f'Best GPU: {top1_gpu[best_gpu_top1_idx]:.4f}',
               xy=(best_gpu_top1_idx, top1_gpu[best_gpu_top1_idx]),
               xytext=(-30, -60), textcoords='offset points',
               arrowprops=dict(arrowstyle='-', color='darkgray', lw=1))
  ax3.annotate(f'Best TPU: {top1_tpu[best_tpu_top1_idx]:.4f}',
               xy=(best_tpu_top1_idx, top1_tpu[best_tpu_top1_idx]),
               xytext=(-90, -90), textcoords='offset points',
               arrowprops=dict(arrowstyle='-', color='darkgray', lw=1))
  ax3.legend()

  # 4. Top-5 Accuracy Plot
  top5_gpu = [m['top5_accuracy'] for m in metrics.gpu_validation_metrics]
  top5_tpu = [m['top5_accuracy'] for m in metrics.tpu_validation_metrics]

  ax4.plot(epochs, top5_gpu, label='GPU Top-5', marker='o')
  ax4.plot(epochs, top5_tpu, label='TPU Top-5', marker='o')
  ax4.set_xlabel('Epoch')
  ax4.set_ylabel('Top-5 Accuracy')
  ax4.set_title('Validation Top-5 Accuracy')

  # Add best top-5 accuracy markers with stars
  best_gpu_top5_idx = np.argmax(top5_gpu)
  best_tpu_top5_idx = np.argmax(top5_tpu)
  ax4.plot(best_gpu_top5_idx, top5_gpu[best_gpu_top5_idx], marker='*', color='blue', markersize=15)
  ax4.plot(best_tpu_top5_idx, top5_tpu[best_tpu_top5_idx], marker='*', color='orange', markersize=15)
  
  # Annotate with connecting lines
  ax4.annotate(f'Best GPU: {top5_gpu[best_gpu_top5_idx]:.4f}',
               xy=(best_gpu_top5_idx, top5_gpu[best_gpu_top5_idx]),
               xytext=(-20, -40), textcoords='offset points',
               arrowprops=dict(arrowstyle='-', color='darkgray', lw=1))
  ax4.annotate(f'Best TPU: {top5_tpu[best_tpu_top5_idx]:.4f}',
               xy=(best_tpu_top5_idx, top5_tpu[best_tpu_top5_idx]),
               xytext=(-90, -90), textcoords='offset points',
               arrowprops=dict(arrowstyle='-', color='darkgray', lw=1))
  ax4.legend()

  # Adjust layout
  plt.tight_layout()
  # Add extra space for the suptitle
  plt.subplots_adjust(top=0.90)
  plt.show()
