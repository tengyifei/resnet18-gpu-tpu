from dataclasses import dataclass
from tqdm import tqdm
import torch
import ray
import numpy as np


class ModelDictMismatchError(ValueError):
  """Custom error for model dictionary mismatches."""

  def __init__(self, message, gpu_dict, tpu_dict):
    super().__init__(message)
    self.gpu_dict = gpu_dict
    self.tpu_dict = tpu_dict
    
  def get_abs_diff(self, key):
    gpu_tensor = self.gpu_dict[key]
    tpu_tensor = self.tpu_dict[key]
    return torch.abs(gpu_tensor - tpu_tensor).cpu().numpy()
  
  def get_pct_diff(self, key):
    gpu_tensor = self.gpu_dict[key]
    tpu_tensor = self.tpu_dict[key]
    abs_diff = torch.abs(gpu_tensor - tpu_tensor)
    pct_diff = 100.0 * abs_diff / (torch.abs(gpu_tensor) + 1e-9)
    return pct_diff.cpu().numpy()


def check_model_dicts_are_the_same(gpu_actor, tpu_actor):
  gpu_state_dict: dict = ray.get(gpu_actor.get_state_dict.remote())
  tpu_state_dict: dict = ray.get(tpu_actor.get_state_dict.remote())
  check_dicts(gpu_state_dict, tpu_state_dict, "state")
  

def check_model_grads_are_the_same(gpu_grads, tpu_grads):
  check_dicts(gpu_grads, tpu_grads, "grad")


def check_dicts(gpu_state_dict, tpu_state_dict, name):
  # Check that both state dicts have the same keys
  assert gpu_state_dict.keys() == tpu_state_dict.keys(), f"{name} dict keys don't match!"

  # Compare all tensors in the state dicts
  all_close = True
  max_diff = 0.0
  max_pct_diff = 0.0
  max_med_pct_diff = 0.0
  problematic_layers = []

  for key in gpu_state_dict.keys():
    gpu_tensor = gpu_state_dict[key]
    tpu_tensor = tpu_state_dict[key]
    
    if not torch.equal(gpu_tensor, tpu_tensor):
      elem_diff = torch.abs(gpu_tensor - tpu_tensor)
      diff = torch.max(elem_diff).item()
      pct_diff = elem_diff / (torch.abs(gpu_tensor) + 1e-9) * 100.0
      max_diff = max(max_diff, diff)
      max_pct_diff = max(max_pct_diff, pct_diff.max().item())
      med_pct_diff = float(pct_diff.median().item())
      max_med_pct_diff = max(max_med_pct_diff, med_pct_diff)
      problematic_layers.append((key, diff, pct_diff.max().item(), med_pct_diff))
      all_close = False

  if all_close:
    print(f"✅ GPU and TPU models have identical {name} dicts")
  else:
    print(f"❌ Found differences in {name} dicts:")
    print(f"Maximum absolute difference: {max_diff:.6f}")
    print(f"Maximum percentage difference: {max_pct_diff:.6f}%")
    print(f"Maximum median-of-layer percentage difference: {max_med_pct_diff:.6f}%")
    print("\nLayers with differences:")
    for layer, diff, pct_diff, med_pct_diff in problematic_layers:
      print(f"- {layer}: max diff = {diff:.6f}, max pct diff = {pct_diff:.3f}%, med pct diff = {med_pct_diff:.3f}%")
    raise ModelDictMismatchError(f"{name} dicts are not identical!", gpu_state_dict, tpu_state_dict)


@dataclass
class Metrics:
  max_abs_diff: float
  avg_abs_diff: float
  med_abs_diff: float
  max_pct_diff: float
  avg_pct_diff: float
  med_pct_diff: float
  avg_softmax_abs_diff: float
  top1_acc: float
  top5_acc: float


def test_eval_run(train_loader, gpu_actor, tpu_actor) -> Metrics:
  abs_deltas = []  # Store absolute differences
  pct_deltas = []  # Store percentage differences
  abs_softmax_deltas = []  # Store softmax absolute differences
  gpu_correct_top1 = 0
  gpu_correct_top5 = 0
  tpu_correct_top1 = 0
  tpu_correct_top5 = 0
  total = 0

  for _, data in tqdm(zip(range(20), train_loader)):
    input_ref = ray.put(data['image'])
    labels = data['label']
    gpu_future = gpu_actor.run_eval_pass.remote(input_ref)
    tpu_future = tpu_actor.run_eval_pass.remote(input_ref)
    gpu_output: torch.Tensor = ray.get(gpu_future)
    tpu_output: torch.Tensor = ray.get(tpu_future)

    assert gpu_output.shape == (data['image'].size(0), 1000)
    if gpu_output.shape != tpu_output.shape:
      raise ValueError("Output shapes differ between GPU and TPU!")

    # Absolute difference
    abs_diff = torch.abs(gpu_output - tpu_output)
    abs_softmax_diff = torch.abs(torch.nn.functional.softmax(gpu_output, dim=1) - torch.nn.functional.softmax(tpu_output, dim=1))
    # Percentage difference calculation
    # Using epsilon to avoid division by zero
    epsilon = 1e-9
    pct_diff = 100.0 * abs_diff / (torch.abs(gpu_output) + epsilon)
    
    abs_deltas.append(abs_diff)
    pct_deltas.append(pct_diff)
    abs_softmax_deltas.append(abs_softmax_diff)
    
    # Calculate top-1 and top-5 accuracy
    _, gpu_pred = gpu_output.topk(5, 1, True, True)
    _, tpu_pred = tpu_output.topk(5, 1, True, True)
    gpu_pred = gpu_pred.cpu()
    tpu_pred = tpu_pred.cpu()
    labels = labels.view(-1, 1)

    # Top-1 accuracy
    gpu_correct_top1 += gpu_pred[:, 0].eq(labels).sum().item()
    tpu_correct_top1 += tpu_pred[:, 0].eq(labels).sum().item()
    
    # Top-5 accuracy
    gpu_correct = labels.eq(gpu_pred).any(dim=1).sum().item()
    tpu_correct = labels.eq(tpu_pred).any(dim=1).sum().item()
    gpu_correct_top5 += gpu_correct
    tpu_correct_top5 += tpu_correct
    
    total += labels.size(0)
  
  # Calculate final metrics
  gpu_top1_acc = 100.0 * gpu_correct_top1 / total
  gpu_top5_acc = 100.0 * gpu_correct_top5 / total
  tpu_top1_acc = 100.0 * tpu_correct_top1 / total 
  tpu_top5_acc = 100.0 * tpu_correct_top5 / total

  print(f"""Absolute difference between GPU and TPU outputs:
  Max: {np.max(abs_deltas):.6f}
  Median: {np.median(abs_deltas):.6f}
  Mean: {np.mean(abs_deltas):.6f}
  """)
  
  print(f"""Percentage difference between GPU and TPU outputs:
  Max: {np.max(pct_deltas):.2f}%
  Median: {np.median(pct_deltas):.2f}%
  Mean: {np.mean(pct_deltas):.2f}%
  """)

  print(f"""Accuracy metrics:
  GPU Top-1: {gpu_top1_acc:.2f}%
  GPU Top-5: {gpu_top5_acc:.2f}%
  TPU Top-1: {tpu_top1_acc:.2f}%
  TPU Top-5: {tpu_top5_acc:.2f}%
  """)
  
  return Metrics(
    max_abs_diff=float(np.max(abs_deltas)),
    avg_abs_diff=float(np.mean(abs_deltas)),
    med_abs_diff=float(np.median(abs_deltas)),
    max_pct_diff=float(np.max(pct_deltas)),
    avg_pct_diff=float(np.mean(pct_deltas)),
    med_pct_diff=float(np.median(pct_deltas)),
    avg_softmax_abs_diff=float(np.mean(abs_softmax_deltas)),
    top1_acc=abs(gpu_top1_acc - tpu_top1_acc),
    top5_acc=abs(gpu_top5_acc - tpu_top5_acc),
  )
