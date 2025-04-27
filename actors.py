import os
from typing import Any
from copy import deepcopy

import ray
import torch
import numpy as np
from torch.utils.data import DataLoader, default_collate
from torchvision import datasets, models, tv_tensors

from data_util import download_dataset


class BaseActor:
  
  def __init__(self, model, device):
    self.model = model
    self.device = device
    self.num_ftrs = model.fc.in_features
    self.pid = os.getpid()
    self.cwd = os.getcwd()
    
    import torch.nn as nn
    self.loss_fct = nn.CrossEntropyLoss()

  def get_cwd(self) -> str:
    """Returns the current working directory of the actor."""
    return self.cwd

  def get_num_features(self) -> int:
    """Returns the number of features in the model."""
    return self.num_ftrs
  
  def reset_classifier(
      self,
      fc_state_dict: dict[str, Any],
      conv1_state_dict: dict[str, Any],
      bn1_state_dict: dict[str, Any]
    ) -> bool:
    """Replaces the weights of fc, conv1, and bn1 layers."""
    import torch.nn as nn
    try:
      # Ensure layers exist before loading state dict
      if hasattr(self.model, 'fc') and self.model.fc is not None:
          num_classes = fc_state_dict['weight'].shape[0]
          self.model.fc = nn.Linear(self.num_ftrs, num_classes)
          self.model.fc.load_state_dict(fc_state_dict, assign=True)
      else:
          raise ValueError("fc layer not found or is None in the model.")

      if hasattr(self.model, 'conv1') and self.model.conv1 is not None:
          self.model.conv1.load_state_dict(conv1_state_dict, assign=True)
      else:
          raise ValueError("conv1 layer not found or is None in the model.")

      if hasattr(self.model, 'bn1') and self.model.bn1 is not None:
          self.model.bn1.load_state_dict(bn1_state_dict, assign=True)
      else:
          raise ValueError("bn1 layer not found or is None in the model.")

      self.model.to(self.device)
      return True
    except Exception as e:
      print(f"Failed to reset classifier layers: {e}")
      raise e
  
  def init_optimizer(self, lr: float):
    import torch.optim as optim
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

  def get_state_dict(self) -> dict[str, Any]:
    """Returns the state dict of the model."""
    return deepcopy(self.model).to('cpu').state_dict()

  def run_eval_pass(self, input_tensor: torch.Tensor) -> torch.Tensor:
    """Runs a forward pass using the resident model."""
    self.framework_pre_hook()
    input_tensor_xpu = input_tensor.to(self.device)
    self.model.eval()
    with torch.no_grad():
      output = self.model(input_tensor_xpu)
    self.framework_post_hook()
    return output.cpu()
  
  def run_training_pass(self, input_tensor: torch.Tensor, labels: torch.Tensor) -> tuple[float, dict]:
    self.framework_pre_hook()
    self.model.train()
    input_tensor_xpu = input_tensor.to(self.device)
    labels_xpu = labels.to(self.device)
    self.optimizer.zero_grad()
    output = self.model(input_tensor_xpu)
    loss = self.loss_fct(output, labels_xpu)
    loss_float = loss.detach().cpu().item()
    loss.backward()
    self.optimizer.step()
    self.framework_post_hook()
    grad_dict = self.get_grad_dict()
    return loss_float, grad_dict

  def run_training_epoch(self, batch_size=512, training_split='test', shuffle=False, run_validation=False, dataloader_seed=0):
    import torch
    import numpy as np
    import random
    from torch.utils.data import DataLoader
    from training_util import seed_worker
    torch.manual_seed(dataloader_seed)
    np.random.seed(dataloader_seed)
    random.seed(dataloader_seed)
    train_dataset, NUM_CLASSES = download_dataset(split=training_split)
    
    if shuffle:
      train_loader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        worker_init_fn=seed_worker,
        in_order=True,
        generator=torch.Generator(),
        multiprocessing_context="spawn",
        drop_last=True,
        persistent_workers=False,
      )
    else:
      train_loader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=batch_size,
        num_workers=8,
        shuffle=False,
        worker_init_fn=seed_worker,
        in_order=True,
        multiprocessing_context="spawn",
        drop_last=False,
        persistent_workers=False,
      )

    # Training pass
    self.model.train()
    for i, batch in enumerate(train_loader):
      self.framework_pre_hook()
      images = batch["image"]
      labels = batch["label"]
      labels_checksum = self.checksum_labels(labels)
      images = images.to(self.device)
      labels = labels.to(self.device)
      loss, _grad_dict = self.run_training_pass(images, labels)
      self.framework_post_hook()
      yield i, loss, labels_checksum
    
    if not run_validation:
      return

    # Eval pass
    self.model.eval()
    valid_dataset, _ = download_dataset(split='valid')
    valid_loader = DataLoader(
      valid_dataset, # type: ignore
      batch_size=batch_size,
      num_workers=8,
      shuffle=False, # No need to shuffle validation data
      worker_init_fn=seed_worker,
      in_order=True,
      generator=torch.Generator(),
      multiprocessing_context="spawn",
      drop_last=False, # Don't drop last batch in validation
      persistent_workers=False,
    )
    total_valid_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    with torch.no_grad():
      for i, batch in enumerate(valid_loader):
        self.framework_pre_hook()
        images = batch["image"]
        labels = batch["label"]
        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(images)
        loss = self.loss_fct(outputs, labels)

        # Calculate accuracy
        _, topk = outputs.topk(5, 1, True, True)
        labels_expanded = labels.view(-1, 1).expand_as(topk)
        correct = topk.eq(labels_expanded)
        self.framework_post_hook()

        total_valid_loss += loss.detach().item()
        correct_top1 += correct[:, :1].sum().item()
        correct_top5 += correct[:, :5].sum().item()
        total_samples += labels.size(0)

    avg_valid_loss = total_valid_loss / len(valid_loader) if len(valid_loader) > 0 else 0
    top1_accuracy = correct_top1 / total_samples if total_samples > 0 else 0
    top5_accuracy = correct_top5 / total_samples if total_samples > 0 else 0

    validation_metrics = {
      "avg_valid_loss": avg_valid_loss,
      "top1_accuracy": top1_accuracy,
      "top5_accuracy": top5_accuracy,
    }
    yield validation_metrics
  
  def checksum_labels(self, labels: torch.Tensor) -> int:
    """Returns the checksum of the labels tensor."""
    numel = labels.numel()
    if numel == 0:
      label_checksum = 0
    elif numel == 1:
      label_checksum = hash(labels.item())
    else:
      rolled_labels = torch.roll(labels, shifts=1, dims=0)
      xor_result = torch.bitwise_xor(labels, rolled_labels)
      checksum_tensor = torch.sum(xor_result.long())
      label_checksum = int(checksum_tensor.item())
    return label_checksum
  
  def get_grad_dict(self):
    grad_dict = {}
    for name, param in self.model.named_parameters():
      if param.grad is not None:
        grad_dict[name] = param.grad.detach().cpu().clone()
      elif param.requires_grad:
        grad_dict[name] = None
    return grad_dict

  def print_model_architecture(self) -> str:
    import io
    with io.StringIO() as output:
      print(self.model, file=output)
      return output.getvalue()

  def framework_pre_hook(self): pass
  def framework_post_hook(self): pass


# ==========================
# Actor for GPU Execution
# ==========================
@ray.remote(num_gpus=1)
class GpuResNetWorker(BaseActor):
  def __init__(self):
    import torch
    if not torch.cuda.is_available():
      raise RuntimeError("CUDA not available for GpuResNetWorker")
    device = torch.device("cuda")
    # Load a pretrained ResNet18 model
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device)
    super().__init__(model, device)

  def get_device_info(self) -> str:
    return f"GPU Actor (PID: {self.pid}) using {self.device} ({torch.cuda.get_device_name(0)})"
  

# ==========================
# Actor for PyTorch/XLA TPU Execution
# ==========================
@ray.remote(resources={"TPU": 1})
class TpuResNetWorker(BaseActor):
  def __init__(self):
    import torch
    import torch.nn as nn

    try:
      import torch_xla
      import torch_xla.core.xla_model as xm
    except ImportError:
      raise ImportError("torch_xla is required for TpuResNetWorker")

    device = torch_xla.device()
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device)
    torch_xla.sync()
    super().__init__(model, device)
  
  def get_device_info(self) -> str:
    import torch_xla
    return f"TPU Actor (PID: {self.pid}) using {self.device} ({torch_xla.tpu.get_tpu_env()['ACCELERATOR_TYPE']})"
  
  def set_matmul_precision(self, precision: str):
    import torch_xla
    torch_xla._XLAC._xla_set_mat_mul_precision(precision)

  def framework_pre_hook(self):
    import torch_xla
    torch_xla.sync()

  def framework_post_hook(self):
    import torch_xla
    torch_xla.sync()


# ==========================
# Actor for torchax TPU Execution
# ==========================
@ray.remote(resources={"TPU": 1})
class TorchaxResNetWorker(BaseActor):
  def __init__(self):
    import torch
    import torch.nn as nn

    try:
      import torchax
      torchax.enable_globally()
    except ImportError:
      raise ImportError("torchax is required for TpuResNetWorker")

    device = 'jax'
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = model.to(device)
    super().__init__(model, device)
  
  def get_device_info(self) -> str:
    import jax
    return f"torchax TPU Actor (PID: {self.pid}) using {self.device} ({jax.devices()[0]})"
  
  def set_matmul_precision(self, precision: str):
    import torchax
    import jax
    match precision:
      case "default":
        torchax.enable_performance_mode()
      case "high":
        torchax.enable_accuracy_mode()
        jax.config.update('jax_default_matmul_precision', precision)
      case "highest":
        torchax.enable_accuracy_mode()
        jax.config.update('jax_default_matmul_precision', precision)
      case _:
        raise ValueError(f"Unknown precision level: {precision}")


def get_gpu_actor():
  gpu_actor: Any = GpuResNetWorker.remote()
  gpu_info = ray.get(gpu_actor.get_device_info.remote())
  print(gpu_info)
  return gpu_actor

def get_tpu_actor():
  tpu_actor: Any = TpuResNetWorker.remote()
  tpu_info = ray.get(tpu_actor.get_device_info.remote())
  print(tpu_info)
  return tpu_actor

def get_torchax_actor():
  torchax_actor: Any = TorchaxResNetWorker.remote()
  torchax_info = ray.get(torchax_actor.get_device_info.remote())
  print(torchax_info)
  return torchax_actor


# This will reliably reset every actor to the same random state.
class ActorResetter:
  def __init__(self, gpu_actor, num_classes):
    import numpy as np
    import torch
    import torch.nn as nn
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    num_ftrs: int = ray.get(gpu_actor.get_num_features.remote())
    self.num_ftrs = num_ftrs
    self.fc = nn.Linear(num_ftrs, num_classes)
    
    # These are not the classifier, but still reasonable to reset because the input data
    # domain distribution has changed.
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
  
  def reset_classifier_layers_to_identical_random(self, actor):
    assert self.num_ftrs == ray.get(actor.get_num_features.remote()), "Number of features mismatch"

    assert ray.get(actor.reset_classifier.remote(
        self.fc.state_dict(),
        self.conv1.state_dict(),
        self.bn1.state_dict()
      )), f"ACTORS {actor} failed to reset classifier layers."


def reset_all_actors(gpu_actor, tpu_actor, actor_resetter):
  if gpu_actor is not None:
    ray.kill(gpu_actor)
  if tpu_actor is not None:
    ray.kill(tpu_actor)
  gpu_actor = get_gpu_actor()
  tpu_actor = get_tpu_actor()
  actor_resetter.reset_classifier_layers_to_identical_random(gpu_actor)
  actor_resetter.reset_classifier_layers_to_identical_random(tpu_actor)
  from model_util import check_model_dicts_are_the_same
  check_model_dicts_are_the_same(gpu_actor, tpu_actor)
  return gpu_actor, tpu_actor
