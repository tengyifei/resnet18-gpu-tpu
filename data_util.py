import torch
import numpy as np
from torch.utils.data import DataLoader, default_collate
from torchvision import datasets, models, tv_tensors


# Download dataset
def download_dataset(split="test"):
  from torchvision.transforms import v2
  from datasets import load_dataset
  train_dataset = load_dataset("flwrlabs/celeba", split=split, trust_remote_code=True, cache_dir='cache')
  train_dataset = train_dataset.with_format("torch")
  CLASSES = train_dataset.unique('celeb_id')  # type: ignore
  NUM_CLASSES = len(CLASSES)

  print("Number of classes:", NUM_CLASSES)

  def transform_example(ex):
      assert isinstance(CLASSES, list)
      imgs = ex["image"]
      labels = ex["celeb_id"]
      transform_fct = v2.Compose([
          v2.PILToTensor(),
          v2.ToDtype(torch.float32, scale=True),
          v2.Resize(size=(224, 224)),
          v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
      img_bbox_pairs = [transform_fct(img, label) for img, label in zip(imgs, labels)]

      return {"image": [img for img, label in img_bbox_pairs],
              "label": [CLASSES.index(label) for img, label in img_bbox_pairs]}

  train_dataset = train_dataset.map(transform_example, num_proc=4, batched=True, writer_batch_size=4_000, load_from_cache_file=True)  # type: ignore
  return train_dataset, NUM_CLASSES


def load_raw_dataset(split):
  from datasets import load_dataset
  dataset = load_dataset("flwrlabs/celeba", split=split, trust_remote_code=True, cache_dir='cache')
  return dataset.with_format("torch")


def download_dataset_custom_split(seed=42):
  """Problem: there is no label overlap between train/test/valid sets.

  We can't directly use the validation set to validate accuracy because it contains
  completely unseen labels.
  
  For the purpose of this specific convergence test, we'll do our own split
  such that the same labels show up in both train and test sets.
  """
  
  # Load raw datasets
  train = load_raw_dataset("train")
  test = load_raw_dataset("test")
  valid = load_raw_dataset("valid")

  # Split into train and test sets, ensuring that the test set is 20% of the total dataset,
  # and each label int (found by "label" dict key) appears roughly equally in both sets.
  from torchvision.transforms import v2
  import random
  random.seed(seed)
  
  # Combine all datasets
  all_data = {
      'image': train['image'] + test['image'] + valid['image'],
      'celeb_id': train['celeb_id'] + test['celeb_id'] + valid['celeb_id']
  }
  
  # Get unique labels and their indices
  unique_labels = list(set(all_data['celeb_id']))
  label_to_indices = {label: [] for label in unique_labels}
  
  # Group indices by label
  for idx, label in enumerate(all_data['celeb_id']):
      label_to_indices[label].append(idx)
  
  # Create train and test indices with stratification
  train_indices = []
  test_indices = []
  
  for label, indices in label_to_indices.items():
      # Shuffle indices for this label
      random.shuffle(indices)
      
      # Split 80% train, 20% test
      split_idx = int(len(indices) * 0.8)
      train_indices.extend(indices[:split_idx])
      test_indices.extend(indices[split_idx:])
  
  # Create the actual datasets
  train_data = {
      'image': [all_data['image'][i] for i in train_indices],
      'celeb_id': [all_data['celeb_id'][i] for i in train_indices]
  }
  
  test_data = {
      'image': [all_data['image'][i] for i in test_indices],
      'celeb_id': [all_data['celeb_id'][i] for i in test_indices]
  }
  
  # Convert to proper dataset format
  from datasets import Dataset
  train_dataset = Dataset.from_dict(train_data)
  test_dataset = Dataset.from_dict(test_data)
  
  # Apply transformations
  CLASSES = unique_labels
  NUM_CLASSES = len(CLASSES)
  
  def transform_example(ex):
      imgs = ex["image"]
      labels = ex["celeb_id"]
      transform_fct = v2.Compose([
          v2.PILToTensor(),
          v2.ToDtype(torch.float32, scale=True),
          v2.Resize(size=(224, 224)),
          v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
      img_label_pairs = [(img, label) for img, label in zip(imgs, labels)]
      
      return {
          "image": [transform_fct(img) for img, _ in img_label_pairs],
          "label": [CLASSES.index(label) for _, label in img_label_pairs]
      }
  
  train_dataset = train_dataset.map(transform_example, batched=True, load_from_cache_file=True)
  test_dataset = test_dataset.map(transform_example, batched=True, load_from_cache_file=True)
  
  print(f"Number of classes: {NUM_CLASSES}")
  print(f"Train set size: {len(train_dataset)}")
  print(f"Test set size: {len(test_dataset)}")
  
  return train_dataset, test_dataset, NUM_CLASSES
