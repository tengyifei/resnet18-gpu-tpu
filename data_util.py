import ray
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

