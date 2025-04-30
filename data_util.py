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
    
    This implementation uses lazy evaluation to avoid loading the entire dataset into memory.
    """
    import random
    from torchvision.transforms import v2
    from datasets import concatenate_datasets
    from tqdm import tqdm
    
    random.seed(seed)
    
    # Load raw datasets (these are lazy by default)
    train = load_raw_dataset("train")
    test = load_raw_dataset("test")
    valid = load_raw_dataset("valid")
    
    # Create a concatenated dataset without loading everything into memory
    print('Concatenating datasets...')
    combined_dataset = concatenate_datasets([train, test, valid])
    
    # Stream through the dataset once to collect unique celebrity IDs and their indices
    print('Collecting label indices...')
    label_to_indices = {}
    for idx, item in tqdm(enumerate(combined_dataset)):
        label = int(item['celeb_id'])
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    # Get unique labels
    unique_labels = list(label_to_indices.keys())
    NUM_CLASSES = len(unique_labels)
    print(f"Found {NUM_CLASSES} unique celebrity IDs")
    
    # Create train/test splits based on indices only
    train_indices = []
    test_indices = []
    
    print('Creating train/test splits...')
    for label, indices in label_to_indices.items():
        # Skip labels with only one sample (though there shouldn't be any)
        if len(indices) <= 1:
            continue
            
        # Shuffle indices for this label
        indices_copy = indices.copy()
        random.shuffle(indices_copy)
        
        # Split 80% train, 20% test
        split_idx = max(1, int(len(indices_copy) * 0.8))  # Ensure at least one sample in each split
        train_indices.extend(indices_copy[:split_idx])
        test_indices.extend(indices_copy[split_idx:])
    
    print(f"Train indices: {len(train_indices)}, Test indices: {len(test_indices)}")
    
    # Create a custom dataset class that applies transformations lazily
    class CelebALazyDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, indices, unique_labels, transform=None):
            self.base_dataset = base_dataset
            self.indices = indices
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.transform = transform
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            # Get the original index
            orig_idx = self.indices[idx]
            
            # Get the image and label from the base dataset
            item = self.base_dataset[orig_idx]
            image = item['image']
            celeb_id = item['celeb_id']
            
            # Map the original celeb_id to a class index
            label_idx = self.label_map[celeb_id.item()]
            
            # Apply transformation if specified
            if self.transform:
                image = self.transform(image)
                
            return {'image': image, 'label': label_idx}
    
    # Define the transformation
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(224, 224)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create lazy datasets for train and test
    print('Creating train dataset')
    train_dataset = CelebALazyDataset(
        combined_dataset, 
        train_indices, 
        unique_labels,
        transform
    )
    
    print('Creating test dataset')
    test_dataset = CelebALazyDataset(
        combined_dataset, 
        test_indices, 
        unique_labels,
        transform
    )
    
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Train set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    return train_dataset, test_dataset, NUM_CLASSES


def download_dataset_custom_split_cached(seed=42):
    # Check if the dataset is already cached
    import os
    from pathlib import Path
    import json
    from tqdm import tqdm
    
    cache_dir = Path("cache")
    train_path = cache_dir / str(seed) / "train_dataset"
    test_path = cache_dir / str(seed) / "test_dataset"
    metadata_path = cache_dir / str(seed) / "metadata.json"
    NUM_CLASSES = None
    
    # Create directories if needed
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    
    # Check if cache is complete and valid
    cache_is_valid = False
    if (train_path / ".done").exists() and (test_path / ".done").exists() and metadata_path.exists():
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify the seed matches
            if metadata.get("seed") == seed:
                cache_is_valid = True
                NUM_CLASSES = metadata["num_classes"]
            else:
                print(f"Warning: Cached dataset has seed {metadata.get('seed')}, but requested seed is {seed}")
        except Exception as e:
            print(f"Error loading cached metadata: {e}")
    
    # Use cache if valid
    if cache_is_valid:
        print("Using cached datasets.")
        assert NUM_CLASSES is not None, "NUM_CLASSES should be set from metadata"
        
        # Create a dataset class that loads from disk
        class CachedDataset(torch.utils.data.Dataset):
            def __init__(self, data_path):
                self.data_path = data_path
                self.file_list = sorted(list(data_path.glob("*.pt")), key=lambda x: int(x.stem))
                
            def __len__(self):
                return len(self.file_list)
                
            def __getitem__(self, idx):
                # Load the saved tensor from disk
                data = torch.load(self.file_list[idx])
                return data
        
        # Create datasets
        train_dataset = CachedDataset(train_path)
        test_dataset = CachedDataset(test_path)
        
        print(f"Number of classes: {NUM_CLASSES}")
        print(f"Train set size: {len(train_dataset)}")
        print(f"Test set size: {len(test_dataset)}")
        
        return train_dataset, test_dataset, NUM_CLASSES
    
    # Create new cache
    else:
        print("Creating new datasets...")
        
        # Get the datasets from the original function
        train_dataset, test_dataset, NUM_CLASSES = download_dataset_custom_split(seed)
        
        # Save metadata
        metadata = {
            "num_classes": NUM_CLASSES,
            "seed": seed
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Save train dataset
        print("Saving train dataset to disk...")
        for idx in tqdm(range(len(train_dataset))):
            data = train_dataset[idx]
            torch.save(data, train_path / f"{idx}.pt")
        
        # Mark train dataset as done
        with open(train_path / ".done", 'w') as f:
            f.write("done")
            
        # Save test dataset
        print("Saving test dataset to disk...")
        for idx in tqdm(range(len(test_dataset))):
            data = test_dataset[idx]
            torch.save(data, test_path / f"{idx}.pt")
            
        # Mark test dataset as done
        with open(test_path / ".done", 'w') as f:
            f.write("done")
            
        print("Datasets cached successfully.")
        return train_dataset, test_dataset, NUM_CLASSES
