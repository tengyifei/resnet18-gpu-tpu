def seed_worker(worker_id):
  import numpy as np
  import random
  import torch
  worker_seed = torch.initial_seed() % 2**32 + worker_id
  np.random.seed(worker_seed)
  random.seed(worker_seed)
  torch.manual_seed(worker_seed)
