import matplotlib.pyplot as plt

# @title Plotting functions
def imshow(img):
  """
  Display unnormalized image

  Args:
    img: np.ndarray
      Datapoint to visualize

  Returns:
    Nothing
  """
  img = img / 2 + 0.5     # Unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.axis(False)
  plt.show()


def plot_weights(norm, labels, ws,
                 title='Weight Size Measurement'):
  """
  Plot of weight size measurement [norm value vs layer]

  Args:
    norm: float
      Norm values
    labels: list
      Targets
    ws: list
      Weights
    title: string
      Title of plot

  Returns:
    Nothing
  """
  plt.figure(figsize=[8, 6])
  plt.title(title)
  plt.ylabel('Frobenius Norm Value')
  plt.xlabel('Model Layers')
  plt.bar(labels, ws)
  plt.axhline(y=norm,
              linewidth=1,
              color='r',
              ls='--',
              label='Total Model F-Norm')
  plt.legend()
  plt.show()


def early_stop_plot(train_acc_earlystop,
                    val_acc_earlystop, best_epoch):
  """
  Plot of early stopping

  Args:
    train_acc_earlystop: np.ndarray
      Training accuracy log until early stop point
    val_acc_earlystop: np.ndarray
      Val accuracy log until early stop point
    best_epoch: int
      Epoch at which early stopping occurs

  Returns:
    Nothing
  """
  plt.figure(figsize=(8, 6))
  plt.plot(val_acc_earlystop,label='Val - Early',c='red',ls = 'dashed')
  plt.plot(train_acc_earlystop,label='Train - Early',c='red',ls = 'solid')
  plt.axvline(x=best_epoch, c='green', ls='dashed',
              label='Epoch for Max Val Accuracy')
  plt.title('Early Stopping')
  plt.ylabel('Accuracy (%)')
  plt.xlabel('Epoch')
  plt.legend()
  plt.show()


  # @title Set random seed

# @markdown Executing `set_seed(seed=seed)` you are setting the seed

# For DL its critical to set the random seed so that students can have a
# baseline to compare their results to expected results.
# Read more here: https://pytorch.org/docs/stable/notes/randomness.html

# Call `set_seed` function in the exercises to ensure reproducibility.
import random
import torch
import numpy as np

def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness. NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')


# In case that `DataLoader` is used
def seed_worker(worker_id):
  """
  DataLoader will reseed workers following randomness in
  multi-process data loading algorithm.

  Args:
    worker_id: integer
      ID of subprocess to seed. 0 means that
      the data will be loaded in the main process
      Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details

  Returns:
    Nothing
  """
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


  # @title Set device (GPU or CPU). Execute `set_device()`
# especially if torch modules used.

# Inform the user if the notebook uses GPU or CPU.

def set_device():
  """
  Set the device. CUDA if available, CPU otherwise

  Args:
    None

  Returns:
    Nothing
  """
  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device != "cuda":
    print("WARNING: For this notebook to perform best, "
        "if possible, in the menu under `Runtime` -> "
        "`Change runtime type.`  select `GPU` ")
  else:
    print("GPU is enabled in this notebook.")

  return device