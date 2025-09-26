import torch
from torch.utils.data import DataLoader

def get_data_loaders(train_dataset, test_dataset, val_dataset, seed_worker,
                     batch_size=32, seed=0):
    """
    Helper function to fetch dataloaders

    Args:
        train_dataset: torch.tensor
        Training data
        test_dataset: torch.tensor
        Test data
        batch_size: int
        Batch Size
        seed: int
        Set seed for reproducibility

    Returns:
        emnist_train: torch.loader
        Training Data
        emnist_test: torch.loader
        Test Data
    """
    g_seed = torch.Generator()
    g_seed.manual_seed(seed)

    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=2,
                                worker_init_fn=seed_worker,
                                generator=g_seed)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            worker_init_fn=seed_worker,
                            generator=g_seed)
    test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            worker_init_fn=seed_worker,
                            generator=g_seed)

    return train_loader, val_loader, test_loader