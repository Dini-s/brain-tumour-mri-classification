import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from src.transforms import train_transforms, val_test_transforms

# Fixed seed — NEVER change this.
# Every model must use this seed so results are comparable.
SEED = 42
torch.manual_seed(SEED)

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def get_dataloaders(data_root: str, batch_size: int = 32):
    """
    Returns train_loader, val_loader, test_loader.

    Split strategy:
      - Training folder -> 80% train, 20% val (fixed seed)
      - Testing folder  -> test (untouched, only for final evaluation)
    """
    data_root = Path(data_root)

    # Load full training folder with train augmentation
    full_train = ImageFolder(root=data_root / "Training",
                              transform=train_transforms)

    # 80% train, 20% val
    n_total = len(full_train)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(full_train, [n_train, n_val],
                                       generator=generator)

    # val_set must use non-augmenting transforms, not train_transforms
    val_set.dataset = ImageFolder(root=data_root / "Training",
                                   transform=val_test_transforms)

    # Test set — never touch until final evaluation
    test_set = ImageFolder(root=data_root / "Testing",
                            transform=val_test_transforms)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                               shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=2)

    print(f"Train:      {len(train_set):,} images")
    print(f"Validation: {len(val_set):,} images")
    print(f"Test:       {len(test_set):,} images")
    print(f"Classes:    {full_train.classes}")

    return train_loader, val_loader, test_loader