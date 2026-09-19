import torchvision.transforms as T

# ImageNet mean and std — required for VGG16, ResNet50, EfficientNet.
# These are the statistics of the 1.2M image dataset these models
# were originally trained on. Using any other values will hurt accuracy.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# TRAINING transforms — includes augmentation
train_transforms = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# VALIDATION and TEST transforms — no augmentation, ever
val_test_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
