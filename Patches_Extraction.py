def extract_patches(img, mask, size, min_valid_pixels=1, stride=None):
    """
    img  : (bands, H, W) numpy array
    mask : (H, W) numpy array with 0..n_classes and 255 as unlabeled
    size : patch size (e.g., 128)
    stride: step size (if None, will default to half of patch size)
    """
    X, Y = [], []
    half = size // 2
    _, H, W = img.shape

    if stride is None:
        stride = size // 2  # 50% overlap by default

    for r in range(half, H-half, stride):
        for c in range(half, W-half, stride):
            patch_img = img[:, r-half:r+half, c-half:c+half]
            patch_mask = mask[r-half:r+half, c-half:c+half]

            if patch_img.shape[1:] != (size, size):
                continue

            # Keep patch if at least one labeled pixel exists
            if (patch_mask != 255).any():
                X.append(patch_img)
                Y.append(patch_mask)

    return np.array(X), np.array(Y)

X, Y = extract_patches(img, mask, PATCH_SIZE)

print("Image patches:", X.shape)
print("Mask patches:", Y.shape)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

class SegDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

