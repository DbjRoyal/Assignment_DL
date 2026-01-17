model.eval()
classified = np.zeros((H, W), dtype=np.uint8)

with torch.no_grad():
    for r in tqdm(range(0, H-PATCH_SIZE, PATCH_SIZE)):
        for c in range(0, W-PATCH_SIZE, PATCH_SIZE):
            patch = img[:, r:r+PATCH_SIZE, c:c+PATCH_SIZE]
            patch = torch.tensor(patch).unsqueeze(0).to(device)
            pred = model(patch).argmax(1).squeeze().cpu().numpy()
            classified[r:r+PATCH_SIZE, c:c+PATCH_SIZE] = pred
