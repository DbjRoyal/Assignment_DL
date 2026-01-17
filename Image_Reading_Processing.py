IMG_PATH = "/content/drive/MyDrive/Sentinel2_DL100m/JECAM_Site.tif"
PTS_PATH = "/content/drive/MyDrive/Sentinel2_DL100m/All_Merged1.shp"
AOI_PATH = "/content/drive/MyDrive/Sentinel2_DL100m/JECAM_Site.shp"
LABEL_COL = "Class1"
PATCH_SIZE = 64     
NUM_EPOCHS = 30
BATCH_SIZE = 4

points = gpd.read_file(PTS_PATH)

with rasterio.open(IMG_PATH) as src:
    img = src.read().astype(np.float32) / 10000.0
    transform = src.transform
    crs = src.crs
    H, W = src.height, src.width

points = points.to_crs(crs)

shapes = ((geom, int(val)) for geom, val in zip(points.geometry, points[LABEL_COL]))

mask = rasterize(
    shapes=shapes,
    out_shape=(H, W),
    transform=transform,
    fill=255,          # ignore index
    dtype=np.uint8
)
img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

unique, counts = np.unique(mask, return_counts=True)
print(dict(zip(unique, counts)))
print("Unique mask values:", np.unique(mask))

X, Y = extract_patches(img, mask, PATCH_SIZE, min_valid_pixels=1)
print(X.shape, Y.shape)
