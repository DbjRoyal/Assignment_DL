class UNet(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()

        def CBR(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(in_ch, 64), CBR(64, 64))
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))

        self.pool = nn.MaxPool2d(2)

        self.mid = nn.Sequential(CBR(256, 512), CBR(512, 512))

        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = nn.Sequential(CBR(512, 256), CBR(256, 256))

        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 128))

        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.mid(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(m), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return self.out(d1)

device = "cuda" if torch.cuda.is_available() else "cpu"
n_classes = len(np.unique(points[LABEL_COL]))

model = UNet(X.shape[1], n_classes).to(device)