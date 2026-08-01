import tifffile as tiff
import matplotlib.pyplot as plt

for i in range(1,6):
    img = tiff.imread(
        fr"D:\Download1\IMG_0041_{i}.tif"
    )

    plt.figure(figsize=(5,5))
    plt.imshow(img,cmap="gray")
    plt.title(f"Band {i}")
    plt.show()
