import os
import glob

base = r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial\Agri\Agri\Maize\maize_season4_RededgeMultispectral_20200125_10m_flight1\000"

files = sorted(
    glob.glob(
        os.path.join(base, "*_1.tif")
    )
)

print("Total Images:", len(files))

for i in range(0, 60):
    print(i, os.path.basename(files[i]))