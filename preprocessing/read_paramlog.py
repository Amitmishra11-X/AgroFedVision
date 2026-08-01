path = path = r"D:\Download1\Agriculture_Multispectral_Aerial\Agri\Paddy\paddy_season4_RededgeMultispectral_20190917_05m\paramlog.dat"

with open(path, "r", errors="ignore") as f:
    text = f.read()

print(text[:5000])