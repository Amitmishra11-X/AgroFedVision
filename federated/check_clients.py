import pandas as pd

for farm in ["a", "b", "c", "d"]:
    df = pd.read_csv(f"data/clients/farm_{farm}.csv")

    print(f"\nFarm {farm.upper()}")
    print("Shape:", df.shape)
    print(df.head(2))