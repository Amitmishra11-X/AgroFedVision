import joblib

cols = joblib.load("results/sensor_only/sensor_columns.pkl")

print("Total columns:", len(cols))
print()

for i, c in enumerate(cols, 1):
    print(f"{i:2d}. {c}")