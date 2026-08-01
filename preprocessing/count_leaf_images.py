import os

root = r"D:\Download1\Multi_Crop_Leaves_Disease"

for crop in os.listdir(root):

    crop_path = os.path.join(root, crop)

    if os.path.isdir(crop_path):

        print("\nCrop:", crop)

        total = 0

        for folder, _, files in os.walk(crop_path):

            count = len([
                f for f in files
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ])

            if count > 0:
                print(folder.split("\\")[-1], ":", count)
                total += count

        print("Total Images =", total)