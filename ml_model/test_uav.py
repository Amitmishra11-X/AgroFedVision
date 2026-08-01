from pprint import pprint
from uav_utils import extract_uav_features

capture_folder = r"D:\Download1\Agriculture_Multispectral_Aerial\Agriculture_Multispectral_Aerial\Agri\Agri\Maize\maize_season4_RededgeMultispectral_20200125_10m_flight1\000"
image_id = "IMG_0041"      # Example

result = extract_uav_features(capture_folder, image_id)

print("\nUAV Result")
print("=" * 40)
pprint(result)