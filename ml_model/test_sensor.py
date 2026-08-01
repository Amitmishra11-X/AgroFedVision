from sensor_utils import predict_sensor

sensor_reading = {
    "N": 35,
    "P": 45,
    "K": 25,
    "temperature": 30,
    "humidity": 65,
    "ph": 6.8,
    "rainfall": 120,
    "soil_moisture": 30,

    # Add every remaining feature your model was trained on
    "soil_type": 1,
    "sunlight_exposure": 7,
    "wind_speed": 8,
    "co2_concentration": 420,
    "organic_matter": 3.2,
    "irrigation_frequency": 2,
    "crop_density": 80,
    "pest_pressure": 1,
    "fertilizer_usage": 40,
    "growth_stage": 2,
    "urban_area_proximity": 15,
    "water_source_type": 1,
    "frost_risk": 0,
    "water_usage_efficiency": 78
}

result = predict_sensor(sensor_reading)

from pprint import pprint
pprint(result)