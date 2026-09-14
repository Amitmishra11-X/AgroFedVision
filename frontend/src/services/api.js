export const API_BASE_URL =
  import.meta.env.VITE_AGROFEDVISION_API_URL || "http://127.0.0.1:8000";

export const SENSOR_FIELDS = [
  "N",
  "P",
  "K",
  "temperature",
  "humidity",
  "ph",
  "rainfall",
  "soil_moisture",
  "soil_type",
  "sunlight_exposure",
  "wind_speed",
  "co2_concentration",
  "organic_matter",
  "irrigation_frequency",
  "crop_density",
  "pest_pressure",
  "fertilizer_usage",
  "growth_stage",
  "urban_area_proximity",
  "water_source_type",
  "frost_risk",
  "water_usage_efficiency",
];

export const UAV_FIELDS = ["NDVI_Mean", "NDVI_Std", "NDRE_Mean", "NDRE_Std"];

export function formatFieldName(name) {
  return name
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function resolveAssetUrl(path) {
  if (!path || typeof path !== "string") return null;
  const normalized = path.replaceAll("\\", "/");
  if (/^https?:\/\//i.test(normalized)) return normalized;
  if (normalized.startsWith("/")) return `${API_BASE_URL}${normalized}`;
  if (normalized.startsWith("results/")) return `${API_BASE_URL}/${normalized}`;
  if (normalized.startsWith("api_uploads/")) {
    return `${API_BASE_URL}/${normalized.replace("api_uploads/", "uploads/")}`;
  }
  return normalized;
}

async function parseResponse(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const detail =
      payload?.detail ||
      payload?.message ||
      "The prediction service returned an error.";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }

  if (!payload || typeof payload !== "object") {
    throw new Error("Invalid response from AgroFedVision API.");
  }

  return payload;
}

function request(method, url, body) {
  if (typeof fetch === "function") {
    return fetch(url, { method, body });
  }

  return new Promise((resolve) => {
    const xhr = new XMLHttpRequest();
    xhr.open(method, url);
    xhr.onload = () => {
      resolve({
        ok: xhr.status >= 200 && xhr.status < 300,
        status: xhr.status,
        json: async () => JSON.parse(xhr.responseText || "null"),
      });
    };
    xhr.onerror = () => {
      resolve({
        ok: false,
        status: 0,
        json: async () => ({ detail: "Backend unavailable." }),
      });
    };
    xhr.send(body);
  });
}

export async function getHealth() {
  const response = await request("GET", `${API_BASE_URL}/health`);
  return parseResponse(response);
}

export async function getCrops() {
  const response = await request("GET", `${API_BASE_URL}/crops`);
  const payload = await parseResponse(response);
  return Array.isArray(payload.crops) ? payload.crops : [];
}

export async function predictImage({ crop, images }) {
  if (!crop) throw new Error("Select a crop before analysis.");
  if (!images?.length) throw new Error("Upload at least one RGB crop image.");

  const formData = new FormData();
  formData.append("crop", crop);
  images.forEach((image) => formData.append("image", image));

  const response = await request("POST", `${API_BASE_URL}/predict/image`, formData);
  return parseResponse(response);
}

export async function predictMultimodal({ crop, images, sensorValues, uavZip }) {
  if (!crop) throw new Error("Select a crop before analysis.");
  if (!images?.length) throw new Error("Upload at least one RGB crop image.");

  const formData = new FormData();
  formData.append("crop", crop);
  images.forEach((image) => formData.append("image", image));

  SENSOR_FIELDS.forEach((field) => {
    const value = sensorValues?.[field];
    if (value !== "" && value !== null && value !== undefined) {
      formData.append(field, String(value));
    }
  });

  if (uavZip) {
    formData.append("uav_zip", uavZip);
  }

  const response = await request("POST", `${API_BASE_URL}/predict/multimodal`, formData);
  return parseResponse(response);
}

export function summarizePrediction(result, crop, activeModalities) {
  if (!result) return null;
  const plant = result.plant_statistics || {};
  const fusion = result.fusion || {};
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    timestamp: new Date().toISOString(),
    crop: result.crop || crop,
    prediction: result.prediction || fusion.overall_status || "Unavailable",
    confidence: result.confidence ?? fusion.overall_confidence ?? null,
    activeModalities,
    healthIndex: plant.health_index ?? fusion.overall_health_score ?? null,
    result,
  };
}
