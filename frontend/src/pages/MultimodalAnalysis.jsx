import { useMemo, useState } from "react";
import ImageUploader from "../components/ImageUploader";
import PredictionCard from "../components/PredictionCard";
import ProbabilityChart from "../components/ProbabilityChart";
import SensorForm from "../components/SensorForm";
import UAVForm from "../components/UAVForm";
import { SENSOR_FIELDS, predictMultimodal, summarizePrediction } from "../services/api";
import { addHistoryItem } from "../state/history";

const emptySensor = Object.fromEntries(SENSOR_FIELDS.map((field) => [field, ""]));

export default function MultimodalAnalysis({ crops, onResult }) {
  const [crop, setCrop] = useState(crops[0] || "guava");
  const [images, setImages] = useState([]);
  const [sensorValues, setSensorValues] = useState(emptySensor);
  const [uavZip, setUavZip] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const activeModalities = useMemo(() => {
    const modalities = images.length ? ["Image"] : [];
    if (Object.values(sensorValues).some((value) => value !== "")) modalities.push("Sensor");
    if (uavZip) modalities.push("UAV");
    return modalities;
  }, [images, sensorValues, uavZip]);

  function updateSensor(field, value) {
    setSensorValues((current) => ({ ...current, [field]: value }));
  }

  async function submit(event) {
    event.preventDefault();
    setError("");
    const providedSensorFields = SENSOR_FIELDS.filter((field) => sensorValues[field] !== "");
    if (providedSensorFields.length > 0 && providedSensorFields.length < SENSOR_FIELDS.length) {
      setError(
        "The backend sensor model requires all 22 sensor fields when sensor data is used."
      );
      return;
    }
    setLoading(true);
    try {
      const payload = await predictMultimodal({ crop, images, sensorValues, uavZip });
      setResult(payload);
      onResult(payload);
      addHistoryItem(summarizePrediction(payload, crop, activeModalities.length ? activeModalities : ["Image"]));
    } catch (err) {
      setError(err.message || "Multimodal prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={submit} className="grid gap-6">
      <section>
        <h2 className="page-title">Multimodal Analysis</h2>
        <p className="page-subtitle">
          Run `/predict/multimodal` with RGB image input, optional sensor readings,
          and optional UAV ZIP captures. Missing optional modalities are shown honestly.
        </p>
      </section>

      <section className="card p-5">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <label className="grid gap-2 md:min-w-64">
            <span className="text-sm font-semibold text-slate-700">Crop</span>
            <select
              value={crop}
              onChange={(event) => setCrop(event.target.value)}
              className="focus-ring rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm capitalize"
            >
              {crops.map((item) => (
                <option key={item} value={item}>
                  {item === "maize" ? "Maize (Corn)" : item}
                </option>
              ))}
            </select>
          </label>
          <div className="flex flex-wrap gap-2">
            {["Image", "Sensor", "UAV"].map((item) => (
              <span
                key={item}
                className={`rounded-lg border px-3 py-2 text-sm font-semibold ${
                  activeModalities.includes(item)
                    ? "border-emerald-200 bg-emerald-50 text-canopy"
                    : "border-slate-200 bg-slate-50 text-slate-500"
                }`}
              >
                {item} {activeModalities.includes(item) ? "active" : "inactive"}
              </span>
            ))}
          </div>
        </div>
        {activeModalities.length === 1 && activeModalities[0] === "Image" && (
          <p className="mt-4 rounded-lg bg-slate-50 p-3 text-sm text-slate-600">Image-only mode</p>
        )}
      </section>

      <div className="grid gap-6">
        <ImageUploader files={images} onFilesChange={setImages} label="RGB Image" />
        <SensorForm values={sensorValues} onChange={updateSensor} onReset={() => setSensorValues(emptySensor)} />
        <UAVForm uavZip={uavZip} onZipChange={setUavZip} onReset={() => setUavZip(null)} />
      </div>

      {error && <p className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p>}
      <button
        type="submit"
        disabled={loading}
        className="focus-ring w-fit rounded-lg bg-canopy px-5 py-3 text-sm font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-400"
      >
        {loading ? "Uploading and analyzing..." : "Run Multimodal Analysis"}
      </button>

      {result && (
        <div className="grid gap-6">
          <PredictionCard result={result} />
          <section className="card p-5">
            <h2 className="text-lg font-semibold text-ink">Probability Distribution</h2>
            <ProbabilityChart probabilities={result.class_probabilities} />
          </section>
        </div>
      )}
    </form>
  );
}
