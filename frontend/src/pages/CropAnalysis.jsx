import { useState } from "react";
import ImageUploader from "../components/ImageUploader";
import PredictionCard from "../components/PredictionCard";
import ProbabilityChart from "../components/ProbabilityChart";
import { addHistoryItem } from "../state/history";
import { predictImage, summarizePrediction } from "../services/api";

export default function CropAnalysis({ crops, onResult }) {
  const [crop, setCrop] = useState(crops[0] || "guava");
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function submit(event) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const payload = await predictImage({ crop, images });
      setResult(payload);
      onResult(payload);
      addHistoryItem(summarizePrediction(payload, crop, ["Image"]));
    } catch (err) {
      setError(err.message || "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid gap-6">
      <section>
        <h2 className="page-title">Crop Analysis</h2>
        <p className="page-subtitle">
          Upload RGB crop imagery and run the existing `/predict/image` backend endpoint.
        </p>
      </section>

      <form onSubmit={submit} className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
        <ImageUploader files={images} onFilesChange={setImages} />
        <section className="card p-5">
          <h2 className="text-lg font-semibold text-ink">Analysis Controls</h2>
          <label className="mt-5 grid gap-2">
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
          {error && <p className="mt-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p>}
          <button
            type="submit"
            disabled={loading}
            className="focus-ring mt-5 w-full rounded-lg bg-canopy px-4 py-3 text-sm font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          >
            {loading ? "Analyzing crop..." : "Analyze Crop"}
          </button>
        </section>
      </form>

      {result && (
        <div className="grid gap-6">
          <PredictionCard result={result} />
          <section className="card p-5">
            <h2 className="text-lg font-semibold text-ink">Class Probabilities</h2>
            <ProbabilityChart probabilities={result.class_probabilities} />
          </section>
          <RecommendationPanel result={result} />
        </div>
      )}
    </div>
  );
}

function RecommendationPanel({ result }) {
  const recommendation = result?.recommendation;
  const fusionRecommendations = result?.fusion?.recommendations || [];
  const actions = recommendation?.actions || fusionRecommendations;

  return (
    <section className="card p-5">
      <h2 className="text-lg font-semibold text-ink">Recommendations</h2>
      {recommendation?.summary && <p className="mt-2 text-sm text-slate-600">{recommendation.summary}</p>}
      {actions.length ? (
        <ul className="mt-4 grid gap-2">
          {actions.map((action, index) => (
            <li key={`${action}-${index}`} className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700">
              {action}
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-3 text-sm text-slate-500">No recommendations were returned by the backend.</p>
      )}
    </section>
  );
}
