import { AlertTriangle, CheckCircle2, ShieldAlert } from "lucide-react";

function percent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "Unavailable";
  const numeric = Number(value);
  return `${(numeric <= 1 ? numeric * 100 : numeric).toFixed(2)}%`;
}

function statusStyle(label = "") {
  const normalized = label.toLowerCase();
  if (normalized.includes("high")) return ["bg-red-50 text-red-700 border-red-200", ShieldAlert];
  if (normalized.includes("moderate") || normalized.includes("medium")) {
    return ["bg-amber-50 text-amber-700 border-amber-200", AlertTriangle];
  }
  return ["bg-emerald-50 text-emerald-700 border-emerald-200", CheckCircle2];
}

export default function PredictionCard({ result }) {
  if (!result) {
    return (
      <section className="card p-5">
        <p className="metric-label">Current Prediction</p>
        <p className="mt-3 text-sm text-slate-500">
          Run an analysis to populate this panel with real API results.
        </p>
      </section>
    );
  }

  const plant = result.plant_statistics || {};
  const fusion = result.fusion || {};
  const status = fusion.overall_status || result.prediction || "Unavailable";
  const [className, Icon] = statusStyle(status);

  return (
    <section className="card p-5">
      <p className="metric-label">Crop Health Prediction</p>
      <div className="mt-4 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <div className={`inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-bold ${className}`}>
            <Icon size={18} aria-hidden="true" />
            {status.replaceAll("_", " ")}
          </div>
          <h2 className="mt-4 text-3xl font-bold text-ink">
            {(result.prediction || "Prediction unavailable").replaceAll("_", " ")}
          </h2>
          <p className="mt-2 text-sm text-slate-500">
            Crop: <span className="font-semibold capitalize text-slate-700">{result.crop}</span>
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <Metric label="Confidence" value={percent(result.confidence)} />
          <Metric
            label="Health Index"
            value={
              plant.health_index !== undefined
                ? `${Number(plant.health_index).toFixed(1)}/100`
                : fusion.overall_health_score !== undefined
                  ? `${Number(fusion.overall_health_score).toFixed(1)}/100`
                  : "Unavailable"
            }
          />
          <Metric label="Reliability" value={percent(plant.reliability_score)} />
          <Metric label="Prediction Mode" value={fusion.prediction_mode || "Image"} />
        </div>
      </div>
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="min-w-40 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
      <p className="metric-label">{label}</p>
      <p className="mt-1 text-base font-semibold text-ink">{value}</p>
    </div>
  );
}
