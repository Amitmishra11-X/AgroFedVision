import { Activity, BarChart3, Gauge, Layers3 } from "lucide-react";
import ArchitectureDiagram from "../components/ArchitectureDiagram";
import PredictionCard from "../components/PredictionCard";

function fmtPercent(value) {
  if (value === null || value === undefined) return "Unavailable";
  const numeric = Number(value);
  return `${(numeric <= 1 ? numeric * 100 : numeric).toFixed(1)}%`;
}

export default function Dashboard({ latestResult, onQuickAnalysis }) {
  const plant = latestResult?.plant_statistics || {};
  const fusion = latestResult?.fusion || {};

  return (
    <div className="grid gap-6">
      <section>
        <p className="metric-label">Research Demonstration</p>
        <h2 className="page-title">AgroFedVision Dashboard</h2>
        <p className="page-subtitle">
          A functional inference dashboard connected to the existing FastAPI
          backend for image, sensor, UAV, Grad-CAM, and SHAP outputs.
        </p>
      </section>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <OverviewCard icon={Activity} label="Current Prediction" value={latestResult?.prediction?.replaceAll("_", " ") || "Unavailable"} />
        <OverviewCard icon={BarChart3} label="Confidence" value={fmtPercent(latestResult?.confidence)} />
        <OverviewCard
          icon={Gauge}
          label="Health Index"
          value={
            plant.health_index !== undefined
              ? `${Number(plant.health_index).toFixed(1)}/100`
              : fusion.overall_health_score !== undefined
                ? `${Number(fusion.overall_health_score).toFixed(1)}/100`
                : "Unavailable"
          }
        />
        <OverviewCard icon={Layers3} label="Active Modalities" value={fusion.prediction_mode || "Unavailable"} />
      </div>

      <PredictionCard result={latestResult} />
      <ArchitectureDiagram />

      <button
        type="button"
        onClick={onQuickAnalysis}
        className="focus-ring w-fit rounded-lg bg-canopy px-5 py-3 text-sm font-semibold text-white hover:bg-emerald-700"
      >
        Start Quick Analysis
      </button>
    </div>
  );
}

function OverviewCard({ icon: Icon, label, value }) {
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between">
        <p className="metric-label">{label}</p>
        <Icon size={18} className="text-canopy" aria-hidden="true" />
      </div>
      <p className="mt-4 min-h-8 text-xl font-bold text-ink">{value}</p>
    </div>
  );
}
