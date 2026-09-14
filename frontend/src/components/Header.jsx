import { Play, Server } from "lucide-react";
import { API_BASE_URL } from "../services/api";

export default function Header({ apiStatus, onQuickAnalysis }) {
  const statusLabel =
    apiStatus === "healthy"
      ? "Backend online"
      : apiStatus === "checking"
        ? "Checking backend"
        : "Backend unavailable";

  return (
    <header className="border-b border-slate-200 bg-white/95 px-5 py-4 backdrop-blur">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-xl font-bold text-ink">AGROFEDVISION</h1>
          <p className="mt-1 text-sm text-slate-600">
            Multimodal AI for Real-Time Crop Health Risk Prediction
          </p>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <div className="flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm">
            <Server size={16} aria-hidden="true" />
            <span
              className={`h-2 w-2 rounded-full ${
                apiStatus === "healthy" ? "bg-emerald-500" : "bg-amber-500"
              }`}
            />
            <span className="font-medium text-slate-700">{statusLabel}</span>
            <span className="hidden text-xs text-slate-500 lg:inline">
              {API_BASE_URL}
            </span>
          </div>
          <button
            type="button"
            onClick={onQuickAnalysis}
            className="focus-ring inline-flex items-center justify-center gap-2 rounded-lg bg-canopy px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700"
          >
            <Play size={16} aria-hidden="true" />
            Quick Analysis
          </button>
        </div>
      </div>
    </header>
  );
}
