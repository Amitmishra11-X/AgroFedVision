import { FileArchive, RotateCcw } from "lucide-react";
import { UAV_FIELDS, formatFieldName } from "../services/api";

export default function UAVForm({ uavZip, onZipChange, onReset }) {
  return (
    <section className="card p-5">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-ink">UAV / Multispectral Data</h2>
          <p className="mt-1 text-sm text-slate-500">
            The API accepts a ZIP of complete multispectral TIFF captures.
          </p>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="focus-ring rounded-lg p-2 text-slate-500 hover:bg-slate-100"
          aria-label="Reset UAV upload"
          title="Reset UAV upload"
        >
          <RotateCcw size={18} aria-hidden="true" />
        </button>
      </div>

      <label className="flex min-h-28 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-200 bg-slate-50 p-4 text-center transition hover:border-canopy">
        <FileArchive className="text-canopy" size={30} aria-hidden="true" />
        <span className="mt-2 text-sm font-semibold text-slate-700">
          {uavZip ? uavZip.name : "Upload UAV ZIP"}
        </span>
        <span className="mt-1 text-xs text-slate-500">
          Optional for multimodal analysis
        </span>
        <input
          type="file"
          accept=".zip,application/zip"
          className="sr-only"
          onChange={(event) => onZipChange(event.target.files?.[0] || null)}
          aria-label="Upload UAV ZIP"
        />
      </label>

      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        {UAV_FIELDS.map((field) => (
          <div key={field} className="rounded-lg border border-slate-200 bg-white p-3">
            <p className="metric-label">{formatFieldName(field)}</p>
            <p className="mt-1 text-xs text-slate-500">
              Computed by backend when UAV captures are provided.
            </p>
          </div>
        ))}
      </div>
    </section>
  );
}
