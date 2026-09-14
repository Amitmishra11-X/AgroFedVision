import { RotateCcw } from "lucide-react";
import { SENSOR_FIELDS, formatFieldName } from "../services/api";

export default function SensorForm({ values, onChange, onReset }) {
  return (
    <section className="card p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-ink">Sensor Data</h2>
          <p className="mt-1 text-sm text-slate-500">
            Exact 22 fields accepted by the FastAPI multimodal endpoint.
          </p>
        </div>
        <button
          type="button"
          onClick={onReset}
          className="focus-ring rounded-lg p-2 text-slate-500 hover:bg-slate-100"
          aria-label="Reset sensor data"
          title="Reset sensor data"
        >
          <RotateCcw size={18} aria-hidden="true" />
        </button>
      </div>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {SENSOR_FIELDS.map((field) => (
          <label key={field} className="grid gap-1">
            <span className="text-xs font-semibold text-slate-600">
              {formatFieldName(field)}
            </span>
            <input
              type="number"
              step="any"
              value={values[field] ?? ""}
              onChange={(event) => onChange(field, event.target.value)}
              className="focus-ring rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-ink"
              placeholder={field}
            />
          </label>
        ))}
      </div>
    </section>
  );
}
