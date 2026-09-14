import { Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { clearHistory, loadHistory } from "../state/history";

export default function History({ onSelectResult }) {
  const [items, setItems] = useState([]);

  useEffect(() => {
    setItems(loadHistory());
  }, []);

  function clear() {
    clearHistory();
    setItems([]);
  }

  return (
    <div className="grid gap-6">
      <section className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div>
          <h2 className="page-title">Prediction History</h2>
          <p className="page-subtitle">
            Client-side history stored in browser localStorage. No database was added.
          </p>
        </div>
        <button
          type="button"
          onClick={clear}
          className="focus-ring inline-flex w-fit items-center gap-2 rounded-lg border border-red-200 px-4 py-2 text-sm font-semibold text-red-700 hover:bg-red-50"
        >
          <Trash2 size={16} aria-hidden="true" />
          Clear History
        </button>
      </section>

      <div className="grid gap-3">
        {items.length ? (
          items.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelectResult(item.result)}
              className="focus-ring card p-4 text-left transition hover:border-canopy"
            >
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <p className="font-semibold text-ink">{item.prediction.replaceAll("_", " ")}</p>
                  <p className="mt-1 text-sm text-slate-500">
                    {new Date(item.timestamp).toLocaleString()} - {item.crop}
                  </p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {(item.activeModalities || []).map((modality) => (
                    <span key={modality} className="rounded-lg bg-emerald-50 px-3 py-1 text-xs font-semibold text-canopy">
                      {modality}
                    </span>
                  ))}
                </div>
              </div>
            </button>
          ))
        ) : (
          <div className="card p-6 text-sm text-slate-500">No predictions have been stored yet.</div>
        )}
      </div>
    </div>
  );
}
