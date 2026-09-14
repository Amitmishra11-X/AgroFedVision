const nodes = [
  ["RGB Image", "Image Encoder"],
  ["Sensor", "Transformer"],
  ["UAV", "Feature Extraction"],
];

export default function ArchitectureDiagram() {
  return (
    <section className="card p-5">
      <p className="metric-label">System Architecture</p>
      <h2 className="mt-2 text-xl font-semibold text-ink">Multimodal Inference Flow</h2>
      <div className="mt-6 grid gap-4 lg:grid-cols-[1fr_auto_1fr_auto_1fr] lg:items-center">
        <div className="grid gap-3">
          {nodes.map(([source, stage]) => (
            <div key={source} className="rounded-lg border border-slate-200 bg-slate-50 p-4">
              <p className="text-sm font-bold text-ink">{source}</p>
              <div className="my-2 h-6 w-px bg-slate-300" aria-hidden="true" />
              <p className="text-sm text-slate-600">{stage}</p>
            </div>
          ))}
        </div>
        <Arrow />
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-5 text-center">
          <p className="text-sm font-bold text-canopy">Multimodal Fusion</p>
          <p className="mt-2 text-xs leading-5 text-emerald-800">
            Existing learnable gating fusion model
          </p>
        </div>
        <Arrow />
        <div className="grid gap-3">
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <p className="text-sm font-bold text-ink">Health Prediction</p>
            <p className="mt-2 text-xs text-slate-500">Class, confidence, probabilities</p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <p className="text-sm font-bold text-ink">Explainable AI</p>
            <p className="mt-2 text-xs text-slate-500">Grad-CAM and SHAP when available</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function Arrow() {
  return <div className="hidden h-px w-10 bg-slate-300 lg:block" aria-hidden="true" />;
}
