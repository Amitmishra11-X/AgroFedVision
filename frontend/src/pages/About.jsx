import ArchitectureDiagram from "../components/ArchitectureDiagram";

export default function About() {
  return (
    <div className="grid gap-6">
      <section>
        <h2 className="page-title">About / System Architecture</h2>
        <p className="page-subtitle">
          AgroFedVision combines RGB crop imagery, 22 sensor features, and UAV
          multispectral features for crop health risk prediction. The dashboard
          is a presentation and inference layer over the existing backend.
        </p>
      </section>
      <ArchitectureDiagram />
      <section className="card p-5">
        <h2 className="text-lg font-semibold text-ink">Model Interface</h2>
        <dl className="mt-4 grid gap-3 md:grid-cols-3">
          <Info label="Current Model" value="AgroFedVision_Learnable_Gating" />
          <Info label="Image Input" value="224 x 224 x 3" />
          <Info label="Grad-CAM Layer" value="top_conv" />
          <Info label="Sensor Inputs" value="22 backend-defined fields" />
          <Info label="UAV Inputs" value="NDVI/NDRE summary features" />
          <Info label="API" value="FastAPI" />
        </dl>
      </section>
    </div>
  );
}

function Info({ label, value }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <dt className="metric-label">{label}</dt>
      <dd className="mt-2 font-semibold text-ink">{value}</dd>
    </div>
  );
}
