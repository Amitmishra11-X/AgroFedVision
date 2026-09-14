import GradCAMViewer from "../components/GradCAMViewer";
import SHAPChart from "../components/SHAPChart";

export default function Explainability({ latestResult }) {
  return (
    <div className="grid gap-6">
      <section>
        <h2 className="page-title">Explainable AI</h2>
        <p className="page-subtitle">
          Visualizes real explainability outputs returned by the backend. SHAP
          remains unavailable until the prediction response includes it.
        </p>
      </section>
      <GradCAMViewer result={latestResult} />
      <SHAPChart result={latestResult} />
    </div>
  );
}
