import { resolveAssetUrl } from "../services/api";

export default function GradCAMViewer({ result }) {
  const gradcam = result?.gradcam;
  const imageUrl = resolveAssetUrl(result?.image_path);
  const gradcamUrl = resolveAssetUrl(typeof gradcam === "string" ? gradcam : gradcam?.path);

  return (
    <section className="card p-5">
      <div className="mb-5">
        <p className="metric-label">Grad-CAM</p>
        <h2 className="mt-2 text-xl font-semibold text-ink">Spatial Image Evidence</h2>
        <p className="mt-2 text-sm text-slate-600">
          Grad-CAM highlights image regions that contributed most strongly to
          the model's prediction.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <ImagePanel title="Original Image" url={imageUrl} />
        <ImagePanel title="Grad-CAM Overlay" url={gradcamUrl} empty="Grad-CAM unavailable for this prediction." />
      </div>

      <dl className="mt-5 grid gap-3 sm:grid-cols-2">
        <Meta label="Target Layer" value={gradcam?.target_layer || "top_conv"} />
        <Meta label="Predicted Class Index" value={gradcam?.class_index ?? "Unavailable"} />
      </dl>
    </section>
  );
}

function ImagePanel({ title, url, empty = "Image unavailable." }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
      <p className="mb-3 text-sm font-semibold text-slate-700">{title}</p>
      {url ? (
        <img src={url} alt={title} className="h-72 w-full rounded-md object-contain bg-white" />
      ) : (
        <div className="flex h-72 items-center justify-center rounded-md border border-dashed border-slate-200 bg-white p-4 text-center text-sm text-slate-500">
          {empty}
        </div>
      )}
    </div>
  );
}

function Meta({ label, value }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <dt className="metric-label">{label}</dt>
      <dd className="mt-1 font-semibold text-ink">{value}</dd>
    </div>
  );
}
