import { ImagePlus, Trash2, UploadCloud } from "lucide-react";
import { useMemo, useRef } from "react";

export default function ImageUploader({ files, onFilesChange, label = "RGB crop/leaf image" }) {
  const inputRef = useRef(null);
  const previews = useMemo(
    () =>
      files.map((file) => ({
        name: file.name,
        url: URL.createObjectURL(file),
      })),
    [files]
  );

  function addFiles(fileList) {
    const accepted = Array.from(fileList || []).filter((file) =>
      file.type.startsWith("image/")
    );
    onFilesChange([...files, ...accepted]);
  }

  function handleDrop(event) {
    event.preventDefault();
    addFiles(event.dataTransfer.files);
  }

  function removeAt(index) {
    onFilesChange(files.filter((_, current) => current !== index));
  }

  return (
    <section className="card p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-ink">{label}</h2>
          <p className="mt-1 text-sm text-slate-500">
            Upload JPG, PNG, WEBP, or BMP files accepted by the API.
          </p>
        </div>
        <ImagePlus className="text-canopy" size={22} aria-hidden="true" />
      </div>

      <div
        onDragOver={(event) => event.preventDefault()}
        onDrop={handleDrop}
        className="flex min-h-48 flex-col items-center justify-center rounded-lg border-2 border-dashed border-emerald-200 bg-emerald-50/50 p-6 text-center transition hover:border-canopy"
      >
        <UploadCloud className="text-canopy" size={34} aria-hidden="true" />
        <p className="mt-3 text-sm font-semibold text-slate-700">
          Drag and drop images here
        </p>
        <p className="mt-1 text-xs text-slate-500">or browse from this device</p>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="focus-ring mt-4 rounded-lg border border-canopy bg-white px-4 py-2 text-sm font-semibold text-canopy transition hover:bg-emerald-50"
        >
          Browse Images
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          className="sr-only"
          onChange={(event) => addFiles(event.target.files)}
          aria-label="Upload RGB crop image"
        />
      </div>

      {previews.length > 0 && (
        <div className="mt-5 grid gap-3 sm:grid-cols-2">
          {previews.map((preview, index) => (
            <div key={`${preview.name}-${index}`} className="relative overflow-hidden rounded-lg border border-slate-200">
              <img
                src={preview.url}
                alt={`Uploaded crop sample ${index + 1}`}
                className="h-44 w-full object-cover"
              />
              <div className="flex items-center justify-between gap-2 bg-white px-3 py-2">
                <span className="truncate text-xs font-medium text-slate-600">
                  {preview.name}
                </span>
                <button
                  type="button"
                  onClick={() => removeAt(index)}
                  className="focus-ring rounded-md p-1.5 text-slate-500 hover:bg-red-50 hover:text-red-600"
                  aria-label={`Remove ${preview.name}`}
                >
                  <Trash2 size={16} aria-hidden="true" />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
