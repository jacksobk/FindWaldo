import { useRef, useState } from "react";
import { Upload, X, Loader2, CheckCircle2 } from "lucide-react";
import { api, type IngestResponse } from "../lib/api";

interface Props {
  onClose: () => void;
  onUploaded: (resp: IngestResponse) => void;
}

export function UploadDialog({ onClose, onUploaded }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [label, setLabel] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<IngestResponse | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleUpload() {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const resp = await api.ingest(file, label || undefined);
      setResult(resp);
      // Brief success display, then close
      setTimeout(() => onUploaded(resp), 1100);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-6"
      onClick={onClose}
    >
      <div
        className="bg-elastic-slate border border-white/10 rounded-lg w-full max-w-lg shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="px-5 py-3 border-b border-white/5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Upload className="w-4 h-4 text-elastic-teal" />
            <span className="text-sm font-semibold">Ingest a new image</span>
          </div>
          <button onClick={onClose} className="text-elastic-gray hover:text-white">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          {result ? (
            <div className="text-center py-4 space-y-2">
              <CheckCircle2 className="w-10 h-10 text-elastic-teal mx-auto" />
              <div className="text-sm">Indexed successfully</div>
              <div className="font-mono text-xs text-elastic-gray space-y-0.5">
                <div>image_id: {result.image_id}</div>
                <div>{result.tiles_indexed} tiles · {result.width}×{result.height}px</div>
                <div>embed: {result.inference_ms}ms · index: {result.indexing_ms}ms</div>
              </div>
            </div>
          ) : (
            <>
              {/* Drop zone */}
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  const f = e.dataTransfer.files[0];
                  if (f && f.type.startsWith("image/")) setFile(f);
                }}
                onClick={() => inputRef.current?.click()}
                className={`border-2 border-dashed rounded p-6 text-center cursor-pointer transition ${
                  dragOver
                    ? "border-elastic-teal bg-elastic-teal/5"
                    : "border-white/10 hover:border-white/20 hover:bg-white/[0.02]"
                }`}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                />
                <Upload className="w-6 h-6 mx-auto text-elastic-gray mb-2" />
                {file ? (
                  <div>
                    <div className="text-sm font-mono">{file.name}</div>
                    <div className="text-xs text-elastic-gray mt-0.5">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </div>
                  </div>
                ) : (
                  <div>
                    <div className="text-sm">Drop an image here or click to browse</div>
                    <div className="text-xs text-elastic-gray mt-1">JPG, PNG · max 25 MB</div>
                  </div>
                )}
              </div>

              {/* Optional caption */}
              <div>
                <label className="text-[10px] font-mono uppercase tracking-widest text-elastic-gray block mb-1.5">
                  Optional caption (used for hybrid search)
                </label>
                <input
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                  placeholder="e.g. Where's Waldo at the beach"
                  className="w-full bg-elastic-ink border border-white/10 rounded px-3 py-2 text-sm focus:border-elastic-teal focus:outline-none"
                />
              </div>

              {error && (
                <div className="text-xs text-elastic-pink bg-elastic-pink/10 border border-elastic-pink/30 rounded px-3 py-2">
                  {error}
                </div>
              )}

              <div className="flex justify-end gap-2 pt-2">
                <button
                  onClick={onClose}
                  className="px-3 py-2 text-sm rounded border border-white/10 hover:bg-white/5 transition"
                >
                  Cancel
                </button>
                <button
                  onClick={handleUpload}
                  disabled={!file || uploading}
                  className="px-4 py-2 text-sm rounded bg-elastic-teal text-elastic-ink font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-elastic-teal/90 flex items-center gap-2 transition"
                >
                  {uploading ? (
                    <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Tiling + embedding…</>
                  ) : (
                    "Ingest"
                  )}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
