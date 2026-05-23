import { Trash2, Plus } from "lucide-react";
import type { ImageSummary } from "../lib/api";

interface Props {
  images: ImageSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onUpload: () => void;
  orientation?: "vertical" | "horizontal";
}

export function ImageGallery({
  images, activeId, onSelect, onDelete, onUpload,
  orientation = "vertical",
}: Props) {
  if (orientation === "horizontal") {
    return (
      <div className="flex items-center gap-2 overflow-x-auto scrollbar-thin pb-1">
        {images.length === 0 ? (
          <button
            onClick={onUpload}
            className="flex-shrink-0 px-4 py-2 rounded border border-dashed border-white/10 hover:border-elastic-teal hover:bg-white/5 transition flex items-center gap-2 text-elastic-gray hover:text-white text-xs"
          >
            <Plus className="w-3.5 h-3.5" />
            Upload your first image
          </button>
        ) : (
          <>
            {images.map((img) => {
              const active = img.image_id === activeId;
              return (
                <div
                  key={img.image_id}
                  onClick={() => onSelect(img.image_id)}
                  className={`group flex-shrink-0 rounded cursor-pointer transition border overflow-hidden relative ${
                    active
                      ? "border-elastic-teal ring-1 ring-elastic-teal/40"
                      : "border-white/10 hover:border-white/30"
                  }`}
                  title={`${img.label ?? img.image_id}\n${img.tile_count} tiles · ${img.width}×${img.height}`}
                >
                  <img
                    src={img.image_url}
                    alt=""
                    className={`h-20 w-32 object-cover transition ${
                      active ? "opacity-100" : "opacity-70 group-hover:opacity-100"
                    }`}
                  />
                  {active && (
                    <div className="absolute top-1.5 left-1.5 w-2 h-2 rounded-full bg-elastic-teal animate-pulse" />
                  )}
                  <button
                    onClick={(e) => { e.stopPropagation(); onDelete(img.image_id); }}
                    className="absolute top-1.5 right-1.5 opacity-0 group-hover:opacity-100 text-white bg-black/60 hover:bg-elastic-pink/80 p-1 rounded transition"
                    title="Delete"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                  <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent px-2 pt-4 pb-1.5">
                    <div className="text-[11px] font-mono text-white/95 truncate">
                      {img.label ?? img.image_id}
                    </div>
                  </div>
                </div>
              );
            })}
            <button
              onClick={onUpload}
              className="flex-shrink-0 h-20 w-20 rounded border border-dashed border-white/10 hover:border-elastic-teal hover:bg-white/5 transition flex items-center justify-center text-elastic-gray hover:text-white"
              title="Upload another image"
            >
              <Plus className="w-5 h-5" />
            </button>
          </>
        )}
      </div>
    );
  }

  // ─── Vertical fallback ─────────────────────────────────────────────
  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin">
      {images.length === 0 ? (
        <button
          onClick={onUpload}
          className="m-3 p-4 rounded border-2 border-dashed border-white/10 hover:border-elastic-teal hover:bg-white/5 transition w-[calc(100%-1.5rem)] flex flex-col items-center gap-2 text-elastic-gray hover:text-white"
        >
          <Plus className="w-5 h-5" />
          <span className="text-xs">Upload image</span>
        </button>
      ) : (
        <div className="p-2 space-y-1">
          {images.map((img) => {
            const active = img.image_id === activeId;
            return (
              <div
                key={img.image_id}
                onClick={() => onSelect(img.image_id)}
                className={`group rounded cursor-pointer transition border overflow-hidden ${
                  active
                    ? "border-elastic-teal bg-elastic-teal/5"
                    : "border-transparent hover:bg-white/[0.04]"
                }`}
              >
                <div className="relative aspect-video bg-elastic-ink/60 overflow-hidden">
                  <img
                    src={img.image_url}
                    alt=""
                    className="w-full h-full object-cover opacity-90 group-hover:opacity-100"
                  />
                  {active && (
                    <div className="absolute top-1.5 left-1.5 w-1.5 h-1.5 rounded-full bg-elastic-teal animate-pulse" />
                  )}
                </div>
                <div className="px-2.5 py-2 flex items-center justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="text-[11px] font-mono truncate">{img.image_id}</div>
                    <div className="text-[10px] text-elastic-gray font-mono">
                      {img.tile_count} tiles · {img.width}×{img.height}
                    </div>
                  </div>
                  <button
                    onClick={(e) => { e.stopPropagation(); onDelete(img.image_id); }}
                    className="opacity-0 group-hover:opacity-100 text-elastic-gray hover:text-elastic-pink p-1 transition"
                    title="Delete"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
