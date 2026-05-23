import type { SearchHit } from "../lib/api";
import { Loader2 } from "lucide-react";

interface Props {
  hits: SearchHit[];
  searching: boolean;
}

export function ResultList({ hits, searching }: Props) {
  if (searching && hits.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-elastic-gray text-sm gap-3">
        <Loader2 className="w-5 h-5 animate-spin text-elastic-teal" />
        <div className="text-xs font-mono uppercase tracking-widest">Embedding query · running kNN</div>
      </div>
    );
  }

  if (hits.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-elastic-gray text-sm font-display italic px-6 text-center">
        Enter a query to see ranked tile matches.
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto scrollbar-thin">
      <div className="px-5 py-3 text-xs font-mono uppercase tracking-widest text-elastic-gray">
        Top-{hits.length} ranked tiles
      </div>
      <div className="px-3 pb-4 space-y-2">
        {hits.map((hit) => (
          <div
            key={hit.tile_id}
            className="px-3 py-3 rounded bg-elastic-ink/50 border border-white/5 hover:border-elastic-teal/40 transition group"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2.5">
                <span
                  className="w-7 h-7 rounded flex items-center justify-center text-sm font-mono font-bold"
                  style={{
                    background: hit.rank === 1 ? "#00BFB3" : "rgba(254, 197, 20, 0.85)",
                    color: "#0B1628",
                  }}
                >
                  {hit.rank}
                </span>
                <span className="text-xs font-mono text-elastic-gray">
                  bbox {hit.bbox.x},{hit.bbox.y}
                </span>
              </div>
              <span className="text-sm font-mono text-white">
                {hit.score.toFixed(3)}
              </span>
            </div>
            {/* Score bar */}
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: `${Math.min(100, hit.score * 100)}%`,
                  background: hit.rank === 1
                    ? "linear-gradient(90deg, #00BFB3, #FEC514)"
                    : "rgba(254, 197, 20, 0.6)",
                }}
              />
            </div>
            <div className="mt-2 text-[11px] font-mono text-elastic-gray/60 truncate">
              {hit.tile_id}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
