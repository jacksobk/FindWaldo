import { useEffect, useMemo, useState } from "react";
import {
  Search as SearchIcon, Upload, Activity, Layers, Zap, Hash, X,
  Presentation, AppWindow, Sliders, ChevronDown, ChevronUp,
} from "lucide-react";
import { api, type ImageSummary, type SearchHit, type SearchResponse } from "./lib/api";
import { ImageCanvas } from "./components/ImageCanvas";
import { ResultList } from "./components/ResultList";
import { ImageGallery } from "./components/ImageGallery";
import { UploadDialog } from "./components/UploadDialog";
import { ClusterStatus } from "./components/ClusterStatus";
import { DemoMode } from "./components/DemoMode";

type AppMode = "app" | "demo";

const SUGGESTED_QUERIES = [
  "Wizard Whitebeard with white robes",
  "someone falling off a horse",
  "a striped beach umbrella",
  "a pink dragon",
  "a man with a red and white striped hat",
];

export default function App() {
  const [mode, setMode] = useState<AppMode>("app");
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [activeImageId, setActiveImageId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [k, setK] = useState(5);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [searchMeta, setSearchMeta] = useState<Pick<SearchResponse, "inference_ms" | "search_ms" | "total_candidates" | "rerank_ms" | "reranked" | "corrected_query" | "corrections"> | null>(null);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [scopeToImage, setScopeToImage] = useState(true);
  const [resultsOpen, setResultsOpen] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const activeImage = useMemo(
    () => images.find((i) => i.image_id === activeImageId) ?? null,
    [images, activeImageId],
  );

  useEffect(() => { refreshImages(); }, []);

  async function refreshImages() {
    try {
      const list = await api.listImages();
      setImages(list);
      if (!activeImageId && list.length > 0) setActiveImageId(list[0].image_id);
    } catch (e: any) {
      setError(e.message);
    }
  }

  async function runSearch(q: string) {
    if (!q.trim()) return;
    setSearching(true);
    setError(null);
    setHits([]);
    setSearchMeta(null);
    setResultsOpen(true);
    try {
      const resp = await api.search({
        query: q,
        image_id: scopeToImage && activeImageId ? activeImageId : undefined,
        k,
        num_candidates: 200,
      });
      setHits(resp.hits);
      setSearchMeta({
        inference_ms: resp.inference_ms,
        search_ms: resp.search_ms,
        total_candidates: resp.total_candidates,
        rerank_ms: resp.rerank_ms,
        reranked: resp.reranked,
        corrected_query: resp.corrected_query,
        corrections: resp.corrections,
      });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setSearching(false);
    }
  }

  // Reset everything related to a search: query text, results, telemetry,
  // bounding box overlays, error, and the results panel itself. Used by the
  // X icon in the search input and by the Escape key.
  function clearSearch() {
    setQuery("");
    setHits([]);
    setSearchMeta(null);
    setError(null);
    setResultsOpen(false);
  }

  // Escape key clears the search from anywhere on the page. Skip when the
  // user is typing in a different input (e.g. upload dialog) so we don't
  // disrupt that interaction.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      const tag = (e.target as HTMLElement)?.tagName;
      const isOurSearchInput =
        tag === "INPUT" &&
        (e.target as HTMLInputElement).placeholder?.startsWith("Describe");
      // Allow Escape to clear from our search input, or from anywhere when
      // not in a different input.
      if (isOurSearchInput || tag !== "INPUT") {
        if (query || hits.length > 0 || resultsOpen) {
          clearSearch();
        }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [query, hits.length, resultsOpen]);

  async function handleDelete(imageId: string) {
    if (!confirm("Delete this image and all its tiles?")) return;
    await api.deleteImage(imageId);
    if (activeImageId === imageId) setActiveImageId(null);
    setHits([]);
    refreshImages();
  }

  async function handleUploaded() {
    setUploadOpen(false);
    await refreshImages();
  }

  if (mode === "demo") {
    return <DemoMode onExit={() => setMode("app")} />;
  }

  return (
    <div className="h-screen overflow-hidden bg-elastic-ink text-white flex flex-col">

      {/* ────────────── Top header ────────────── */}
      <header className="border-b border-white/5 bg-elastic-slate/60 backdrop-blur z-20">
        <div className="px-6 py-3 flex items-center justify-between gap-6">
          <div className="flex items-center gap-5 flex-shrink-0">
            <div className="flex items-center gap-2.5">
              <div className="w-10 h-10 rounded bg-gradient-to-br from-elastic-teal to-elastic-pink flex items-center justify-center">
                <SearchIcon className="w-5 h-5 text-elastic-ink" strokeWidth={2.5} />
              </div>
              <div>
                <div className="text-base font-semibold tracking-tight leading-tight">Find Waldo</div>
                <div className="text-xs text-elastic-gray font-mono uppercase tracking-wider leading-tight mt-0.5">
                  Multimodal · Elastic 9.4
                </div>
              </div>
            </div>
            <ClusterStatus />
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={() => setUploadOpen(true)}
              className="px-4 py-2 rounded text-sm font-medium bg-white/5 hover:bg-white/10 border border-white/10 flex items-center gap-2 transition"
            >
              <Upload className="w-4 h-4" /> Upload image
            </button>
            <button
              onClick={() => setMode("demo")}
              className="px-4 py-2 rounded text-sm font-medium bg-elastic-teal text-elastic-ink hover:bg-elastic-teal/90 flex items-center gap-2 transition"
            >
              <Presentation className="w-4 h-4" /> Demo mode
            </button>
          </div>
        </div>

        {/* ────────────── Search bar — full width ────────────── */}
        <div className="px-6 pb-4">
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <SearchIcon className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-elastic-gray pointer-events-none" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && runSearch(query)}
                placeholder="Describe what you're looking for, in any language…"
                className="w-full bg-elastic-ink border border-white/10 focus:border-elastic-teal rounded pl-12 pr-12 py-3.5 text-base font-display italic placeholder:text-white/30 focus:outline-none transition"
              />
              {/* Clear button — visible only when there's text or active results */}
              {(query || hits.length > 0 || searchMeta) && (
                <button
                  onClick={clearSearch}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1.5 rounded hover:bg-white/10 text-elastic-gray hover:text-white transition"
                  title="Clear search (Esc)"
                  aria-label="Clear search"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>

            <button
              onClick={() => runSearch(query)}
              disabled={!query.trim() || searching}
              className="px-6 py-3.5 rounded bg-elastic-teal text-elastic-ink text-base font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:bg-elastic-teal/90 flex items-center justify-center gap-2 transition flex-shrink-0"
            >
              {searching ? (
                <><Activity className="w-4 h-4 animate-pulse" /> Searching</>
              ) : (
                <><SearchIcon className="w-4 h-4" /> Search</>
              )}
            </button>

            <button
              onClick={() => setAdvancedOpen(!advancedOpen)}
              className="p-3.5 rounded border border-white/10 bg-white/5 hover:bg-white/10 transition flex-shrink-0"
              title="Advanced settings"
            >
              <Sliders className="w-5 h-5" />
            </button>
          </div>

          {/* Suggested chips */}
          <div className="flex flex-wrap gap-2 mt-3">
            {SUGGESTED_QUERIES.map((s) => (
              <button
                key={s}
                onClick={() => { setQuery(s); runSearch(s); }}
                className="text-sm px-3 py-1.5 rounded border border-white/10 bg-white/[0.03] hover:bg-white/10 hover:border-elastic-teal/50 transition text-elastic-gray hover:text-white"
              >
                {s}
              </button>
            ))}
          </div>

          {/* Spell-correction notice — shown when the backend rewrote a query */}
          {searchMeta?.corrected_query && searchMeta.corrections && searchMeta.corrections.length > 0 && (
            <div className="mt-3 px-3 py-2 rounded border border-elastic-teal/30 bg-elastic-teal/5 text-xs text-elastic-gray flex items-center gap-2">
              <span className="text-elastic-teal font-mono uppercase tracking-wider">Corrected:</span>
              <span>
                Searched for <span className="text-white font-medium">“{searchMeta.corrected_query}”</span>{" "}
                <span className="text-elastic-gray">(autocorrected from</span>{" "}
                {searchMeta.corrections.map(([orig, fixed], i) => (
                  <span key={i}>
                    {i > 0 && <span className="text-elastic-gray">, </span>}
                    <span className="text-white/70 line-through">{orig}</span>
                    {" → "}
                    <span className="text-elastic-teal">{fixed}</span>
                  </span>
                ))}
                <span className="text-elastic-gray">)</span>
              </span>
            </div>
          )}

          {/* Advanced controls (collapsible) */}
          {advancedOpen && (
            <div className="mt-3 px-4 py-3 rounded border border-white/10 bg-elastic-ink/60 flex items-center gap-6 text-sm">
              <label className="flex items-center gap-2 text-elastic-gray">
                <input
                  type="checkbox"
                  checked={scopeToImage}
                  onChange={(e) => setScopeToImage(e.target.checked)}
                  className="w-4 h-4 accent-elastic-teal"
                />
                Scope to active image
              </label>
              <div className="flex items-center gap-2">
                <span className="text-elastic-gray">Top-K results:</span>
                <select
                  value={k}
                  onChange={(e) => setK(Number(e.target.value))}
                  className="bg-elastic-ink border border-white/10 rounded px-3 py-1 text-sm"
                >
                  {[3, 5, 10, 20].map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>

              {/* Live telemetry */}
              {searchMeta && (
                <div className="ml-auto flex items-center gap-4 text-elastic-gray font-mono text-xs">
                  <span className="flex items-center gap-1">
                    <Zap className="w-3.5 h-3.5 text-elastic-teal" /> Inference {searchMeta.inference_ms}ms
                  </span>
                  <span className="flex items-center gap-1">
                    <Layers className="w-3.5 h-3.5 text-elastic-teal" /> kNN {searchMeta.search_ms}ms
                  </span>
                  {searchMeta.reranked && (
                    <span className="flex items-center gap-1" title="Cross-encoder rerank latency">
                      <Sliders className="w-3.5 h-3.5 text-elastic-pink" /> Rerank {searchMeta.rerank_ms}ms
                    </span>
                  )}
                  <span className="flex items-center gap-1">
                    <Hash className="w-3.5 h-3.5 text-elastic-teal" /> {hits.length} hits
                  </span>
                </div>
              )}
            </div>
          )}
        </div>
      </header>

      {/* ────────────── Image gallery — horizontal strip ────────────── */}
      <div className="border-b border-white/5 bg-elastic-slate/30 flex-shrink-0">
        <div className="px-6 py-3 flex items-center gap-4">
          <div className="text-xs font-mono uppercase tracking-widest text-elastic-gray flex-shrink-0">
            Indexed images <span className="text-white/50">({images.length})</span>
          </div>
          <div className="flex-1 overflow-x-auto">
            <ImageGallery
              images={images}
              activeId={activeImageId}
              onSelect={setActiveImageId}
              onDelete={handleDelete}
              onUpload={() => setUploadOpen(true)}
              orientation="horizontal"
            />
          </div>
        </div>
      </div>

      {/* ────────────── Body: canvas + optional results panel ────────────── */}
      <div className="flex-1 flex overflow-hidden relative">

        {/* Canvas — full width minus optional results sidebar */}
        <main className="flex-1 flex flex-col bg-elastic-ink overflow-hidden">
          {activeImage ? (
            <ImageCanvas
              image={activeImage}
              hits={hits.filter((h) => h.image_id === activeImage.image_id)}
              searching={searching}
            />
          ) : (
            <EmptyState onUpload={() => setUploadOpen(true)} />
          )}
        </main>

        {/* Results panel — slides in only after a search */}
        {resultsOpen && (
          <aside className="w-80 border-l border-white/5 bg-elastic-slate/40 flex flex-col flex-shrink-0">
            <div className="px-4 py-2.5 border-b border-white/5 flex items-center justify-between">
              <div className="text-[10px] font-mono uppercase tracking-widest text-elastic-gray">
                Results {searchMeta && `· ${hits.length}`}
              </div>
              <button
                onClick={() => setResultsOpen(false)}
                className="p-1 rounded hover:bg-white/5 text-elastic-gray hover:text-white transition"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>

            {error && (
              <div className="mx-4 my-3 px-3 py-2 rounded bg-elastic-pink/10 border border-elastic-pink/30 text-xs text-elastic-pink flex items-start gap-2">
                <X className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                <span className="break-all">{error}</span>
              </div>
            )}

            <ResultList hits={hits} searching={searching} />
          </aside>
        )}

        {/* Floating "show results" tab when collapsed */}
        {!resultsOpen && hits.length > 0 && (
          <button
            onClick={() => setResultsOpen(true)}
            className="absolute right-0 top-20 px-3 py-2 rounded-l bg-elastic-teal text-elastic-ink text-sm font-semibold shadow-lg flex items-center gap-2 z-10"
          >
            <Hash className="w-3.5 h-3.5" /> {hits.length} results
          </button>
        )}
      </div>

      {uploadOpen && (
        <UploadDialog onClose={() => setUploadOpen(false)} onUploaded={handleUploaded} />
      )}
    </div>
  );
}

function EmptyState({ onUpload }: { onUpload: () => void }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center text-center px-8">
      <div className="w-16 h-16 rounded-full border border-white/10 flex items-center justify-center mb-4">
        <AppWindow className="w-7 h-7 text-elastic-gray" />
      </div>
      <h2 className="font-display text-2xl mb-2">No image selected</h2>
      <p className="text-sm text-elastic-gray max-w-sm mb-5">
        Pick an image from the gallery above, or upload a new one. Then describe anything you want to find — Elastic does the rest.
      </p>
      <button
        onClick={onUpload}
        className="px-4 py-2 rounded bg-elastic-teal text-elastic-ink text-sm font-semibold flex items-center gap-2 hover:bg-elastic-teal/90 transition"
      >
        <Upload className="w-4 h-4" /> Upload an image
      </button>
    </div>
  );
}
