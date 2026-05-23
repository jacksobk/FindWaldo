import { useEffect, useState } from "react";
import {
  ChevronLeft, ChevronRight, X, Search as SearchIcon,
  Building2, Heart, ShoppingBag, Briefcase, Headphones,
  Database, Cpu, Layers, Zap, ArrowRight, Sparkles, Image as ImageIcon,
} from "lucide-react";
import { api, type ImageSummary, type SearchHit } from "../lib/api";
import { ImageCanvas } from "./ImageCanvas";

interface Props { onExit: () => void; }

const SECTIONS = [
  "opening",
  "problem",
  "solution",
  "architecture",
  "live",
  "value",
  "differentiation",
  "expansion",
  "close",
] as const;

type SectionId = typeof SECTIONS[number];

export function DemoMode({ onExit }: Props) {
  const [idx, setIdx] = useState(0);
  const section: SectionId = SECTIONS[idx];

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight" || e.key === " ") setIdx(i => Math.min(i + 1, SECTIONS.length - 1));
      if (e.key === "ArrowLeft") setIdx(i => Math.max(i - 1, 0));
      if (e.key === "Escape") onExit();
      const n = parseInt(e.key);
      if (!isNaN(n) && n >= 1 && n <= SECTIONS.length) setIdx(n - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onExit]);

  return (
    <div className="min-h-screen bg-elastic-ink text-white relative overflow-hidden">
      {/* Top bar */}
      <div className="absolute top-0 left-0 right-0 z-30 px-6 py-3 flex items-center justify-between border-b border-white/5 bg-elastic-ink/80 backdrop-blur">
        <div className="flex items-center gap-3 text-[10px] font-mono uppercase tracking-widest text-elastic-gray">
          <span className="w-1.5 h-1.5 rounded-full bg-elastic-teal animate-pulse" />
          Find Waldo · Executive Demo
        </div>
        <div className="text-[10px] font-mono uppercase tracking-widest text-elastic-gray">
          {String(idx + 1).padStart(2, "0")} / {String(SECTIONS.length).padStart(2, "0")} · {section}
        </div>
        <button
          onClick={onExit}
          className="text-[10px] font-mono uppercase tracking-widest text-elastic-gray hover:text-white flex items-center gap-1.5"
        >
          <X className="w-3 h-3" /> exit
        </button>
      </div>

      {/* Slide content */}
      <div className="absolute inset-0 pt-14 pb-14 overflow-hidden">
        {section === "opening" && <Opening />}
        {section === "problem" && <Problem />}
        {section === "solution" && <Solution />}
        {section === "architecture" && <Architecture />}
        {section === "live" && <LiveDemo />}
        {section === "value" && <Value />}
        {section === "differentiation" && <Differentiation />}
        {section === "expansion" && <Expansion />}
        {section === "close" && <Close />}
      </div>

      {/* Bottom nav */}
      <div className="absolute bottom-0 left-0 right-0 z-30 px-6 py-3 flex items-center justify-between border-t border-white/5 bg-elastic-ink/80 backdrop-blur">
        <button
          onClick={() => setIdx(i => Math.max(0, i - 1))}
          disabled={idx === 0}
          className="text-xs font-mono uppercase tracking-widest text-elastic-gray hover:text-white disabled:opacity-20 flex items-center gap-1.5"
        >
          <ChevronLeft className="w-3 h-3" /> prev
        </button>
        <div className="flex items-center gap-1.5">
          {SECTIONS.map((_, i) => (
            <button
              key={i}
              onClick={() => setIdx(i)}
              className={`h-[3px] rounded-full transition-all ${i === idx ? "w-10 bg-elastic-teal" : i < idx ? "w-5 bg-white/40" : "w-5 bg-white/10"}`}
            />
          ))}
        </div>
        <button
          onClick={() => setIdx(i => Math.min(SECTIONS.length - 1, i + 1))}
          disabled={idx === SECTIONS.length - 1}
          className="text-xs font-mono uppercase tracking-widest text-elastic-gray hover:text-white disabled:opacity-20 flex items-center gap-1.5"
        >
          next <ChevronRight className="w-3 h-3" />
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// 1. Opening
// ─────────────────────────────────────────────────────────────────

function Opening() {
  return (
    <div className="h-full flex flex-col items-center justify-center px-12 text-center">
      <div className="text-[10px] font-mono uppercase tracking-[0.4em] text-elastic-teal mb-8">
        Multimodal Visual Search · Powered by Elastic
      </div>
      <h1 className="font-display text-7xl md:text-8xl lg:text-9xl leading-[0.9] mb-8 max-w-5xl">
        If you can find <span className="italic text-elastic-teal">Waldo</span><br />
        you can find <span className="italic text-elastic-yellow">anything</span><br />
        in your data.
      </h1>
      <p className="text-lg text-elastic-gray max-w-2xl font-display italic">
        80% of enterprise data is unstructured. Most of it is unsearchable. This changes that.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// 2. Problem
// ─────────────────────────────────────────────────────────────────

function Problem() {
  const verticals = [
    { icon: Building2, title: "Insurance",        body: "Adjusters wade through hundreds of claim photos to find a single dent or roof tile. Hours per claim, multiplied by thousands of claims." },
    { icon: Heart,     title: "Healthcare",       body: "Radiologists review imaging studies one by one. Prior comparable cases live in a PACS that no one can search by what's actually in the image." },
    { icon: ShoppingBag, title: "Retail",          body: "Customers describe products in their own words. Catalog search returns SKU codes. The disconnect costs conversion." },
    { icon: Briefcase, title: "Financial services", body: "Analysts hunt through scanned documents, charts, ID photos, and screenshots. OCR misses the visual context that matters most." },
    { icon: Headphones, title: "Customer support", body: "Agents ask customers to upload screenshots. Then they search those screenshots manually because nothing else can." },
  ];
  return (
    <div className="h-full flex flex-col px-16 py-12">
      <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-elastic-gray mb-2">The Problem</div>
      <h2 className="font-display text-5xl mb-4 max-w-4xl leading-tight">
        Every industry is drowning in <span className="italic text-elastic-teal">visual</span> data it can't search.
      </h2>
      <p className="text-base text-elastic-gray max-w-3xl mb-10">
        Keyword search assumes someone already wrote down what's in the picture. Nobody did. The signal is locked in the pixels.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 flex-1 content-start">
        {verticals.map((v, i) => (
          <div key={i} className="border border-white/10 rounded-lg p-5 bg-elastic-slate/30 hover:border-elastic-teal/40 transition">
            <v.icon className="w-5 h-5 text-elastic-teal mb-3" />
            <div className="font-display text-xl mb-2">{v.title}</div>
            <p className="text-sm text-elastic-gray leading-relaxed">{v.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// 3. Solution
// ─────────────────────────────────────────────────────────────────

function Solution() {
  return (
    <div className="h-full grid grid-cols-12 gap-12 px-16 py-12">
      <div className="col-span-7 flex flex-col justify-center">
        <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-elastic-gray mb-2">The Solution</div>
        <h2 className="font-display text-6xl leading-tight mb-6">
          One platform. <span className="italic text-elastic-teal">One API.</span> No glue.
        </h2>
        <p className="text-lg text-elastic-gray leading-relaxed mb-4">
          Elasticsearch already runs your full-text search, your logs, your APM. With Elastic Inference Service in 9.4, the same cluster runs the AI model that converts images and text into vectors — and serves them at production scale.
        </p>
        <p className="text-base text-elastic-gray leading-relaxed">
          No Pinecone. No Weaviate. No separate Python service. No external API key. The model runs <em>inside</em> Elastic, and the dense_vector field, the kNN search, and the inference call all share one query language.
        </p>
      </div>
      <div className="col-span-5 flex flex-col gap-3 justify-center">
        <Pillar icon={Database} label="Vector storage"  body="dense_vector field with HNSW indexing, native to Elasticsearch." />
        <Pillar icon={Cpu}      label="Native inference" body="EIS hosts jina-clip-v2. No external provider, no API key." highlight />
        <Pillar icon={Layers}   label="Hybrid retrieval"  body="kNN, BM25, filters, aggregations — one DSL, one round trip." />
        <Pillar icon={Zap}      label="Production scale"  body="Same infra running your logs and APM today." />
      </div>
    </div>
  );
}

function Pillar({ icon: Icon, label, body, highlight }: { icon: any; label: string; body: string; highlight?: boolean }) {
  return (
    <div className={`p-4 rounded-lg border ${highlight ? "border-elastic-teal bg-elastic-teal/5" : "border-white/10 bg-elastic-slate/30"}`}>
      <div className="flex items-start gap-3">
        <Icon className={`w-5 h-5 mt-0.5 ${highlight ? "text-elastic-teal" : "text-elastic-gray"}`} />
        <div>
          <div className="font-display text-lg leading-none mb-1">{label}</div>
          <div className="text-xs text-elastic-gray leading-relaxed">{body}</div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// 4. Architecture
// ─────────────────────────────────────────────────────────────────

function Architecture() {
  return (
    <div className="h-full px-16 py-12 flex flex-col">
      <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-elastic-gray mb-2">How It Works</div>
      <h2 className="font-display text-5xl mb-3">
        Four moving parts. <span className="italic text-elastic-teal">All inside Elastic.</span>
      </h2>
      <p className="text-base text-elastic-gray max-w-3xl mb-10">
        Images come in, get tiled into overlapping patches, get embedded by Jina CLIP v2 via EIS, and land in a single index. Queries take the same path — only the input is text.
      </p>

      <div className="flex-1 flex items-center">
        <div className="w-full">
          {/* Ingest pipeline */}
          <div className="text-[10px] font-mono uppercase tracking-widest text-elastic-gray mb-3">Ingest</div>
          <div className="grid grid-cols-9 gap-3 items-stretch mb-6">
            <ArchBox col={2} title="Image"     mono="JPG / PNG"             icon={ImageIcon} />
            <ArchArrow />
            <ArchBox col={2} title="Tile"      mono="overlap 33%"           icon={Layers} />
            <ArchArrow />
            <ArchBox col={2} title="Embed"     mono="EIS · jina-clip-v2"    icon={Cpu} highlight />
            <ArchArrow />
            <ArchBox col={1} title="Index"     mono="dense_vector"          icon={Database} />
          </div>

          {/* Query pipeline */}
          <div className="text-[10px] font-mono uppercase tracking-widest text-elastic-gray mb-3">Query</div>
          <div className="grid grid-cols-9 gap-3 items-stretch">
            <ArchBox col={2} title="Text"      mono='"red striped shirt"'   icon={SearchIcon} />
            <ArchArrow />
            <ArchBox col={2} title="Embed"     mono="EIS · jina-clip-v2"    icon={Cpu} highlight />
            <ArchArrow />
            <ArchBox col={2} title="kNN"       mono="cosine · top-K"        icon={Zap} />
            <ArchArrow />
            <ArchBox col={1} title="Boxes"     mono="bbox overlay"          icon={ImageIcon} />
          </div>
        </div>
      </div>

      <div className="mt-8 text-sm text-elastic-gray italic font-display max-w-3xl">
        The same model embeds both images and text into the same 1024-dim space. That's why a phrase in English finds a patch in a picture — they live next to each other in vector space.
      </div>
    </div>
  );
}

function ArchBox({ col, title, mono, icon: Icon, highlight }: { col: number; title: string; mono: string; icon: any; highlight?: boolean }) {
  return (
    <div
      className={`rounded-lg p-4 border min-h-[110px] flex flex-col justify-between ${
        highlight ? "border-elastic-teal bg-elastic-teal/10" : "border-white/10 bg-elastic-slate/30"
      }`}
      style={{ gridColumn: `span ${col} / span ${col}` }}
    >
      <Icon className={`w-4 h-4 ${highlight ? "text-elastic-teal" : "text-elastic-gray"}`} />
      <div>
        <div className="font-display text-base leading-none mb-1">{title}</div>
        <div className="text-[10px] font-mono text-elastic-gray truncate">{mono}</div>
      </div>
    </div>
  );
}

function ArchArrow() {
  return (
    <div className="flex items-center justify-center" style={{ gridColumn: "span 1 / span 1" }}>
      <ArrowRight className="w-4 h-4 text-elastic-teal" />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// 5. Live demo (real backend calls inside Demo Mode)
// ─────────────────────────────────────────────────────────────────

function LiveDemo() {
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [active, setActive] = useState<ImageSummary | null>(null);
  const [hits, setHits] = useState<SearchHit[]>([]);
  const [searching, setSearching] = useState(false);
  const [meta, setMeta] = useState<{ embed: number; knn: number } | null>(null);
  const [step, setStep] = useState(0);

  const SCRIPT = [
    { query: "",                                  caption: "We start with a busy crowd scene. The audience tries to find Waldo by eye. They can't, fast." },
    { query: "red and white striped shirt",       caption: "First query. We're not naming Waldo — we're describing what makes him recognizable." },
    { query: "person with glasses and a hat",     caption: "Tighter description. Watch the rank #1 box shift to the patch that matches both attributes." },
    { query: "find Waldo",                        caption: "Plain natural language. CLIP has seen enough of the world to know who Waldo is." },
  ];

  useEffect(() => {
    api.listImages().then((list) => {
      setImages(list);
      if (list.length > 0) setActive(list[0]);
    });
  }, []);

  async function runStep(stepIdx: number) {
    setStep(stepIdx);
    const item = SCRIPT[stepIdx];
    if (!item.query || !active) {
      setHits([]); setMeta(null); return;
    }
    setSearching(true);
    try {
      const resp = await api.search({
        query: item.query,
        image_id: active.image_id,
        k: 5,
      });
      setHits(resp.hits);
      setMeta({ embed: resp.inference_ms, knn: resp.search_ms });
    } finally {
      setSearching(false);
    }
  }

  if (!active) {
    return (
      <div className="h-full flex flex-col items-center justify-center px-12 text-center">
        <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-elastic-gray mb-2">Live Demo</div>
        <h2 className="font-display text-5xl mb-4">No image indexed yet.</h2>
        <p className="text-base text-elastic-gray max-w-md">
          Exit Demo Mode, upload a Where's Waldo scene, then come back. The live demo runs against the same backend the application uses.
        </p>
      </div>
    );
  }

  return (
    <div className="h-full grid grid-cols-12 gap-6 px-10 py-8">
      <div className="col-span-8 flex flex-col">
        <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-elastic-gray mb-2">Live Demo</div>
        <h2 className="font-display text-3xl mb-3 leading-tight">
          Real cluster. Real inference. <span className="italic text-elastic-teal">Real boxes.</span>
        </h2>
        <div className="flex-1 rounded-lg border border-white/10 overflow-hidden bg-black">
          <ImageCanvas image={active} hits={hits} searching={searching} />
        </div>
      </div>
      <div className="col-span-4 flex flex-col gap-3">
        <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-elastic-gray">Walkthrough</div>
        {SCRIPT.map((item, i) => (
          <button
            key={i}
            onClick={() => runStep(i)}
            className={`text-left p-4 rounded-lg border transition ${
              step === i
                ? "border-elastic-teal bg-elastic-teal/10"
                : "border-white/10 bg-elastic-slate/30 hover:border-white/20"
            }`}
          >
            <div className="flex items-start gap-3 mb-2">
              <div className={`w-7 h-7 rounded flex items-center justify-center text-xs font-mono font-bold flex-shrink-0 ${
                step === i ? "bg-elastic-teal text-elastic-ink" : "bg-white/10 text-white"
              }`}>
                {i + 1}
              </div>
              <div className="flex-1">
                {item.query ? (
                  <div className="font-mono text-sm">"{item.query}"</div>
                ) : (
                  <div className="font-display italic text-sm">no query yet</div>
                )}
              </div>
            </div>
            <div className="text-xs text-elastic-gray leading-relaxed pl-10">
              {item.caption}
            </div>
          </button>
        ))}
        {meta && (
          <div className="mt-auto p-3 rounded bg-elastic-slate/40 border border-white/5 text-[11px] font-mono text-elastic-gray flex items-center justify-between">
            <span>embed: <span className="text-elastic-teal">{meta.embed}ms</span></span>
            <span>kNN: <span className="text-elastic-teal">{meta.knn}ms</span></span>
            <span>hits: <span className="text-elastic-teal">{hits.length}</span></span>
          </div>
        )}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// 6. Why this matters (business value)
// ─────────────────────────────────────────────────────────────────

function Value() {
  const items = [
    { stat: "Hours → seconds",  label: "Decision velocity",
      body: "Adjusters, analysts, and clinicians find the relevant evidence in one query instead of scrolling." },
    { stat: "0 manual tagging", label: "Operational cost",
      body: "Visual content is searchable on ingest. No labeling teams, no taxonomy meetings, no tag debt." },
    { stat: "Native UX",        label: "Customer experience",
      body: "Customers describe what they want in their own words and find it. In any of 89 languages." },
    { stat: "Latent → liquid",  label: "Data monetization",
      body: "Image archives that were dead weight become a queryable asset class — RAG ground truth, training data, recommendations." },
  ];
  return (
    <div className="h-full px-16 py-12 flex flex-col">
      <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-elastic-gray mb-2">Why This Matters</div>
      <h2 className="font-display text-5xl mb-10 max-w-4xl leading-tight">
        The capability is multimodal search.<br />
        The <span className="italic text-elastic-teal">outcome</span> is operating leverage.
      </h2>
      <div className="grid grid-cols-2 gap-5 flex-1 content-start">
        {items.map((it, i) => (
          <div key={i} className="border border-white/10 rounded-lg p-6 bg-elastic-slate/30">
            <div className="font-display text-4xl text-elastic-teal mb-2 leading-none">{it.stat}</div>
            <div className="text-[10px] font-mono uppercase tracking-widest text-elastic-gray mb-3">{it.label}</div>
            <p className="text-sm text-white/80 leading-relaxed">{it.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// 7. Differentiation
// ─────────────────────────────────────────────────────────────────

function Differentiation() {
  const rows = [
    { capability: "Vector search",                elastic: true,  vectorDB: true,  llmAPI: false },
    { capability: "Full-text (BM25)",             elastic: true,  vectorDB: false, llmAPI: false },
    { capability: "Filtering + aggregations",     elastic: true,  vectorDB: "limited", llmAPI: false },
    { capability: "Real-time analytics",          elastic: true,  vectorDB: false, llmAPI: false },
    { capability: "Native model hosting",         elastic: true,  vectorDB: false, llmAPI: true },
    { capability: "Multimodal embeddings",        elastic: true,  vectorDB: false, llmAPI: "external" },
    { capability: "Production ops surface",       elastic: true,  vectorDB: "limited", llmAPI: false },
    { capability: "One auth boundary",            elastic: true,  vectorDB: false, llmAPI: false },
  ];
  return (
    <div className="h-full px-16 py-12 flex flex-col">
      <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-elastic-gray mb-2">Why Elastic</div>
      <h2 className="font-display text-5xl mb-3">
        Vector-only DBs don't have search. <span className="italic text-elastic-teal">LLM APIs don't have data.</span>
      </h2>
      <p className="text-base text-elastic-gray max-w-3xl mb-8">
        You need all of it in the same place — full-text, vectors, filters, aggregations, and the model itself. That's Elastic.
      </p>
      <div className="rounded-lg border border-white/10 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-elastic-slate/50">
            <tr>
              <th className="text-left px-5 py-3 text-[10px] font-mono uppercase tracking-widest text-elastic-gray">Capability</th>
              <th className="px-5 py-3 text-[10px] font-mono uppercase tracking-widest text-elastic-teal">Elastic 9.4 + EIS</th>
              <th className="px-5 py-3 text-[10px] font-mono uppercase tracking-widest text-elastic-gray">Vector-only DB</th>
              <th className="px-5 py-3 text-[10px] font-mono uppercase tracking-widest text-elastic-gray">LLM API alone</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className={i % 2 ? "bg-elastic-slate/20" : ""}>
                <td className="px-5 py-2.5 font-medium">{r.capability}</td>
                <td className="px-5 py-2.5 text-center"><Cell value={r.elastic} /></td>
                <td className="px-5 py-2.5 text-center"><Cell value={r.vectorDB} /></td>
                <td className="px-5 py-2.5 text-center"><Cell value={r.llmAPI} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Cell({ value }: { value: boolean | string }) {
  if (value === true)  return <span className="text-elastic-teal text-base">●</span>;
  if (value === false) return <span className="text-white/15 text-base">○</span>;
  return <span className="text-elastic-yellow text-[11px] font-mono italic">{value}</span>;
}

// ─────────────────────────────────────────────────────────────────
// 8. Expansion
// ─────────────────────────────────────────────────────────────────

function Expansion() {
  const stages = [
    { tag: "Today",      title: "Multimodal kNN",      body: "Image + text in one vector space, on one index, in one query." },
    { tag: "+ 1 day",    title: "Hybrid (RRF)",        body: "Fuse vector ranks with BM25 ranks over OCR'd text, captions, or metadata. Better recall on rare terms." },
    { tag: "+ 1 week",   title: "Reranking",           body: "Add a reranker on top of the kNN candidates. Bigger model, smaller candidate set, sharper relevance." },
    { tag: "+ 1 quarter", title: "Closed loop",         body: "Click logs feed Learn-to-Rank. The system gets better with every query the business asks." },
  ];
  return (
    <div className="h-full px-16 py-12 flex flex-col">
      <div className="text-[10px] font-mono uppercase tracking-[0.3em] text-elastic-gray mb-2">Where This Goes</div>
      <h2 className="font-display text-5xl mb-3">
        Today's demo is <span className="italic">Phase Zero.</span>
      </h2>
      <p className="text-base text-elastic-gray max-w-3xl mb-10">
        Each phase below ships on the same Elastic cluster. No new vendors. No new query language. Each one tightens relevance and unlocks adjacent use cases.
      </p>
      <div className="grid grid-cols-4 gap-4 flex-1 content-start">
        {stages.map((s, i) => (
          <div key={i} className="rounded-lg border border-white/10 bg-elastic-slate/30 p-5">
            <div className="text-[10px] font-mono uppercase tracking-widest text-elastic-teal mb-2">{s.tag}</div>
            <div className="font-display text-2xl mb-3 leading-tight">{s.title}</div>
            <p className="text-sm text-elastic-gray leading-relaxed">{s.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// 9. Close
// ─────────────────────────────────────────────────────────────────

function Close() {
  return (
    <div className="h-full flex flex-col items-center justify-center px-12 text-center">
      <Sparkles className="w-8 h-8 text-elastic-teal mb-6" />
      <h1 className="font-display text-7xl md:text-8xl leading-[0.95] mb-8 max-w-5xl">
        If you can find <span className="italic text-elastic-teal">Waldo</span>,<br />
        you can find <span className="italic text-elastic-yellow">signal in chaos</span>.
      </h1>
      <p className="text-lg text-elastic-gray max-w-2xl font-display italic mb-10">
        The same platform that runs your search, your observability, and your security data — now searches your visual world.
      </p>
      <div className="text-[10px] font-mono uppercase tracking-[0.4em] text-elastic-gray">
        Let's talk about your data.
      </div>
    </div>
  );
}
