/**
 * API client. The frontend never speaks to Elasticsearch directly —
 * all requests go through the FastAPI backend, which holds the cluster
 * credentials and orchestrates the inference + search calls.
 */

export interface BBox { x: number; y: number; w: number; h: number; }

export interface SearchHit {
  tile_id: string;
  image_id: string;
  image_url: string;
  bbox: BBox;
  score: number;
  rank: number;
}

export interface SearchResponse {
  query: string;
  corrected_query?: string | null;
  corrections?: string[][];
  hits: SearchHit[];
  total_candidates: number;
  inference_ms: number;
  search_ms: number;
  rerank_ms?: number;
  reranked?: boolean;
  elapsed_ms: number;
}

export interface IngestResponse {
  image_id: string;
  width: number;
  height: number;
  tiles_indexed: number;
  elapsed_ms: number;
  inference_ms: number;
  indexing_ms: number;
}

export interface ImageSummary {
  image_id: string;
  width: number;
  height: number;
  tile_count: number;
  image_url: string;
  label: string | null;
  uploaded_at: string | null;
}

export interface SearchRequest {
  query: string;
  image_id?: string;
  k?: number;
  num_candidates?: number;
  min_score?: number;
  hybrid?: boolean;
}

const BASE = "";   // same origin via Vite proxy

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${detail}`);
  }
  return res.json();
}

export const api = {
  health: () => request<{ status: string; cluster_status?: string; cluster_name?: string }>("/api/health"),

  listImages: () => request<ImageSummary[]>("/api/images"),

  ingest: async (file: File, label?: string): Promise<IngestResponse> => {
    const fd = new FormData();
    fd.append("file", file);
    if (label) fd.append("label", label);
    return request<IngestResponse>("/api/ingest", { method: "POST", body: fd });
  },

  search: (req: SearchRequest) => request<SearchResponse>("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  }),

  deleteImage: (imageId: string) => request<{ tiles_deleted: number }>(
    `/api/images/${imageId}`,
    { method: "DELETE" },
  ),
};
