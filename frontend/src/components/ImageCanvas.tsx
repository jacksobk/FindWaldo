import { useEffect, useRef, useState } from "react";
import type { ImageSummary, SearchHit } from "../lib/api";
import { Crosshair, Maximize2 } from "lucide-react";

interface Props {
  image: ImageSummary;
  hits: SearchHit[];
  searching: boolean;
}

/**
 * Renders the original image and overlays bounding boxes drawn in SVG.
 *
 * Layout strategy:
 *   - The <img> determines its own rendered size (object-contain inside a
 *     bounded parent).
 *   - A ResizeObserver tracks the image's rendered width/height in real
 *     pixels.
 *   - The SVG overlay is absolutely positioned at the same coordinates
 *     and size as the rendered image, with a viewBox in the original
 *     image's pixel coordinates. This guarantees bounding boxes align
 *     with the visible image regardless of zoom, window resize, or
 *     aspect-ratio mismatch with the parent.
 */
export function ImageCanvas({ image, hits, searching }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [zoomedHit, setZoomedHit] = useState<SearchHit | null>(null);
  const [imgRect, setImgRect] = useState({ left: 0, top: 0, width: 0, height: 0 });

  // Reset zoom when the image changes
  useEffect(() => { setZoomedHit(null); }, [image.image_id]);

  // Track the image's rendered position and size relative to its parent.
  // We update on resize, on image load, and on hits change (in case the
  // results panel toggling alters the available width).
  useEffect(() => {
    const el = imgRef.current;
    if (!el) return;

    const update = () => {
      const parent = el.offsetParent as HTMLElement | null;
      if (!parent) return;
      setImgRect({
        left: el.offsetLeft,
        top: el.offsetTop,
        width: el.offsetWidth,
        height: el.offsetHeight,
      });
    };

    update();
    if (el.complete) update();
    el.addEventListener("load", update);

    const ro = new ResizeObserver(update);
    ro.observe(el);
    if (el.parentElement) ro.observe(el.parentElement);
    window.addEventListener("resize", update);

    return () => {
      el.removeEventListener("load", update);
      ro.disconnect();
      window.removeEventListener("resize", update);
    };
  }, [image.image_id, hits.length]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="px-5 py-3 border-b border-white/5 flex items-center justify-between bg-elastic-slate/40">
        <div className="flex items-center gap-3 text-sm">
          <span className="font-mono text-elastic-gray uppercase tracking-widest text-xs">image_id</span>
          <span className="font-mono text-white">{image.image_id}</span>
          <span className="text-elastic-gray">·</span>
          <span className="font-mono text-white/80">{image.width}×{image.height}px</span>
          <span className="text-elastic-gray">·</span>
          <span className="font-mono text-white/80">{image.tile_count} tiles</span>
        </div>
        {hits.length > 0 && (
          <div className="text-sm font-mono text-elastic-teal flex items-center gap-1.5">
            <Crosshair className="w-4 h-4" />
            {hits.length} match{hits.length === 1 ? "" : "es"} highlighted
          </div>
        )}
      </div>

      {/* Stage */}
      <div className="flex-1 relative overflow-hidden flex items-center justify-center p-4 bg-elastic-ink">
        <img
          ref={imgRef}
          src={image.image_url}
          alt="indexed scene"
          className="block max-w-full max-h-full object-contain rounded shadow-2xl"
          style={{ maxHeight: "calc(100vh - 320px)" }}
        />

        {/* Scanning effect during search — uses tracked image rect */}
        {searching && imgRect.width > 0 && (
          <div
            className="absolute overflow-hidden rounded pointer-events-none"
            style={{
              left: imgRect.left,
              top: imgRect.top,
              width: imgRect.width,
              height: imgRect.height,
            }}
          >
            <div className="absolute left-0 right-0 h-12 bg-gradient-to-b from-transparent via-elastic-teal/30 to-transparent animate-scan" />
          </div>
        )}

        {/* SVG overlay — pinned to the image's rendered rectangle */}
        {imgRect.width > 0 && (
          <svg
            className="absolute pointer-events-none"
            style={{
              left: imgRect.left,
              top: imgRect.top,
              width: imgRect.width,
              height: imgRect.height,
            }}
            viewBox={`0 0 ${image.width} ${image.height}`}
            preserveAspectRatio="none"
          >
            {/*
              Spotlight mask: when there are hits, dim the entire image then
              "punch out" the bounding boxes so they appear lit while the
              rest of the image is darkened. This is the single biggest
              readability win for visual search results.
            */}
            {hits.length > 0 && (
              <defs>
                <mask id="spotlight-mask">
                  <rect x={0} y={0} width={image.width} height={image.height} fill="white" />
                  {hits.map((hit) => (
                    <rect
                      key={`mask-${hit.tile_id}`}
                      x={hit.bbox.x}
                      y={hit.bbox.y}
                      width={hit.bbox.w}
                      height={hit.bbox.h}
                      fill="black"
                    />
                  ))}
                </mask>
              </defs>
            )}

            {/* Dimming layer that gets punched out by the mask */}
            {hits.length > 0 && (
              <rect
                x={0} y={0}
                width={image.width}
                height={image.height}
                fill="#0B1628"
                opacity={0.55}
                mask="url(#spotlight-mask)"
              />
            )}

            {/* Boxes drawn on top of the dimming layer */}
            {hits.map((hit) => (
              <BoundingBox
                key={hit.tile_id}
                hit={hit}
                isTop={hit.rank === 1}
                onClick={() => setZoomedHit(hit)}
              />
            ))}
          </svg>
        )}

        {/* Zoom modal */}
        {zoomedHit && (
          <ZoomedPatch
            hit={zoomedHit}
            imageUrl={image.image_url}
            imageW={image.width}
            imageH={image.height}
            onClose={() => setZoomedHit(null)}
          />
        )}
      </div>
    </div>
  );
}

function BoundingBox({ hit, isTop, onClick }: { hit: SearchHit; isTop: boolean; onClick: () => void }) {
  const { x, y, w, h } = hit.bbox;
  // High-contrast colors against both light and dark image regions.
  // Top-1: bright Elastic teal. Others: bright Elastic amber.
  const color = isTop ? "#00BFB3" : "#FEC514";
  // Stroke width scales with the smaller bbox side; clamp so it's always
  // visible but never overwhelming.
  const strokeWidth = Math.max(8, Math.min(20, Math.min(w, h) * 0.018));
  const badgeH = Math.max(56, strokeWidth * 5);
  const badgeW = Math.max(80, strokeWidth * 8);
  const scoreW = Math.max(120, strokeWidth * 11);
  const scoreH = Math.max(40, strokeWidth * 4);

  return (
    <g style={{ pointerEvents: "auto", cursor: "pointer" }} onClick={onClick}>
      {/* Outer glow halo — a slightly larger, blurred stroke for emphasis */}
      <rect
        x={x - strokeWidth / 2}
        y={y - strokeWidth / 2}
        width={w + strokeWidth}
        height={h + strokeWidth}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth * 0.4}
        opacity={isTop ? 0.4 : 0.25}
        rx={4}
      />
      {/* Main stroke — solid for top-1, dashed for others */}
      <rect
        x={x} y={y} width={w} height={h}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeDasharray={isTop ? "0" : `${strokeWidth * 3} ${strokeWidth * 1.5}`}
        rx={2}
      />
      {/* Inner contrast stroke — a thin dark line just inside the colored
          stroke gives the boxes definition against bright image regions */}
      <rect
        x={x + strokeWidth / 2}
        y={y + strokeWidth / 2}
        width={w - strokeWidth}
        height={h - strokeWidth}
        fill="none"
        stroke="#0B1628"
        strokeWidth={Math.max(1, strokeWidth * 0.15)}
        opacity={0.6}
        rx={1}
      />

      {/* Rank badge — large, opaque, top-left corner */}
      <g transform={`translate(${x}, ${y})`}>
        <rect
          x={0} y={0}
          width={badgeW}
          height={badgeH}
          fill={color}
          rx={2}
        />
        <text
          x={badgeW / 2}
          y={badgeH / 2}
          textAnchor="middle"
          dominantBaseline="central"
          fill="#0B1628"
          fontSize={badgeH * 0.62}
          fontWeight="800"
          fontFamily="JetBrains Mono, monospace"
        >
          #{hit.rank}
        </text>
      </g>

      {/* Score badge — bottom-right corner with a dark background and
          colored text matching the box color */}
      <g transform={`translate(${x + w - scoreW}, ${y + h - scoreH})`}>
        <rect
          x={0} y={0}
          width={scoreW}
          height={scoreH}
          fill="#0B1628"
          opacity={0.92}
          rx={2}
        />
        <rect
          x={0} y={0}
          width={scoreW}
          height={scoreH}
          fill="none"
          stroke={color}
          strokeWidth={2}
          rx={2}
        />
        <text
          x={scoreW / 2}
          y={scoreH / 2}
          textAnchor="middle"
          dominantBaseline="central"
          fill={color}
          fontSize={scoreH * 0.55}
          fontWeight="700"
          fontFamily="JetBrains Mono, monospace"
        >
          {hit.score.toFixed(3)}
        </text>
      </g>
    </g>
  );
}

function ZoomedPatch({
  hit, imageUrl, imageW, imageH, onClose,
}: { hit: SearchHit; imageUrl: string; imageW: number; imageH: number; onClose: () => void }) {
  const { x, y, w, h } = hit.bbox;
  const scale = 320 / Math.max(w, h);
  const bgW = imageW * scale;
  const bgH = imageH * scale;
  const bgX = -x * scale;
  const bgY = -y * scale;

  return (
    <div
      className="absolute inset-0 flex items-center justify-center bg-elastic-ink/85 backdrop-blur-sm rounded z-30"
      onClick={onClose}
    >
      <div
        className="bg-elastic-slate border border-elastic-teal rounded p-4 max-w-md"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2 text-xs font-mono">
            <Maximize2 className="w-3 h-3 text-elastic-teal" />
            <span className="text-elastic-teal">Rank #{hit.rank}</span>
            <span className="text-elastic-gray">·</span>
            <span>cosine {hit.score.toFixed(4)}</span>
          </div>
          <button onClick={onClose} className="text-elastic-gray hover:text-white text-xs font-mono">
            close ✕
          </button>
        </div>
        <div
          className="rounded border border-elastic-teal"
          style={{
            width: 320,
            height: 320 * (h / w),
            backgroundImage: `url(${imageUrl})`,
            backgroundSize: `${bgW}px ${bgH}px`,
            backgroundPosition: `${bgX}px ${bgY}px`,
            backgroundRepeat: "no-repeat",
          }}
        />
        <div className="mt-2 text-[11px] font-mono text-elastic-gray">
          tile_id: {hit.tile_id} · bbox=({x}, {y}, {w}×{h})
        </div>
      </div>
    </div>
  );
}
