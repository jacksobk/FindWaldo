"""
Image tiling pipeline.

Why tile? CLIP models embed an entire image into a single vector. For a
busy scene like a Where's Waldo image, that vector averages over thousands
of objects — Waldo's signal gets drowned. Tiling solves this: we split the
image into overlapping patches, embed each patch separately, and let kNN
find the patch that best matches the query.

Why multi-scale? CLIP doesn't know what scale a query refers to. "A red
striped shirt" could be a Waldo (~30px tall) or a giant in striped clothing
(~700px tall). With a fixed tile size you only see matches at that scale.
By indexing tiles at three scales (224, 384, 768 by default) we let kNN
pick the right scale automatically.

The overlap matters. Without it, an object straddling two tiles is
half-represented in each, and neither tile scores well. ~50% overlap is
the right default with multi-scale — the larger scales already see plenty
of context, and 50% overlap on the smallest scale is what catches small
objects regardless of where they fall in the grid.
"""
import base64
import io
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image, ImageOps


@dataclass(frozen=True)
class Tile:
    row: int         # row position within this scale's grid
    col: int         # column position within this scale's grid
    x: int           # top-left x in original (post-resize) image coordinates
    y: int           # top-left y
    w: int           # width in pixels (== tile_size unless on the right edge)
    h: int           # height in pixels (== tile_size unless on the bottom edge)
    scale: int       # which tile_size this tile belongs to (e.g. 224, 384, 768)
    b64: str         # base64-encoded JPEG of the patch


class ImageTiler:
    """
    Configurable multi-scale tiler.

    Two modes:
      * Multi-scale: pass tile_sizes=(224, 384, 768) and the tiler emits a
        flat list of tiles across all scales, each tagged with its `scale`.
      * Single-scale (legacy): pass tile_size=384, omit tile_sizes. The
        emitted tiles all have scale = tile_size.

    Holds no state per-image; safe to share across requests.
    """

    def __init__(
        self,
        tile_size: int = 384,
        tile_sizes: tuple[int, ...] | None = None,
        overlap: float = 0.5,
        max_dim: int = 3072,
        jpeg_quality: int = 88,
    ):
        if not (0.0 <= overlap < 1.0):
            raise ValueError("overlap must be in [0, 1)")
        # Normalize: if tile_sizes is provided, use it; otherwise single-size.
        sizes = tuple(tile_sizes) if tile_sizes else (tile_size,)
        if not sizes:
            raise ValueError("at least one tile size required")
        self.tile_sizes = sizes
        self.overlap = overlap
        self.max_dim = max_dim
        self.jpeg_quality = jpeg_quality

    def stride_for(self, size: int) -> int:
        return max(1, int(round(size * (1.0 - self.overlap))))

    def tile(self, raw_bytes: bytes) -> Tuple[bytes, int, int, List[Tile]]:
        """
        Returns:
          full_jpeg_bytes: the (possibly resized) full image, JPEG-encoded,
                           for the frontend to display.
          width, height:   dimensions of that full image.
          tiles:           flat list of Tile records across all scales.
        """
        img = Image.open(io.BytesIO(raw_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = self._resize_within(img, self.max_dim)
        width, height = img.size

        # Encode the full image once, for the frontend's <img> tag.
        full_buf = io.BytesIO()
        img.save(full_buf, format="JPEG", quality=self.jpeg_quality, optimize=True)
        full_jpeg = full_buf.getvalue()

        tiles: List[Tile] = []
        for size in self.tile_sizes:
            tiles.extend(self._tile_at_scale(img, width, height, size))
        return full_jpeg, width, height, tiles

    def _tile_at_scale(self, img: Image.Image, width: int, height: int,
                       size: int) -> List[Tile]:
        """Emit tiles for a single scale. Skipped if size > image dim, in
        which case we emit a single tile covering the whole image (rare for
        2560+ inputs, but happens for small thumbnails)."""
        out: List[Tile] = []
        stride = self.stride_for(size)

        if width <= size and height <= size:
            # Image is smaller than this scale's tile — emit one whole-image
            # tile so the largest scale always sees scene-level context.
            patch_b64 = self._encode_patch(img.crop((0, 0, width, height)))
            out.append(Tile(row=0, col=0, x=0, y=0, w=width, h=height,
                            scale=size, b64=patch_b64))
            return out

        xs = self._step_positions(width, size, stride)
        ys = self._step_positions(height, size, stride)
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                w = min(size, width - x)
                h = min(size, height - y)
                patch = img.crop((x, y, x + w, y + h))
                out.append(Tile(
                    row=row, col=col, x=x, y=y, w=w, h=h,
                    scale=size, b64=self._encode_patch(patch),
                ))
        return out

    def _encode_patch(self, patch: Image.Image) -> str:
        buf = io.BytesIO()
        patch.save(buf, format="JPEG", quality=self.jpeg_quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _resize_within(img: Image.Image, max_dim: int) -> Image.Image:
        w, h = img.size
        scale = max_dim / max(w, h)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            return img.resize(new_size, Image.LANCZOS)
        return img

    @staticmethod
    def _step_positions(total: int, tile: int, stride: int) -> List[int]:
        """
        Step positions covering [0, total). Last step is anchored to total - tile
        so the rightmost/bottommost tile fits cleanly without padding.
        """
        if total <= tile:
            return [0]
        positions = list(range(0, total - tile + 1, stride))
        last = total - tile
        if positions[-1] != last:
            positions.append(last)
        return positions


# Module-level convenience for tests / scripts
def tile_image(raw_bytes: bytes, **kwargs) -> Tuple[bytes, int, int, List[Tile]]:
    return ImageTiler(**kwargs).tile(raw_bytes)
