"""Project-wide constants: NoData encodings and GeoTIFF creation options.

Centralizes values that were previously scattered as literals across the
data / process / model stages.

NoData conventions are fixed by CLAUDE.md and MUST NOT change:
  - Byte / categorical rasters : 255
  - Float32 rasters            : -9999.0
  - Risk output (UInt16)       : 0   (0 = NoData, 1..65535 = probability)

``GTIFF_OPTS`` is the canonical GeoTIFF creation-option list. The stages used to
disagree — the data downloaders wrote ``COMPRESS=DEFLATE`` while the process /
model stages wrote ``COMPRESS=LZW`` — which produced silently different output
files. Standardizing on LZW (lossless, already used by the bulk of the
pipeline-internal rasters) makes every stage consistent. Pixel values are
unchanged; only the compression codec — and therefore the on-disk bytes — of
the formerly-DEFLATE rasters changes.
"""

# NoData sentinels (see CLAUDE.md "Data conventions" table).
NODATA_BYTE = 255       # Byte / categorical rasters (forest, FCC, plantation, masks)
NODATA_FLOAT = -9999.0  # Float32 rasters (distances, slope, residuals)
NODATA_RISK = 0         # UInt16 risk output: 0 = NoData, 1..65535 = probability

# Canonical GeoTIFF creation options: a single lossless codec across all stages,
# tiled for efficient windowed I/O.
GTIFF_OPTS = ["COMPRESS=LZW", "TILED=YES"]
