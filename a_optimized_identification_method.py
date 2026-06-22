from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sunpy.map

from astropy.io import fits
from scipy.ndimage import label
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int64]


NEIGHBOR_OFFSETS = np.array(
    [
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],  [0, 0],  [0, 1],
        [1, -1],  [1, 0],  [1, 1],
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class Config:
    num: int = 1
    root: Path = Path(r"G:\paper1\data")

    start_index: int = 40
    tail_margin: int = 80

    coefficient: float = 1.0
    hit_limit: int = 9
    min_component_size: int = 8

    use_abs_denominator: bool = True
    eps: float = 1e-8

    @property
    def data_dir(self) -> Path:
        return self.root / str(self.num)

    @property
    def hmi_dir(self) -> Path:
        return self.data_dir / "remove_dff"

    @property
    def flare_region_csv(self) -> Path:
        return self.data_dir / "flare_region.csv"

    @property
    def delta_csv(self) -> Path:
        return self.data_dir / "delta" / "0.csv"


def list_fits_files(folder: Path) -> list[Path]:
    """Return sorted FITS files."""
    suffixes = {".fits", ".fit", ".fts"}

    paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    ]

    if not paths:
        raise FileNotFoundError(f"No FITS files found in: {folder}")

    return sorted(paths)


def read_image(path: Path) -> ArrayF:

    data = fits.getdata(path)
    return np.asarray(data, dtype=np.float64)


def read_first_columns(path: Path, names: list[str]) -> pd.DataFrame:

    df = pd.read_csv(path)
    df = df.iloc[:, : len(names)].copy()
    df.columns = names
    return df


def load_flare_pixels(cfg: Config) -> tuple[ArrayI, ArrayI, ArrayF]:

    flare = read_first_columns(cfg.flare_region_csv, ["row", "column"])
    delta = read_first_columns(cfg.delta_csv, ["row", "column", "delta"])

    flare[["row", "column"]] = flare[["row", "column"]].astype(np.int64)
    delta[["row", "column"]] = delta[["row", "column"]].astype(np.int64)
    delta["delta"] = delta["delta"].astype(np.float64)

    delta = delta.drop_duplicates(["row", "column"], keep="last")

    merged = flare.merge(delta, on=["row", "column"], how="inner")

    if merged.empty:
        raise ValueError("No matched pixels between flare_region.csv and delta/0.csv")

    rows = merged["row"].to_numpy(dtype=np.int64)
    cols = merged["column"].to_numpy(dtype=np.int64)
    deltas = merged["delta"].to_numpy(dtype=np.float64)

    return rows, cols, deltas


def relative_change(
    after: ArrayF,
    before: ArrayF,
    *,
    use_abs_denominator: bool,
    eps: float,
) -> ArrayF:

    denominator = np.abs(before) if use_abs_denominator else before

    out = np.zeros_like(after, dtype=np.float64)

    valid = np.abs(denominator) > eps
    np.divide(
        np.abs(after - before),
        denominator,
        out=out,
        where=valid,
    )

    return out


def detect_pixels_in_one_frame(
    previous: ArrayF,
    current: ArrayF,
    following: ArrayF,
    rows: ArrayI,
    cols: ArrayI,
    deltas: ArrayF,
    cfg: Config,
) -> tuple[ArrayI, ArrayI]:
    
    height, width = current.shape

    valid = (
        (rows > 0)
        & (rows < height - 1)
        & (cols > 0)
        & (cols < width - 1)
        & np.isfinite(deltas)
    )

    if not np.any(valid):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    r = rows[valid]
    c = cols[valid]
    q = deltas[valid]

    rr = r[:, None] + NEIGHBOR_OFFSETS[:, 0]
    cc = c[:, None] + NEIGHBOR_OFFSETS[:, 1]

    previous_patch = previous[rr, cc]
    current_patch = current[rr, cc]
    following_patch = following[rr, cc]

    change_1 = relative_change(
        current_patch,
        previous_patch,
        use_abs_denominator=cfg.use_abs_denominator,
        eps=cfg.eps,
    )

    change_2 = relative_change(
        following_patch,
        current_patch,
        use_abs_denominator=cfg.use_abs_denominator,
        eps=cfg.eps,
    )

    threshold = cfg.coefficient * q[:, None]

    hit_count = (
        np.count_nonzero(change_1 >= threshold, axis=1)
        + np.count_nonzero(change_2 >= threshold, axis=1)
    )

    selected = hit_count >= cfg.hit_limit

    return r[selected], c[selected]


def keep_large_connected_components(
    rows: ArrayI,
    cols: ArrayI,
    image_shape: tuple[int, int],
    min_size: int,
) -> NDArray[np.bool_]:

    mask = np.zeros(image_shape, dtype=bool)

    if rows.size == 0:
        return mask

    mask[rows, cols] = True

    labeled, num_features = label(mask)

    if num_features == 0:
        return mask

    sizes = np.bincount(labeled.ravel())

    keep_labels = np.flatnonzero(sizes > min_size)
    keep_labels = keep_labels[keep_labels != 0]

    if keep_labels.size == 0:
        return np.zeros(image_shape, dtype=bool)

    return np.isin(labeled, keep_labels)


def iter_fits_triplets(
    paths: list[Path],
    start: int,
    stop: int,
):
    previous = read_image(paths[start - 1])
    current = read_image(paths[start])

    for idx in range(start, stop):
        following = read_image(paths[idx + 1])
        yield idx, previous, current, following
        previous, current = current, following


def run_detection(
    hmi_paths: list[Path],
    rows: ArrayI,
    cols: ArrayI,
    deltas: ArrayF,
    cfg: Config,
) -> tuple[ArrayI, ArrayI]:

    stop = len(hmi_paths) - cfg.tail_margin

    if cfg.start_index <= 0:
        raise ValueError("start_index must be >= 1 because previous frame is required.")

    if stop <= cfg.start_index:
        raise ValueError(
            f"Not enough FITS files. Got {len(hmi_paths)}, "
            f"but start_index={cfg.start_index}, tail_margin={cfg.tail_margin}."
        )

    image_shape = read_image(hmi_paths[0]).shape
    accumulated_mask = np.zeros(image_shape, dtype=bool)

    for idx, previous, current, following in iter_fits_triplets(
        hmi_paths,
        cfg.start_index,
        stop,
    ):
        cand_rows, cand_cols = detect_pixels_in_one_frame(
            previous,
            current,
            following,
            rows,
            cols,
            deltas,
            cfg,
        )

        component_mask = keep_large_connected_components(
            cand_rows,
            cand_cols,
            current.shape,
            cfg.min_component_size,
        )

        accumulated_mask |= component_mask

    detected_rows, detected_cols = np.where(accumulated_mask)

    return detected_rows.astype(np.int64), detected_cols.astype(np.int64)


def plot_detection(
    map_path: Path,
    flare_rows: ArrayI,
    flare_cols: ArrayI,
    detected_rows: ArrayI,
    detected_cols: ArrayI,
):

    omap = sunpy.map.Map(str(map_path))
    image_shape = omap.data.shape

    flare_mask = np.zeros(image_shape, dtype=np.uint8)

    valid_flare = (
        (flare_rows >= 0)
        & (flare_rows < image_shape[0])
        & (flare_cols >= 0)
        & (flare_cols < image_shape[1])
    )

    flare_mask[flare_rows[valid_flare], flare_cols[valid_flare]] = 1

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(projection=omap)

    omap.plot(axes=ax)

    if detected_rows.size > 0:
        ax.scatter(
            detected_cols,
            detected_rows,
            s=2,
            linewidths=0.1,
            alpha=1,
            marker="s",
            c="skyblue",
            transform=ax.get_transform("pixel"),
            label="Detected pixels",
        )

    ax.contour(
        flare_mask,
        levels=[0.5],
        colors="r",
        linewidths=0.3,
        transform=ax.get_transform("pixel"),
    )

    ax.set_title("")
    ax.set_xlabel("Solar X [arcsec]")
    ax.set_ylabel("Solar Y [arcsec]")

    if detected_rows.size > 0:
        ax.legend(loc="best", fontsize=6)

    plt.show()


def main():
    cfg = Config(
        num=1,
        coefficient=1.0,
        hit_limit=9,
        min_component_size=8,
        start_index=40,
        tail_margin=80,
    )

    hmi_paths = list_fits_files(cfg.hmi_dir)

    rows, cols, deltas = load_flare_pixels(cfg)

    detected_rows, detected_cols = run_detection(
        hmi_paths,
        rows,
        cols,
        deltas,
        cfg,
    )

    print(f"Input flare pixels: {rows.size}")
    print(f"Detected pixels: {detected_rows.size}")

    plot_detection(
        hmi_paths[0],
        rows,
        cols,
        detected_rows,
        detected_cols,
    )


if __name__ == "__main__":
    main()
