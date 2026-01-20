from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import re
import os
import numpy as np
from tqdm import tqdm

# --- helpers ---
_suffix_re = re.compile(r"ij_link(\d+)_(\d+)\.csv$")

def _matches_files(path: Path, files: int) -> tuple[bool, int | None]:
    m = _suffix_re.search(path.name)
    if not m:
        return False, None
    grp_files, grp_suffix = int(m.group(1)), int(m.group(2))
    return (grp_files == files, grp_suffix if grp_files == files else None)


def _load_csv(path: str, skip_header: int = 0, delimiter: str = ','):
    arr = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header)

    # Remove trailing NaN column
    if arr.ndim == 2 and np.all(np.isnan(arr[:, -1])):
        arr = arr[:, :-1]

    # Check even number of columns and reshape
    if arr.ndim == 2:
        n_cols = arr.shape[1]
        if n_cols % 2 != 0:
            raise ValueError(f"{path.name} has an odd number of columns ({n_cols}). Expected even number.")
        arr = arr.reshape(arr.shape[0], n_cols // 2, 2)

    return arr


def load_ij_links_parallel(
    directory: str | Path,
    files: int,
    *,
    skip_header: int = 0,
    delimiter: str = ',',
    max_workers: int | None = None,
    sort_by_suffix: bool = True,
    return_map: bool = True,
):
    """Parallel load of all ij_link{files}_*.csv files, returns dict or list of arrays."""
    directory = Path(directory)
    candidates = list(directory.glob(f"ij_link{files}_*.csv"))

    targets: list[tuple[int, Path]] = []
    for p in candidates:
        ok, suf = _matches_files(p, files)
        if ok:
            targets.append((suf, p))
    if not targets:
        raise FileNotFoundError(f"No files found for files={files} in {directory}")

    if sort_by_suffix:
        targets.sort(key=lambda t: t[0])

    if max_workers is None:
        max_workers = min(os.cpu_count() or 1, len(targets))

    results: dict[int, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_load_csv, str(p), skip_header, delimiter): suf
            for suf, p in targets
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Loading ij_link{files}", ncols=80):
            suf = futures[fut]
            arr = fut.result()
            results[suf] = arr

    if return_map:
        return results
    else:
        ordered_suffixes = sorted(results.keys()) if sort_by_suffix else list(results.keys())
        return [results[s] for s in ordered_suffixes]

def load_and_stack_ij_links(
    directory: str | Path,
    data_iteration: int,
    n_cores: int | None = None,
    **kwargs
) -> np.ndarray:

    arrays = load_ij_links_parallel(directory, data_iteration, return_map=False, max_workers=n_cores, **kwargs)

    return np.concatenate(arrays)
