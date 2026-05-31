"""
CytoTrack AI — Open-Licensed Cell Image Database Client
========================================================

Small client that lets a user search a curated catalogue of **open-licensed**
cell/microscopy image datasets and download a subset for phenotype-classifier
training. Every catalogue entry carries an explicit licence field. The
download path refuses to fetch anything whose licence is not on the
permissive allow-list, so whatever ends up on disk is safe to use, share,
and redistribute alongside a derivative classifier.

This is a deliberately *curated* catalogue rather than arbitrary web
scraping: we only list datasets whose licences are documented on their
own homepages. Users can extend the catalogue locally via
``register_dataset()``.

Major sources used here:
  * BBBC – Broad Bioimage Benchmark Collection (https://bbbc.broadinstitute.org)
  * Cell Image Library (http://www.cellimagelibrary.org)
  * Human Protein Atlas (https://www.proteinatlas.org) — subcellular

The module is unit-tested without network access by injecting a custom
``url_opener``.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Licence allow-list
# ----------------------------------------------------------------------
# Only these licences are treated as safe to redistribute as training data
# alongside an open-source classifier. Anything else is refused even if a
# user explicitly tries to register it.
OPEN_LICENCES = frozenset({
    "CC0",
    "CC0-1.0",
    "PUBLIC-DOMAIN",
    "CC-BY-3.0",
    "CC-BY-4.0",
    "CC-BY-SA-3.0",
    "CC-BY-SA-4.0",
    "MIT",
    "APACHE-2.0",
    "BSD-2-CLAUSE",
    "BSD-3-CLAUSE",
})


def normalise_licence(s: str) -> str:
    """Canonical form used for comparison against OPEN_LICENCES."""
    return (s or "").strip().upper().replace(" ", "-")


def is_licence_open(licence: str) -> bool:
    """Return True iff ``licence`` is on the permissive allow-list."""
    return normalise_licence(licence) in OPEN_LICENCES


# ----------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------
@dataclass
class Dataset:
    """One open-licensed cell image dataset."""
    id: str
    name: str
    organism: str
    phenotype: str
    keywords: List[str]
    licence: str
    attribution: str
    homepage: str
    download_url: str
    description: str = ""
    approx_image_count: int = 0
    format_hint: str = "zip"

    def matches(self, query: str) -> int:
        """Return a simple match score (>0 is a hit)."""
        q = (query or "").strip().lower()
        if not q:
            return 0
        hay = " ".join([self.name, self.organism, self.phenotype,
                        self.description, " ".join(self.keywords)]).lower()
        score = 0
        for term in q.replace(",", " ").split():
            if term in hay:
                score += 1
            # also allow short-hand prefix match on id, e.g. "bbbc021"
            if term and self.id.lower().startswith(term):
                score += 2
        return score


# Curated catalogue — every entry here carries an explicit open licence.
# Licence tags reflect what each dataset's own homepage states. When
# downloading, the client re-checks and writes the licence into the
# per-download manifest so derivative work can properly attribute.
_DEFAULT_CATALOG: List[Dataset] = [
    Dataset(
        id="BBBC005",
        name="Simulated HL-60 cells",
        organism="Human",
        phenotype="HL-60 leukemia",
        keywords=["hl60", "leukemia", "simulated", "nuclei",
                  "synthetic", "foci"],
        licence="CC-BY-3.0",
        attribution="Broad Bioimage Benchmark Collection / Ljosa et al. 2012",
        homepage="https://bbbc.broadinstitute.org/BBBC005",
        download_url=(
            "https://data.broadinstitute.org/bbbc/BBBC005/"
            "BBBC005_v1_images.zip"
        ),
        description="Synthetic HL-60 cells at varied focus and cell count. "
                    "Useful for out-of-distribution robustness.",
        approx_image_count=19200,
    ),
    Dataset(
        id="BBBC006",
        name="Human U2OS cells",
        organism="Human",
        phenotype="U2OS osteosarcoma",
        keywords=["u2os", "osteosarcoma", "nuclei", "hoechst"],
        licence="CC-BY-3.0",
        attribution="Broad Bioimage Benchmark Collection",
        homepage="https://bbbc.broadinstitute.org/BBBC006",
        download_url=(
            "https://data.broadinstitute.org/bbbc/BBBC006/"
            "BBBC006_v1_images_z_16.zip"
        ),
        description="Real U2OS cells imaged at a single focal plane (z=16). "
                    "Good baseline for a nuclei-phenotype class.",
        approx_image_count=768,
    ),
    Dataset(
        id="BBBC021",
        name="MCF-7 breast cancer, compound panel (week 1)",
        organism="Human",
        phenotype="MCF-7 breast cancer",
        keywords=["mcf7", "mcf-7", "breast", "cancer", "compound",
                  "phenotype", "mechanism"],
        licence="CC-BY-3.0",
        attribution="Caie et al. 2010 / BBBC021",
        homepage="https://bbbc.broadinstitute.org/BBBC021",
        download_url=(
            "https://data.broadinstitute.org/bbbc/BBBC021/"
            "BBBC021_v1_images_Week1_22123.zip"
        ),
        description="MCF-7 cells treated with a panel of compounds — strong "
                    "phenotype separation for classifier training.",
        approx_image_count=2200,
    ),
    Dataset(
        id="BBBC038",
        name="Diverse nuclei (Data Science Bowl 2018)",
        organism="Mixed",
        phenotype="Diverse nuclei",
        keywords=["nuclei", "dsb2018", "kaggle", "diverse",
                  "segmentation"],
        licence="CC-BY-4.0",
        attribution="Caicedo et al. 2019 / Kaggle DSB 2018",
        homepage="https://bbbc.broadinstitute.org/BBBC038",
        download_url=(
            "https://data.broadinstitute.org/bbbc/BBBC038/"
            "stage1_train.zip"
        ),
        description="Heterogeneous nuclei from many tissues and imaging "
                    "modalities — useful as a 'generic nuclei' class.",
        approx_image_count=670,
    ),
    Dataset(
        id="BBBC007",
        name="Drosophila Kc167 cells",
        organism="Drosophila",
        phenotype="Kc167",
        keywords=["drosophila", "kc167", "insect", "fluorescence"],
        licence="CC-BY-3.0",
        attribution="Jones et al. 2005 / BBBC007",
        homepage="https://bbbc.broadinstitute.org/BBBC007",
        download_url=(
            "https://data.broadinstitute.org/bbbc/BBBC007/"
            "BBBC007_v1_images.zip"
        ),
        description="Drosophila Kc167 cells. A good non-human counterpoint "
                    "class.",
        approx_image_count=32,
    ),
    Dataset(
        id="BBBC019",
        name="Collective cell migration wound-healing assay",
        organism="Human",
        phenotype="migrating epithelial cells",
        keywords=["brightfield", "phase", "migration", "wound", "scratch",
                  "label-free", "light microscopy"],
        licence="CC-BY-3.0",
        attribution="Broad Bioimage Benchmark Collection / collective cell migration",
        homepage="https://bbbc.broadinstitute.org/BBBC019",
        download_url=(
            "https://data.broadinstitute.org/bbbc/BBBC019/"
            "BBBC019_v2_images.zip"
        ),
        description="Brightfield/label-free wound-healing migration assay. "
                    "Useful when the tracking movie is light microscopy "
                    "rather than fluorescence.",
        approx_image_count=200,
    ),
]


def _sanity_check_catalogue() -> None:
    for d in _DEFAULT_CATALOG:
        if not is_licence_open(d.licence):
            raise RuntimeError(
                f"Catalogue entry {d.id} has non-open licence "
                f"'{d.licence}'. Refusing to ship CytoTrack AI with this.")


_sanity_check_catalogue()

_EXTRA_CATALOG: List[Dataset] = []


def register_dataset(d: Dataset) -> None:
    """Runtime extension point — users can add their own open datasets.

    Refuses anything whose licence is not on the open allow-list, so the
    catalogue can never be poisoned with non-redistributable entries.
    """
    if not is_licence_open(d.licence):
        raise ValueError(
            f"Refusing to register '{d.id}': licence '{d.licence}' is not "
            f"on the open allow-list.")
    _EXTRA_CATALOG.append(d)


def catalogue() -> List[Dataset]:
    """Full catalogue (default + user-registered)."""
    return list(_DEFAULT_CATALOG) + list(_EXTRA_CATALOG)


def search(query: str) -> List[Dataset]:
    """Case-insensitive keyword search. Returns datasets sorted by score."""
    scored = [(d.matches(query), d) for d in catalogue()]
    return [d for s, d in sorted(scored, key=lambda x: -x[0]) if s > 0]


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------
UrlOpener = Callable[[str], bytes]


def _default_url_opener(url: str) -> bytes:
    """Network-backed opener. Kept small so it can be swapped in tests."""
    from urllib.request import Request, urlopen
    req = Request(url, headers={"User-Agent": "CytoTrackAI/1.0"})
    with urlopen(req, timeout=60) as resp:
        return resp.read()


def _is_image_name(name: str) -> bool:
    n = name.lower()
    return n.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))


def download(dataset: Dataset,
             target_dir: str,
             max_samples: int = 100,
             url_opener: Optional[UrlOpener] = None,
             verbose: bool = True) -> Dict:
    """
    Download a dataset (or a subset) to ``target_dir``. Refuses to run if
    the dataset's licence is not on the open allow-list.

    Writes a ``manifest.json`` alongside the images recording the source,
    licence, attribution, homepage, and the sha-ish-equivalent size-per-file
    so anyone inheriting the folder knows where it came from.
    """
    if not is_licence_open(dataset.licence):
        raise PermissionError(
            f"Dataset '{dataset.id}' has non-open licence "
            f"'{dataset.licence}'. Refusing to download.")

    opener = url_opener or _default_url_opener
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[cell-image-library] fetching {dataset.id} "
              f"({dataset.licence}) from {dataset.download_url}")

    raw = opener(dataset.download_url)

    saved: List[Dict] = []
    if dataset.format_hint == "zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            members = [m for m in zf.namelist() if _is_image_name(m)]
            members.sort()
            for member in members[:max_samples]:
                data = zf.read(member)
                out_name = Path(member).name
                out_path = target / out_name
                # avoid path traversal — we only use basenames
                with open(out_path, "wb") as fh:
                    fh.write(data)
                saved.append({"name": out_name, "bytes": len(data)})
    else:
        # Fall-through: treat as single image file
        out_path = target / f"{dataset.id}{os.path.splitext(dataset.download_url)[1]}"
        with open(out_path, "wb") as fh:
            fh.write(raw)
        saved.append({"name": out_path.name, "bytes": len(raw)})

    manifest = {
        "dataset_id": dataset.id,
        "name": dataset.name,
        "organism": dataset.organism,
        "phenotype": dataset.phenotype,
        "licence": dataset.licence,
        "attribution": dataset.attribution,
        "homepage": dataset.homepage,
        "download_url": dataset.download_url,
        "downloaded_utc": datetime.utcnow().isoformat() + "Z",
        "file_count": len(saved),
        "files": saved,
        "notes": "All content under its upstream open licence; retain this "
                 "manifest in any redistribution for attribution.",
    }
    with open(target / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    if verbose:
        print(f"[cell-image-library] saved {len(saved)} files to {target}")
    return manifest


def build_phenotype_folders(selections: List[Tuple[str, Dataset]],
                            target_root: str,
                            max_samples_per_class: int = 100,
                            url_opener: Optional[UrlOpener] = None) -> str:
    """
    Assemble a ``target_root/class_name/`` layout suitable for
    ``CellClassifierTrainer.prepare_data``. Each tuple in ``selections``
    is (user_chosen_class_label, Dataset).

    Returns ``target_root`` on success.
    """
    root = Path(target_root)
    root.mkdir(parents=True, exist_ok=True)

    license_log: List[Dict] = []
    for class_label, dataset in selections:
        class_label = class_label.strip().replace(os.sep, "_")
        if not class_label:
            class_label = dataset.id
        class_dir = root / class_label
        class_dir.mkdir(parents=True, exist_ok=True)
        manifest = download(dataset, str(class_dir),
                            max_samples=max_samples_per_class,
                            url_opener=url_opener)
        license_log.append({
            "class": class_label,
            "dataset": dataset.id,
            "licence": dataset.licence,
            "attribution": dataset.attribution,
            "homepage": dataset.homepage,
            "files": manifest["file_count"],
        })

    with open(root / "LICENSES.json", "w", encoding="utf-8") as fh:
        json.dump({
            "generated_utc": datetime.utcnow().isoformat() + "Z",
            "classes": license_log,
            "readme": ("Every image under this folder is covered by the "
                       "open licence shown for its class. Retain this "
                       "file in any redistribution."),
        }, fh, indent=2)

    return str(root)
