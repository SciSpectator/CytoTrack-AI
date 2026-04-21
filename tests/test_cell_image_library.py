"""
Tests for the open-licensed cell-image-database client.

No network access — a fake url_opener is injected so we can verify the
licence-filter enforcement and the download / manifest flow end-to-end
against an in-memory zip.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile

import numpy as np

import cell_image_library as cil


def _make_fake_zip(num_images: int = 3) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(num_images):
            # Minimal valid PNG-ish payload (content doesn't matter for
            # the test — we only check that bytes are written).
            zf.writestr(f"fake/cell_{i:03d}.png", bytes([137, 80, 78, 71, i]))
        # Non-image sibling — must be skipped by the downloader.
        zf.writestr("fake/readme.txt", b"not an image")
    return buf.getvalue()


def test_licence_allowlist_recognises_permissive_and_rejects_rest():
    assert cil.is_licence_open("CC-BY-4.0")
    assert cil.is_licence_open("cc-by-3.0")
    assert cil.is_licence_open("CC0")
    assert cil.is_licence_open("MIT")
    assert not cil.is_licence_open("proprietary")
    assert not cil.is_licence_open("CC-BY-NC-4.0")  # non-commercial is NOT ok
    assert not cil.is_licence_open("CC-BY-ND-4.0")  # no-derivs is NOT ok
    assert not cil.is_licence_open("")


def test_catalogue_is_all_open_licences():
    for d in cil.catalogue():
        assert cil.is_licence_open(d.licence), (
            f"Shipped catalogue entry {d.id} has non-open licence "
            f"'{d.licence}'")


def test_register_dataset_refuses_non_open_licence():
    bad = cil.Dataset(
        id="TEST-PROP", name="proprietary test", organism="Human",
        phenotype="anything", keywords=["test"], licence="proprietary",
        attribution="nobody", homepage="http://example.com",
        download_url="http://example.com/data.zip",
    )
    try:
        cil.register_dataset(bad)
    except ValueError:
        return
    raise AssertionError(
        "register_dataset should have refused a proprietary dataset")


def test_search_matches_keyword_case_insensitive():
    hits = cil.search("MCF-7")
    assert any(d.id == "BBBC021" for d in hits)
    hits2 = cil.search("nuclei")
    assert any(d.id == "BBBC038" for d in hits2)


def test_download_refuses_blocked_licence(tmp_path_factory=None):
    # Build a doctored dataset with a bad licence to make sure download
    # won't run on it even if somebody sneaks one past register_dataset.
    bad = cil.Dataset(
        id="X", name="x", organism="x", phenotype="x", keywords=["x"],
        licence="CC-BY-NC-4.0",
        attribution="x", homepage="http://x", download_url="http://x/x.zip",
    )
    with tempfile.TemporaryDirectory() as td:
        try:
            cil.download(bad, td, url_opener=lambda url: b"")
        except PermissionError:
            return
    raise AssertionError("download should have refused non-open licence")


def test_download_writes_manifest_and_images():
    data = _make_fake_zip(num_images=4)
    opener = lambda url: data  # noqa: E731

    # Pick any real catalogue entry — we don't actually hit the network.
    ds = next(d for d in cil.catalogue() if d.id == "BBBC006")
    with tempfile.TemporaryDirectory() as td:
        manifest = cil.download(ds, td, max_samples=10,
                                url_opener=opener, verbose=False)
        assert manifest["file_count"] == 4
        assert manifest["licence"] == ds.licence
        # Manifest file exists and parses.
        with open(os.path.join(td, "manifest.json"), "r") as fh:
            m = json.load(fh)
        assert m["dataset_id"] == ds.id
        assert m["licence"] == ds.licence
        # Images were written with their basenames, not their nested paths.
        for entry in manifest["files"]:
            assert os.path.exists(os.path.join(td, entry["name"]))


def test_build_phenotype_folders_writes_licenses_json():
    data = _make_fake_zip(num_images=3)
    opener = lambda url: data  # noqa: E731
    catalogue = cil.catalogue()
    a = next(d for d in catalogue if d.id == "BBBC006")
    b = next(d for d in catalogue if d.id == "BBBC021")
    selections = [("u2os", a), ("mcf7", b)]
    with tempfile.TemporaryDirectory() as td:
        root = cil.build_phenotype_folders(selections, td,
                                           max_samples_per_class=3,
                                           url_opener=opener)
        # One subfolder per class label.
        assert os.path.isdir(os.path.join(root, "u2os"))
        assert os.path.isdir(os.path.join(root, "mcf7"))
        with open(os.path.join(root, "LICENSES.json"), "r") as fh:
            lic = json.load(fh)
        classes = {c["class"]: c for c in lic["classes"]}
        assert classes["u2os"]["licence"] == a.licence
        assert classes["mcf7"]["licence"] == b.licence
