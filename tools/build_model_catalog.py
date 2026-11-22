#!/usr/bin/env python3
"""Build skills/cdeep3m/cdeep3m_models.json: the CDeep3M model zoo as a catalog.

The descriptive fields are copied from https://cdeep3m.crbs.ucsd.edu/home/pre_trained_models.
Each model's download button is its DOI, which redirects to a .tar on cildata.crbs.ucsd.edu.
This script follows each DOI and lists the archive's files by reading only the 512-byte tar
headers (HTTP Range requests over one connection), so nothing is downloaded. It records
the file URL, its size, and which networks (1fm, 3fm, 5fm) the archive holds.

    python tools/build_model_catalog.py skills/cdeep3m/cdeep3m_models.json [--http]

--http reads the archive over plain HTTP. It is for machines whose TLS connection to cildata stalls;
the catalog always records https URLs.
"""
import json
import re
import sys
import time

import requests

# Copied from the zoo page (2026-11-21). Pixel sizes are in nm as written there; the two
# "CompImBio Teste" entries say 5 µm, kept as written and flagged as test entries.
ZOO_PAGE = [
    (50681, "sbem-mitochondria", "SBEM and ssTEM Mitochondria", "mitochondria",
     "Broadly trained mitochondria model on SBEM and TEM data (CDeep3M 1.6.3)", "SBEM, TEM", 4.8, 70, "Matt Haberl", "1.2.0"),
    (50689, "sbem-mitochondria-denoised", "Mitochondria (SBEM, denoised)", "mitochondria",
     "Broadly trained mitochondria model on SBEM and TEM data, denoised images", "SBEM, TEM", 4.8, 70, "Matthias Haberl", "1.6.3rc2"),
    (50789, "membranes-aug110", "SBEM, TEM Membranes (Aug-110)", "membranes",
     "Membranes, TEM and SBEM data, 60k iterations (CDeep3M2)", "SBEM, TEM", 5, 50, "Matthias Haberl", "2.1.1"),
    (50772, "membranes-aug05", "SBEM, TEM Membranes (Aug05)", "membranes",
     "Membranes, TEM and SBEM data (CDeep3M2)", "SBEM, TEM", 5, 50, "Matthias Haberl", "2.1.0"),
    (50749, "membranes-aug10", "SBEM and TEM Membranes (Aug-10)", "membranes",
     "Membranes, TEM and SBEM data (CDeep3M2); the page gives Z as 50 µm, read as 50 nm", "SBEM, TEM", 5, 50, "Matthias Haberl", "2.1.0"),
    (50692, "sbem-membranes-denoised-resize", "SBEM Membranes (denoised, resize augmentation)", "membranes",
     "Membranes, TEM and SBEM data, CDeep3M 1.6 then transfer learning with CDeep3M2", "SBEM, TEM", 5, 70, "Matthias Haberl", "1.6.3rc2"),
    (50687, "sbem-membranes-denoised", "SBEM Membranes (denoised)", "membranes",
     "Membranes, SBEM, trained 50k-55k iterations on denoised images", "SBEM", 4.8, 70, "Matthias Haberl", "1.6.3rc2"),
    (50673, "tem-membranes", "SBEM and TEM membranes", "membranes",
     "Membranes, TEM and SBEM data (CDeep3M 1.6.3)", "SBEM, TEM", 5, None, "Matt Haberl", "1.2.0rc1"),
    (50683, "tomography-membranes", "Tomography Membranes", "membranes",
     "Membranes in serial-section multi-tilt electron tomography of high-pressure frozen, freeze-substituted tissue",
     "electron tomography", 1.6, 1.6, "Matthias G Haberl", "1.2.0"),
    (50725, "sbem-synapses-denoised", "SBEM Synapses (denoised)", "synapses",
     "Synapse structure (postsynaptic densities, vesicles docking at the presynaptic site), denoised SBEM",
     "SBEM", 12, 60, "Matthias Haberl, Matthew Madany", "1.6.3rc2"),
    (50685, "sbem-synapses", "SBEM (Synapses)", "synapses",
     "Synapse structure: vesicles docking at the membrane and the postsynaptic density", "SBEM", 12, 40, "Matthew Madany", "1.6.2"),
    (50821, "sbem-synaptic-vesicles-aug12", "Synaptic Vesicles SBEM (Aug-12)", "synaptic vesicles",
     "Synaptic vesicles, SBEM data (CDeep3M2)", "SBEM", 4, 20, "Matthias Haberl", "2.1.5"),
    (50682, "sbem-synaptic-vesicles", "SBEM synaptic vesicles", "synaptic vesicles",
     "Synaptic vesicles, SBEM data", "SBEM", 2.4, 24, "Matthias G Haberl", "1.2.0"),
    (54542, "compimbio-test", "CompImBio Teste", "membranes",
     "Test entry: retrain of SBEM and TEM Membranes (Aug-110) with brainiac 2015 v2 images; page gives 5 µm pixels",
     "SBEM, TEM", None, None, "", "1"),
    (50829, "compimbio-test1", "CompImBio Teste1", "membranes",
     "Test entry: retrain of SBEM and TEM Membranes (Aug-110) with brainiac 2015 images; page gives 5 µm pixels",
     "SBEM, TEM", None, None, "", "0"),
]


def _range(session, url, lo, hi, tries=6):
    """Bytes lo..hi; the server closes kept-alive connections now and then, so retry on a new one."""
    for k in range(tries):
        try:
            return session.get(url, headers={"Range": "bytes=%d-%d" % (lo, hi)}, timeout=180)
        except requests.RequestException:
            if k == tries - 1:
                raise
            session.close()
            time.sleep(2 * (k + 1))


def tar_members(session, url):
    """(name, size) of every member, from the 512-byte headers alone."""
    out, pos, longname = [], 0, None
    while True:
        r = _range(session, url, pos, pos + 511)
        if r.status_code != 206:
            raise RuntimeError("no range support (%s) for %s" % (r.status_code, url))
        h = r.content
        if len(h) < 512 or h == b"\0" * 512:
            break
        name = h[0:100].split(b"\0")[0].decode("utf-8", "replace")
        prefix = h[345:500].split(b"\0")[0].decode("utf-8", "replace")
        size = int(h[124:136].split(b"\0")[0].strip() or b"0", 8)
        kind = h[156:157]
        if kind == b"L":                      # GNU long name: the next block(s) hold the name
            r2 = _range(session, url, pos + 512, pos + 511 + size)
            longname = r2.content.split(b"\0")[0].decode("utf-8", "replace")
        else:
            full = longname or ("%s/%s" % (prefix, name) if prefix else name)
            longname = None
            if kind in (b"0", b"\0"):
                out.append((full, size))
        pos += 512 + (size + 511) // 512 * 512
    return out


def main():
    dest = sys.argv[1]
    plain = "--http" in sys.argv
    s = requests.Session()
    s.headers["User-Agent"] = "argus-ncuvo-catalog/1.0"
    models = []
    for (cid, key, title, target, about, imaging, px, pz, authors, version) in ZOO_PAGE:
        doi = "10.7295/W9CDEEP3M%d" % cid
        r = requests.head("https://doi.org/" + doi, allow_redirects=False, timeout=60)
        url = r.headers.get("Location", "")
        if not url.startswith("http"):
            print("!! %s: DOI did not redirect (%s)" % (cid, r.status_code))
            continue
        url = url.replace("http://", "https://")
        read = url.replace("https://", "http://") if plain else url
        t0 = time.time()
        try:
            mem = tar_members(s, read)
        except Exception as e:                # keep the entry, without the archive details
            print("!! %s: could not list %s: %s" % (cid, url, e))
            mem = None
        total = None
        nets = []
        if mem is not None:
            cr = _range(s, read, 0, 0).headers.get("Content-Range", "")
            total = int(cr.rsplit("/", 1)[-1]) if "/" in cr else None
            for f in ("1fm", "3fm", "5fm"):
                names = [n for n, _ in mem if re.search(r"(^|/)%s/" % f, n)]
                if any(n.endswith("deploy.prototxt") for n in names) and any(n.endswith(".caffemodel") for n in names):
                    nets.append(f)
        models.append({
            "id": cid, "name": key, "title": title, "target": target, "about": about, "imaging": imaging,
            "pixel_nm": px, "slice_nm": pz, "authors": authors, "version": version,
            "test_entry": key.startswith("compimbio"),
            "doi": doi, "url": url, "bytes": total, "networks": nets,
            "files_in_archive": len(mem) if mem is not None else None,
        })
        print("%-6s %-32s %8s MB  networks %-15s  (%d files, %.0f s)" % (
            cid, key, "%.0f" % (total / 1e6) if total else "?", ",".join(nets) or "?",
            len(mem or []), time.time() - t0), flush=True)
    doc = {"source": "https://cdeep3m.crbs.ucsd.edu/home/pre_trained_models",
           "built": time.strftime("%Y-%m-%d"),
           "cite": "Haberl et al., CDeep3M-Plug-and-Play cloud-based deep learning for image segmentation, "
                   "Nat Methods 15, 677-680 (2018), doi:10.1038/s41592-018-0106-z",
           "models": models}
    json.dump(doc, open(dest, "w"), indent=1)
    print("wrote %d models to %s" % (len(models), dest))


if __name__ == "__main__":
    main()
