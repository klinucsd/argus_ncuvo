#!/usr/bin/env python3
"""Build skills/cil/cil_records.parquet from CIL's public metadata.

Source: https://github.com/slash-segmentation/CIL_Public_Data_JSON (Version8_6, one JSON per public
record, last updated 2018-04). Download and unpack it, then:

    curl -sL -o cil.tgz https://codeload.github.com/slash-segmentation/CIL_Public_Data_JSON/tar.gz/refs/heads/master
    mkdir cil_src && tar xzf cil.tgz -C cil_src
    python tools/build_cil_records.py cil_src skills/cil/cil_records.parquet

One row per record. Ontology fields become "; "-joined names; pixel sizes are converted to nm
(pixel_nm_x/y/z) when the record states microns or nanometres, and kept empty otherwise. `files` is a
JSON list of {name, type, bytes, url}: the record's files and any extra ("alternative") files.
"""
import glob
import json
import os
import re
import sys

import pandas as pd

BASE = "https://cildata.crbs.ucsd.edu"
ONTO = [("IMAGINGMODE", "imaging_mode"), ("CELLULARCOMPONENT", "component"), ("CELLTYPE", "cell_type"),
        ("NCBIORGANISMALCLASSIFICATION", "organism"), ("BIOLOGICALPROCESS", "process"),
        ("PREPARATION", "preparation"), ("SOURCEOFCONTRAST", "contrast"),
        ("VISUALIZATIONMETHODS", "visualization"), ("ITEMTYPE", "item_type"),
        ("IMAGEDESCRIPTION", "description"), ("TERMSANDCONDITIONS", "terms")]
TO_NM = {"microns": 1000.0, "micrometers": 1000.0, "um": 1000.0, "nanometers": 1.0, "nm": 1.0}


def names(x):
    if x is None:
        return None
    xs = x if isinstance(x, list) else [x]
    v = [str(i.get("onto_name") or i.get("free_text")) for i in xs
         if isinstance(i, dict) and (i.get("onto_name") or i.get("free_text"))]
    return "; ".join(dict.fromkeys(v)) or None


def as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def record(rid, d):
    c = d.get("CIL_CCDB", {})
    cil, st, dt = c.get("CIL", {}), c.get("Status", {}), c.get("Data_type", {})
    core = cil.get("CORE", {})
    if not st.get("Is_public", True) or st.get("Deleted"):
        return None
    video = bool(dt.get("Video"))
    r = {"id": rid, "kind": "video" if video else "image", "z_stack": bool(dt.get("Z_stack")),
         "time_series": bool(dt.get("Time_series"))}
    for k, col in ONTO:
        r[col] = names(core.get(k))
    for dim in core.get("DIMENSION") or []:
        sp = dim.get("Space") if isinstance(dim, dict) else None
        if isinstance(sp, dict) and sp.get("axis") in ("X", "Y", "Z"):
            a = sp["axis"].lower()
            r["size_" + a] = as_int(sp.get("Image_size"))
            ps = sp.get("Pixel_size") or {}
            v, f = as_float(ps.get("value")), TO_NM.get(str(ps.get("unit", "")).lower())
            r["pixel_nm_" + a] = v * f if (v is not None and f) else None
    files = []
    fl = cil.get("Image_files") or []
    for x in (fl if isinstance(fl, list) else [fl]):
        if isinstance(x, dict) and x.get("File_path"):
            files.append({"name": x["File_path"], "type": x.get("File_type"), "bytes": as_int(x.get("Size")),
                          "url": "%s/media/%s/%d/%s" % (BASE, "videos" if video else "images", rid, x["File_path"])})
    alt = cil.get("Alternative_image_files") or []
    for x in (alt if isinstance(alt, list) else [alt]):
        if isinstance(x, dict) and x.get("URL_postfix"):
            files.append({"name": x["URL_postfix"].rstrip("/").rsplit("/", 1)[-1], "type": x.get("File_type"),
                          "bytes": as_int(x.get("Size")), "url": BASE + x["URL_postfix"], "extra": True})
    r["files"] = json.dumps(files)
    r["bytes"] = sum(f["bytes"] or 0 for f in files)
    at = core.get("ATTRIBUTION") or {}
    r["contributors"] = "; ".join(map(str, at.get("Contributors") or [])) or None
    cit = c.get("Citation") or {}
    r["doi"] = (cit.get("DOI") or "").replace("doi:", "") or None
    r["citation"] = cit.get("Title")
    # The citation reads "Authors (year) CIL:12345, what it shows"; the part after the ID is the title.
    m = re.search(r"CIL:\s*\d+\s*,\s*(.+)$", r["citation"] or "", re.S)
    t = (m.group(1).strip() if m else None) or r["citation"] or ""
    r["title"] = re.sub(r"\.?\s*CIL\.\s*Dataset\.?\s*$", "", t).strip() or None
    return r


def main():
    src, dest = sys.argv[1], sys.argv[2]
    rows = []
    for f in glob.glob(os.path.join(src, "**", "CIL_PUBLIC_DATA", "*.json"), recursive=True):
        try:
            r = record(int(os.path.basename(f)[:-5]), json.load(open(f)))
        except (ValueError, OSError):
            continue
        if r:
            rows.append(r)
    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    for a in "xyz":
        for p in ("size_", "pixel_nm_"):
            if p + a not in df:
                df[p + a] = None
        df["size_" + a] = df["size_" + a].astype("Int64")
        df["pixel_nm_" + a] = df["pixel_nm_" + a].astype("float64")
    df.to_parquet(dest, compression="zstd", index=False)
    print("%d records, %d columns, %.2f MB -> %s" % (len(df), len(df.columns), os.path.getsize(dest) / 1e6, dest))


if __name__ == "__main__":
    main()
