"""The Cell Image Library (CIL), for agents: search its public records, read their details, look
inside their files and download them, without an account.

    import cil_lib as cil
    cil.about()                                   # what the catalog holds
    cil.search("mitochondria", imaging="electron", z_stack=True)
    cil.record(7605)                              # details, files, terms of use, citation
    cil.contents(7605)                            # what is inside a record's ZIP, read remotely
    cil.download(7605)                            # to /tmp/cil/7605 (asks first when large)
    cil.slices(7605)                              # a folder of 2-D image slices, ready to segment

The catalog (cil_records.parquet, shipped with this skill) is built from CIL's public metadata
snapshot (github.com/slash-segmentation/CIL_Public_Data_JSON, April 2018): 10,023 records. Records
added later are not in it. Files come from https://cildata.crbs.ucsd.edu, which needs no login.
"""
import io
import json
import os
import re
import shutil
import time
import zipfile

LIB_VERSION = 2
BASE = "https://cildata.crbs.ucsd.edu"
SNAPSHOT = "CIL public metadata snapshot of April 2018 (10,023 records)"
ASK_ABOVE_GB = 2.0          # downloads larger than this need confirm=True
IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
TERMS = {"public_domain": "public domain", "attribution_cc_by": "CC BY (attribution)",
         "attribution_nc_sa": "CC BY-NC-SA (non-commercial, share alike)",
         "attribution_nc_nd": "CC BY-NC-ND (non-commercial, no derivatives)",
         "attribution_sa": "CC BY-SA (share alike)", "attribution_nc": "CC BY-NC (non-commercial)",
         "attribution_nd": "CC BY-ND (no derivatives)", "copyright": "copyright (ask the contributor before reuse)"}
_session = None


# ----------------------------------------------------------------------------- catalog
def _here():
    return os.path.dirname(os.path.abspath(__file__))


def _db():
    import duckdb
    con = duckdb.connect()
    path = os.path.join(_here(), "cil_records.parquet").replace("'", "''")
    con.execute("CREATE VIEW records AS SELECT * FROM read_parquet('%s')" % path)
    return con


def _scratch():
    s = os.environ.get("CIL_SCRATCH", "/tmp/cil")
    os.makedirs(s, exist_ok=True)
    return s


def _id(x):
    """A record ID from 7605, "7605", "CIL_7605", "CIL:7605", or a dict returned by this library."""
    if isinstance(x, dict):
        x = x.get("id")
    m = re.fullmatch(r"\s*(?:CIL[_: ]?)?(\d+)\s*", str(x), re.IGNORECASE)
    if not m:
        raise ValueError("not a CIL record ID: %r" % (x,))
    return int(m.group(1))


def _size(n):
    if n is None:
        return "?"
    n = float(n)
    for unit in ("bytes", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return ("%d %s" % (n, unit)) if unit == "bytes" else ("%.1f %s" % (n, unit))
        n /= 1000


def _px(r):
    x, y, z = r.get("pixel_nm_x"), r.get("pixel_nm_y"), r.get("pixel_nm_z")
    if x is None or x != x:
        return "—"
    s = "%g" % round(x, 3)
    if y is not None and y == y and abs(y - x) > 1e-9:
        s += " × %g" % round(y, 3)
    if z is not None and z == z:
        s += ", Z %g" % round(z, 3)
    return s + " nm"


def _dims(r):
    v = [r.get("size_x"), r.get("size_y"), r.get("size_z")]
    v = [int(a) for a in v if a is not None and a == a]
    return " × ".join(map(str, v)) if v else "—"


def _short(s, n=60):
    s = (s or "").replace("|", "/").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _row(con, rid):
    cur = con.execute("SELECT * FROM records WHERE id = ?", [rid])
    vals = cur.fetchone()
    if vals is None:
        return None
    return dict(zip([d[0] for d in cur.description], vals))


def about():
    """What the catalog holds: record counts by kind, imaging, terms of use, and sizes."""
    con = _db()
    n, size, px, zs, vid = con.execute(
        "SELECT count(*), sum(bytes), count(pixel_nm_x), sum(z_stack::int), sum((kind='video')::int) FROM records").fetchone()
    print("## Cell Image Library catalog\n")
    print("%s. Files are downloaded from %s without an account.\n" % (SNAPSHOT, BASE))
    print("| | |\n|---|---|")
    print("| Records | %d (%d videos, %d Z-stacks / volumes) |" % (n, vid, zs))
    print("| Files in total | %s |" % _size(size))
    print("| Records stating a pixel size | %d |" % px)
    em = con.execute("SELECT count(*) FROM records WHERE imaging_mode ILIKE '%electron%'").fetchone()[0]
    print("| Electron-microscopy records | %d |" % em)
    print("\n**Imaging (most common)**\n\n| Imaging | Records |\n|---|---|")
    for m, c in con.execute("SELECT imaging_mode, count(*) c FROM records WHERE imaging_mode IS NOT NULL "
                            "GROUP BY 1 ORDER BY c DESC LIMIT 10").fetchall():
        print("| %s | %d |" % (_short(m, 80), c))
    print("\n**Terms of use**\n\n| Terms | Records |\n|---|---|")
    for t, c in con.execute("SELECT terms, count(*) c FROM records GROUP BY 1 ORDER BY c DESC").fetchall():
        print("| %s | %d |" % (TERMS.get(t, t or "not stated"), c))
    return {"records": n, "bytes": size, "electron_microscopy": em}


def _stem(word):
    """Drop the ending of a long word, so that singular and plural tags both match: the catalog has
    both "mitochondria" and "mitochondrion", "synapse" and "synapses"."""
    w = str(word).strip()
    return w[:-2] if len(w) >= 7 and " " not in w else w


def search(text=None, imaging=None, component=None, organism=None, cell_type=None, kind=None, z_stack=None,
           pixel_nm=None, max_gb=None, terms=None, limit=20):
    """Find records. Every argument narrows the search (case-insensitive substring matches):
    text        words matched against title, description, component, cell type, organism, imaging
    imaging     e.g. "electron", "confocal", "TEM", "SEM"
    component   cellular component, e.g. "mitochondri", "synapse", "membrane"
    organism, cell_type, kind ("image" or "video"), z_stack (True for volumes)
    pixel_nm    (low, high) pixel size range in nm, or one number (within a factor of 1.5)
    max_gb      only records whose files total at most this many GB
    terms       e.g. "public_domain", "cc_by", "nc" (non-commercial), "copyright"
    """
    where, args = [], []
    if text:
        for w in str(text).split():
            where.append("(title || ' ' || coalesce(description,'') || ' ' || coalesce(component,'') || ' ' || "
                         "coalesce(cell_type,'') || ' ' || coalesce(organism,'') || ' ' || coalesce(imaging_mode,'')) ILIKE ?")
            args.append("%" + _stem(w) + "%")
    for col, v in (("imaging_mode", imaging), ("component", component), ("organism", organism),
                   ("cell_type", cell_type), ("terms", terms)):
        if v:
            where.append("%s ILIKE ?" % col)
            args.append("%" + (_stem(v) if col in ("component", "organism", "cell_type") else str(v)) + "%")
    if kind:
        where.append("kind = ?")
        args.append(kind)
    if z_stack is not None:
        where.append("z_stack = ?")
        args.append(bool(z_stack))
    if pixel_nm is not None:
        lo, hi = (pixel_nm[0], pixel_nm[1]) if isinstance(pixel_nm, (list, tuple)) else (pixel_nm / 1.5, pixel_nm * 1.5)
        where.append("pixel_nm_x BETWEEN ? AND ?")
        args += [float(lo), float(hi)]
    if max_gb is not None:
        where.append("bytes <= ?")
        args.append(float(max_gb) * 1e9)
    sql = "FROM records" + (" WHERE " + " AND ".join(where) if where else "")
    con = _db()
    total = con.execute("SELECT count(*) " + sql, args).fetchone()[0]
    cur = con.execute("SELECT * " + sql + " ORDER BY bytes DESC LIMIT ?", args + [int(limit)])
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, v)) for v in cur.fetchall()]
    crit = ", ".join("%s = %s" % (k, v) for k, v in (("text", text), ("imaging", imaging), ("component", component),
                     ("organism", organism), ("cell_type", cell_type), ("kind", kind), ("z_stack", z_stack),
                     ("pixel_nm", pixel_nm), ("max_gb", max_gb), ("terms", terms)) if v is not None)
    print("## CIL search: %s\n" % (crit or "all records"))
    print("%d records match%s; %s.\n" % (total, ", the %d largest shown" % len(rows) if total > len(rows) else "",
                                          SNAPSHOT))
    if rows:
        print("| ID | Subject | About | Imaging | Component | Pixel size | Dimensions | Size | Terms |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            print("| %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["id"], _short(r["title"], 40), _short(r["description"], 60), _short(r["imaging_mode"], 40),
                _short(r["component"], 40), _px(r), _dims(r), _size(r["bytes"]),
                TERMS.get(r["terms"], r["terms"] or "—").split(" (")[0]))
    return {"matches": total, "records": [{k: r[k] for k in ("id", "title", "imaging_mode", "component", "pixel_nm_x",
                                                              "pixel_nm_y", "pixel_nm_z", "size_z", "bytes", "terms")}
                                          for r in rows]}


def _files(r):
    return json.loads(r["files"] or "[]")


def record(cil_id):
    """Everything the catalog knows about one record: description, imaging, sizes, files, terms, citation."""
    rid = _id(cil_id)
    r = _row(_db(), rid)
    if r is None:
        print("CIL_%d is not in the catalog (%s). It may be newer, or not public; its page is "
              "https://www.cellimagelibrary.org/images/%d" % (rid, SNAPSHOT, rid))
        return {"id": rid, "found": False}
    print("## CIL_%d: %s\n" % (rid, _short(r["title"], 120)))
    if r.get("citation"):
        print("Citation: %s\n" % r["citation"].strip())
    if r["description"]:
        print(r["description"].strip() + "\n")
    print("| | |\n|---|---|")
    for label, v in (("Kind", r["kind"] + (", Z-stack" if r["z_stack"] else "") + (", time series" if r["time_series"] else "")),
                     ("Imaging", r["imaging_mode"]), ("Cellular component", r["component"]), ("Cell type", r["cell_type"]),
                     ("Organism", r["organism"]), ("Preparation", r["preparation"]),
                     ("Dimensions (pixels)", _dims(r)), ("Pixel size (as recorded)", _px(r)),
                     ("Terms of use", TERMS.get(r["terms"], r["terms"] or "not stated")),
                     ("Contributors", r["contributors"]), ("DOI", r["doi"]),
                     ("Page", "https://www.cellimagelibrary.org/images/%d" % rid)):
        if v:
            print("| %s | %s |" % (label, _short(str(v), 200)))
    files = _files(r)
    print("\n**Files**\n\n| File | Type | Size | Address |\n|---|---|---|---|")
    for f in files:
        print("| %s%s | %s | %s | %s |" % (f["name"], " (extra)" if f.get("extra") else "", f.get("type") or "",
                                          _size(f.get("bytes")), f["url"]))
    print("\nThe ZIP holds the data as submitted to CIL and the TIFF is CIL's converted copy; an extra file, "
          "when there is one, is often the full original dataset. Pixel sizes are as the contributor recorded them.")
    return {"id": rid, "found": True, "kind": r["kind"], "z_stack": bool(r["z_stack"]),
            "pixel_nm": [r["pixel_nm_x"], r["pixel_nm_y"], r["pixel_nm_z"]], "files": files,
            "terms": r["terms"], "imaging": r["imaging_mode"], "component": r["component"], "title": r["title"]}


# ----------------------------------------------------------------------------- remote files
def _http():
    global _session
    if _session is None:
        import requests
        _session = requests.Session()
        _session.headers["User-Agent"] = "argus-cil/1.0"
    return _session


def _get(url, headers=None, stream=False, tries=4):
    import requests
    last = None
    for k in range(tries):
        try:
            r = _http().get(url, headers=headers or {}, stream=stream, timeout=(120, 300))
            if r.status_code in (200, 206):
                return r
            if r.status_code == 404:
                raise FileNotFoundError("not on the CIL server: %s" % url)
            last = "HTTP %s" % r.status_code
        except requests.RequestException as e:
            last = "%s: %s" % (type(e).__name__, e)
        time.sleep(3 * (k + 1))
    raise ConnectionError("could not read %s (%s)" % (url, last))


class _Remote(io.RawIOBase):
    """A read-only, seekable view of a file on a web server that supports byte ranges, so zipfile and
    tifffile can read the parts they need without downloading the whole file."""

    BLOCK = 4 << 20

    def __init__(self, url, size):
        self.url, self.size, self.pos, self.cache = url, size, 0, {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            if len(self.cache) > 16:
                self.cache.pop(next(iter(self.cache)))
            lo = i * self.BLOCK
            hi = min(self.size, lo + self.BLOCK) - 1
            r = _get(self.url, {"Range": "bytes=%d-%d" % (lo, hi)})
            if r.status_code != 206:
                raise IOError("the server ignored a byte-range request for %s" % self.url)
            self.cache[i] = r.content
        return self.cache[i]

    def readinto(self, b):
        n = min(len(b), max(0, self.size - self.pos))
        got = 0
        while got < n:
            i, off = divmod(self.pos + got, self.BLOCK)
            chunk = self._block(i)[off: off + n - got]
            if not chunk:
                break
            b[got: got + len(chunk)] = chunk
            got += len(chunk)
        self.pos += got
        return got


def _remote(f):
    return io.BufferedReader(_Remote(f["url"], f["bytes"] or _length(f["url"])), buffer_size=1 << 20)


def _length(url):
    r = _get(url, {"Range": "bytes=0-0"})
    m = re.search(r"/(\d+)$", r.headers.get("Content-Range", ""))
    return int(m.group(1)) if m else int(r.headers.get("Content-Length", 0))


def _pick(r, file=None, prefer=("Zip", "OME_tif")):
    files = _files(r)
    if not files:
        raise FileNotFoundError("CIL_%d lists no files" % r["id"])
    if file == "extra":                   # the record's extra file: often the full original dataset
        hit = [f for f in files if f.get("extra")]
        if not hit:
            raise FileNotFoundError("CIL_%d has no extra file; it has %s" % (r["id"], ", ".join(f["name"] for f in files)))
        return hit[0]
    if file:
        hit = [f for f in files if f["name"] == file or f["name"].lower().endswith(str(file).lower())
               or (f.get("type") or "").lower() == str(file).lower()]
        if not hit:
            raise FileNotFoundError("CIL_%d has no file %r; it has %s" % (r["id"], file, ", ".join(f["name"] for f in files)))
        return hit[0]
    for t in prefer:
        for f in files:
            if f.get("type") == t:
                return f
    return files[0]


def _record_or_fail(cil_id):
    rid = _id(cil_id)
    r = _row(_db(), rid)
    if r is None:
        raise LookupError("CIL_%d is not in the catalog (%s)" % (rid, SNAPSHOT))
    return r


def contents(cil_id, file=None, limit=40):
    """List what is inside a record's ZIP (its original submitted data) by reading only the ZIP's index
    from the server. Nothing is downloaded."""
    r = _record_or_fail(cil_id)
    f = _pick(r, file, prefer=("Zip",))
    if not f["name"].lower().endswith(".zip"):
        raise ValueError("%s is not a ZIP; contents() reads ZIP files" % f["name"])
    with zipfile.ZipFile(_remote(f)) as z:
        items = [i for i in z.infolist() if not i.is_dir()]
    imgs = [i for i in items if i.filename.lower().endswith(IMAGE_EXT)]
    print("## Inside %s (CIL_%d, %s)\n" % (f["name"], r["id"], _size(f.get("bytes"))))
    print("%d files, %d of them images; read from the ZIP's index without downloading it.\n" % (len(items), len(imgs)))
    print("| File | Size |\n|---|---|")
    for i in items[:limit]:
        print("| %s | %s |" % (i.filename, _size(i.file_size)))
    if len(items) > limit:
        print("| … %d more | |" % (len(items) - limit))
    return {"id": r["id"], "file": f["name"], "members": [(i.filename, i.file_size) for i in items],
            "images": [i.filename for i in imgs]}


def _free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def _guard(nbytes, dest, confirm, what):
    gb = (nbytes or 0) / 1e9
    free = _free_gb(dest)
    if gb > free * 0.9:
        raise OSError("%s needs %.1f GB but only %.1f GB are free in %s" % (what, gb, free, dest))
    if gb > ASK_ABOVE_GB and not confirm:
        print("%s is %.1f GB (%.1f GB free in `%s`). Nothing has been downloaded; it starts only when you "
              "confirm." % (what, gb, free, dest))
        return False
    return True


def _fetch(f, dest):
    """Download one file, continuing a partial download and checking the size at the end."""
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    total = f.get("bytes") or _length(f["url"])
    if os.path.exists(dest) and os.path.getsize(dest) == total:
        return dest, 0
    hdr = {"Range": "bytes=%d-" % have} if have else {}
    r = _get(f["url"], hdr, stream=True)
    if have and r.status_code != 206:
        have = 0
    t0, done = time.time(), have
    with open(part, "ab" if have else "wb") as out:
        for chunk in r.iter_content(chunk_size=4 << 20):
            out.write(chunk)
            done += len(chunk)
    if os.path.getsize(part) != total:
        raise IOError("%s: got %d of %d bytes; run the download again to continue" % (f["name"], os.path.getsize(part), total))
    os.replace(part, dest)
    return dest, time.time() - t0


def download(cil_id, file=None, members=None, to=None, confirm=False):
    """Download a record's file (by default its ZIP of original data, else its TIFF) into /tmp/cil/<id>,
    or with members=[...] only those files from inside its ZIP. An interrupted download continues where
    it stopped. Larger than 2 GB: shows the size and downloads only with confirm=True."""
    r = _record_or_fail(cil_id)
    f = _pick(r, file)
    dest = to or os.path.join(_scratch(), str(r["id"]))
    os.makedirs(dest, exist_ok=True)
    print("## Download: CIL_%d\n" % r["id"])
    print("Terms of use: %s.\n" % TERMS.get(r["terms"], r["terms"] or "not stated"))
    if members:
        with zipfile.ZipFile(_remote(f)) as z:
            want = [z.getinfo(m) for m in members]
            if not _guard(sum(i.file_size for i in want), dest, confirm, "%d files from %s" % (len(want), f["name"])):
                return {"id": r["id"], "downloaded": False}
            t0 = time.time()
            out = [z.extract(i, dest) for i in want]
        print("Extracted %d files from `%s` without downloading the rest, in %.0f s, to `%s`." % (
            len(out), f["name"], time.time() - t0, dest))
        return {"id": r["id"], "downloaded": True, "folder": dest, "files": out}
    if not _guard(f.get("bytes"), dest, confirm, f["name"]):
        return {"id": r["id"], "downloaded": False, "bytes": f.get("bytes")}
    path, took = _fetch(f, os.path.join(dest, f["name"]))
    print("| File | Size | Time | Saved to |\n|---|---|---|---|")
    print("| %s | %s | %s | `%s` |" % (f["name"], _size(os.path.getsize(path)),
                                      "already here" if not took else "%.0f s" % took, path))
    return {"id": r["id"], "downloaded": True, "folder": dest, "path": path}


def _grey(a):
    """2-D greyscale from a 2-D image or an RGB(A) image (channels last)."""
    import numpy as np
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        rgb = a[..., :3]
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
            return rgb[..., 0]
        return (rgb @ np.array([0.299, 0.587, 0.114])).round().astype(a.dtype)
    return a


def _normalize(folder):
    """One 2-D greyscale PNG or TIFF per slice, as CDeep3M expects: split multi-page TIFFs, turn colour
    into grey, and convert JPEGs (CDeep3M reads PNG and TIFF only). Names keep their sorted order."""
    import numpy as np
    import tifffile
    from PIL import Image
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        low, stem = name.lower(), os.path.splitext(p)[0]
        if not low.endswith(IMAGE_EXT):
            continue
        if low.endswith((".tif", ".tiff")):
            with tifffile.TiffFile(p) as t:
                arr = t.asarray()
            arr = np.squeeze(arr)
            stack = arr.ndim == 4 or (arr.ndim == 3 and arr.shape[-1] not in (3, 4))
            if not stack:
                g = _grey(arr)
                if g is not arr:
                    tifffile.imwrite(p, g)
                continue
            for k in range(arr.shape[0]):
                tifffile.imwrite("%s_z%05d.tif" % (stem, k), _grey(arr[k]))
            os.remove(p)
        else:
            im = Image.open(p)
            if low.endswith((".jpg", ".jpeg")) or im.mode not in ("L", "I;16", "I"):
                im.convert("L").save(stem + ".png")
                if not low.endswith(".png"):
                    os.remove(p)


def slices(cil_id, to=None, confirm=False):
    """A folder of 2-D image slices from a record, ready for segmentation: the images inside its ZIP, or
    the pages of its TIFF (a multi-page TIFF becomes one file per slice). Returns the folder and the
    pixel size as recorded."""
    r = _record_or_fail(cil_id)
    out = to or os.path.join(_scratch(), str(r["id"]), "slices")
    done = sorted(x for x in os.listdir(out) if x.lower().endswith(IMAGE_EXT)) if os.path.isdir(out) else []
    if not done:
        files = _files(r)
        zf = [f for f in files if f["name"].lower().endswith(".zip") and not f.get("extra")]
        tf = [f for f in files if f["name"].lower().endswith((".tif", ".tiff"))]
        src = None
        if zf:
            with zipfile.ZipFile(_remote(zf[0])) as z:
                imgs = sorted(i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(IMAGE_EXT)
                              and not os.path.basename(i.filename).startswith("."))
                if imgs:
                    src = zf[0]
                    if not _guard(sum(i.file_size for i in imgs), _scratch(), confirm,
                                  "%d images from %s" % (len(imgs), zf[0]["name"])):
                        return {"id": r["id"], "ready": False}
                    os.makedirs(out, exist_ok=True)
                    for k, i in enumerate(imgs):
                        name = "%05d_%s" % (k, os.path.basename(i.filename))
                        with z.open(i) as a, open(os.path.join(out, name), "wb") as b:
                            shutil.copyfileobj(a, b, 4 << 20)
                    _normalize(out)
        if src is None and tf:
            src = tf[0]
            if not _guard(src.get("bytes"), _scratch(), confirm, src["name"]):
                return {"id": r["id"], "ready": False}
            got = download(r["id"], file=src["name"], confirm=True)
            os.makedirs(out, exist_ok=True)
            shutil.copy(got["path"], os.path.join(out, "00000_" + src["name"]))
            _normalize(out)
        if src is None:
            raise FileNotFoundError("CIL_%d has no ZIP of images and no TIFF to make slices from" % r["id"])
        done = sorted(x for x in os.listdir(out) if x.lower().endswith(IMAGE_EXT))
    print("## Slices: CIL_%d\n" % r["id"])
    print("| | |\n|---|---|")
    print("| Slices | %d in `%s` |" % (len(done), out))
    print("| Pixel size (as recorded) | %s |" % _px(r))
    print("| Imaging | %s |" % (r["imaging_mode"] or "not stated"))
    print("| Terms of use | %s |" % TERMS.get(r["terms"], r["terms"] or "not stated"))
    return {"id": r["id"], "ready": True, "folder": out, "slices": len(done),
            "pixel_nm": [r["pixel_nm_x"], r["pixel_nm_y"], r["pixel_nm_z"]], "imaging": r["imaging_mode"],
            "component": r["component"], "terms": r["terms"]}
