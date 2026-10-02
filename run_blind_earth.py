#!/usr/bin/env python3
"""Blind-Earth land/water experiment (Henry / outsidetext recipe) for Clef, Clef-flash, Jev."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent
PROGRESS = ROOT / "progress"
GRIDS = ROOT / "grids"
MAPS = ROOT / "maps"
for d in (PROGRESS, GRIDS, MAPS):
    d.mkdir(parents=True, exist_ok=True)

ACCOUNT = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
if not ACCOUNT:
    # Required for Cloudflare models; Jev does not need it.
    ACCOUNT = ""
CF_TOKEN = os.environ.get("CLOUDFLARE_AUTH_TOKEN")
TS_KEY = os.environ.get("TYPESAFE_API_KEY")

BATCH = 64
MAX_WORKERS = 12
MAX_RETRIES = 8
STATE_INSTR = (
    "Answer geographic land/water questions about Earth coordinates. "
    "Land includes continents, islands, ice sheets, and snow-covered ground. "
    "Water includes oceans, seas, and other open water."
)
CRITERIA = {
    "Land": "Over land, ice, or snow",
    "Water": "Over ocean, sea, or other water",
}

def _cf_url(slug: str) -> str:
    if not ACCOUNT:
        return ""
    return f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/@cf/cloudflare/{slug}"


def get_models() -> dict:
    return {
        "clef": {
            "kind": "cloudflare",
            "url": _cf_url("clef"),
            "model": "clef",
        },
        "clef_flash": {
            "kind": "cloudflare",
            "url": _cf_url("clef-flash"),
            "model": "clef-flash",
        },
        "jev": {
            "kind": "typesafe",
            "url": "https://api.typesafe.ai/v1/systemone",
            "model": "jev-latest",
        },
    }


MODELS = get_models()  # rebuilt after ACCOUNT is read; call get_models() if env changes


def make_grid():
    # 90 x 180 = 16200: lat -89..+89 step 2, lon -179..+179 step 2
    lats = np.arange(-89, 90, 2, dtype=np.int16)  # 90
    lons = np.arange(-179, 180, 2, dtype=np.int16)  # 180
    assert lats.size == 90 and lons.size == 180
    points = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            points.append((int(i), int(j), int(lat), int(lon)))
    assert len(points) == 16200
    return lats, lons, points


def fmt_coord(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat)}°{ns}, {abs(lon)}°{ew}"


def question_for(lat: int, lon: int, qid: str) -> dict:
    return {
        "type": "choice",
        "instructions": f"Is the location at {fmt_coord(lat, lon)} over land or over water?",
        "criteria": CRITERIA,
        "_qid": qid,  # stripped before send; used for dict key
    }


def build_payload(model_cfg: dict, batch_points: list[tuple[int, int, int, int]]) -> tuple[dict, list[str]]:
    questions: dict[str, Any] = {}
    qids = []
    for i, j, lat, lon in batch_points:
        qid = f"p_{i}_{j}"
        q = question_for(lat, lon, qid)
        q.pop("_qid", None)
        questions[qid] = q
        qids.append(qid)
    payload = {
        "model": model_cfg["model"],
        "state": STATE_INSTR,
        "questions": questions,
    }
    return payload, qids


def auth_headers(kind: str) -> dict:
    if kind == "cloudflare":
        if not CF_TOKEN:
            raise RuntimeError("CLOUDFLARE_AUTH_TOKEN not set")
        return {"Authorization": f"Bearer {CF_TOKEN}", "Content-Type": "application/json"}
    if not TS_KEY:
        raise RuntimeError("TYPESAFE_API_KEY not set")
    return {"Authorization": f"Bearer {TS_KEY}", "Content-Type": "application/json"}


def extract_answers(resp: dict, kind: str) -> dict:
    """Return {qid: {Land: p, Water: p, ...}} plus usage."""
    if kind == "cloudflare":
        result = resp.get("result", resp)
    else:
        result = resp
    answers = result.get("answers") or {}
    usage = result.get("usage") or resp.get("usage") or {}
    out = {}
    for qid, ans in answers.items():
        probs = ans.get("probabilities")
        if not isinstance(probs, dict):
            continue
        out[qid] = {k: float(v) for k, v in probs.items()}
    return out, usage


def post_with_retries(url: str, headers: dict, payload: dict, kind: str) -> tuple[dict, dict]:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=180)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(60, (2**attempt) + (0.1 * attempt))
                time.sleep(wait)
                last_err = f"HTTP {r.status_code}"
                continue
            if r.status_code >= 400:
                # non-retryable
                try:
                    body = r.json()
                except Exception:
                    body = {"_text": r.text[:500]}
                raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(body)[:800]}")
            data = r.json()
            if kind == "cloudflare" and data.get("success") is False:
                errs = data.get("errors") or data
                # retry on upstream blips
                wait = min(60, (2**attempt))
                time.sleep(wait)
                last_err = f"success=false {errs}"
                continue
            answers, usage = extract_answers(data, kind)
            return answers, usage
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(min(60, 2**attempt))
    raise RuntimeError(f"failed after retries: {last_err}")


def progress_path(model_key: str) -> Path:
    return PROGRESS / f"{model_key}_progress.jsonl"


def usage_path(model_key: str) -> Path:
    return PROGRESS / f"{model_key}_usage.json"


def load_done(model_key: str) -> set[str]:
    path = progress_path(model_key)
    done = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "qid" in rec and "p_land" in rec:
                done.add(rec["qid"])
    return done


def append_records(model_key: str, records: list[dict]):
    path = progress_path(model_key)
    with path.open("a") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")


def add_usage(model_key: str, usage: dict):
    path = usage_path(model_key)
    cur = {"input_tokens": 0, "output_tokens": 0, "batches": 0, "calls": 0}
    if path.exists():
        try:
            cur = json.loads(path.read_text())
        except Exception:
            pass
    cur["input_tokens"] = int(cur.get("input_tokens", 0)) + int(usage.get("input_tokens") or 0)
    cur["output_tokens"] = int(cur.get("output_tokens", 0)) + int(usage.get("output_tokens") or 0)
    cur["batches"] = int(cur.get("batches", 0)) + 1
    cur["calls"] = int(cur.get("calls", 0)) + 1
    path.write_text(json.dumps(cur, indent=2))


def run_model(model_key: str, workers: int = MAX_WORKERS):
    MODELS.update(get_models())
    cfg = MODELS[model_key]
    if cfg["kind"] == "cloudflare" and not cfg["url"]:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID not set")
    lats, lons, points = make_grid()
    done = load_done(model_key)
    remaining = [(i, j, lat, lon) for (i, j, lat, lon) in points if f"p_{i}_{j}" not in done]
    print(f"[{model_key}] done={len(done)} remaining={len(remaining)} total=16200", flush=True)
    if not remaining:
        return assemble(model_key, lats, lons)

    # chunk into batches of BATCH
    batches = [remaining[k : k + BATCH] for k in range(0, len(remaining), BATCH)]
    print(f"[{model_key}] batches_to_run={len(batches)} workers={workers}", flush=True)

    headers = auth_headers(cfg["kind"])
    t0 = time.time()
    completed_batches = 0
    errors = []

    def do_batch(batch):
        payload, qids = build_payload(cfg, batch)
        answers, usage = post_with_retries(cfg["url"], headers, payload, cfg["kind"])
        records = []
        missing = []
        for i, j, lat, lon in batch:
            qid = f"p_{i}_{j}"
            probs = answers.get(qid)
            if not probs or "Land" not in probs:
                missing.append(qid)
                continue
            p_land = float(probs["Land"])
            records.append(
                {
                    "qid": qid,
                    "i": i,
                    "j": j,
                    "lat": lat,
                    "lon": lon,
                    "p_land": p_land,
                    "p_water": float(probs.get("Water", 1.0 - p_land)),
                }
            )
        return records, usage, missing

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(do_batch, b): bi for bi, b in enumerate(batches)}
        for fut in as_completed(futs):
            bi = futs[fut]
            try:
                records, usage, missing = fut.result()
                if missing:
                    # retry missing one-by-one once
                    retry_pts = [
                        (i, j, lat, lon)
                        for (i, j, lat, lon) in batches[bi]
                        if f"p_{i}_{j}" in missing
                    ]
                    if retry_pts:
                        rec2, usage2, missing2 = do_batch(retry_pts)
                        records.extend(rec2)
                        usage = {
                            "input_tokens": int(usage.get("input_tokens") or 0)
                            + int(usage2.get("input_tokens") or 0),
                            "output_tokens": int(usage.get("output_tokens") or 0)
                            + int(usage2.get("output_tokens") or 0),
                        }
                        if missing2:
                            raise RuntimeError(f"still missing {missing2[:5]}...")
                append_records(model_key, records)
                add_usage(model_key, usage)
                completed_batches += 1
                if completed_batches % 10 == 0 or completed_batches == len(batches):
                    elapsed = time.time() - t0
                    print(
                        f"[{model_key}] batches {completed_batches}/{len(batches)} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
            except Exception as e:
                errors.append((bi, str(e)))
                print(f"[{model_key}] BATCH ERROR {bi}: {e}", flush=True)

    if errors:
        print(f"[{model_key}] {len(errors)} batch errors; will leave gaps for resume", flush=True)
    return assemble(model_key, lats, lons)


def assemble(model_key: str, lats=None, lons=None):
    if lats is None:
        lats, lons, _ = make_grid()
    grid = np.full((len(lats), len(lons)), np.nan, dtype=np.float32)
    path = progress_path(model_key)
    n = 0
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                i, j = int(rec["i"]), int(rec["j"])
                grid[i, j] = float(rec["p_land"])
                n += 1
    filled = int(np.isfinite(grid).sum())
    print(f"[{model_key}] assemble records={n} filled={filled}/16200", flush=True)
    out_npz = GRIDS / f"{model_key}_land_probs.npz"
    np.savez_compressed(out_npz, land_probs=grid, lats=lats, lons=lons)
    # also npy
    np.save(GRIDS / f"{model_key}_land_probs.npy", grid)
    np.save(GRIDS / f"{model_key}_lats.npy", lats)
    np.save(GRIDS / f"{model_key}_lons.npy", lons)
    render_maps(model_key, grid, lats, lons)
    return filled


def render_maps(model_key: str, grid: np.ndarray, lats, lons):
    # Primary: hard B/W — white=Land, black=Water (P(Land)>0.5)
    hard = np.zeros_like(grid, dtype=np.float32)
    known = np.isfinite(grid)
    hard[known & (grid > 0.5)] = 1.0
    hard[known & (grid <= 0.5)] = 0.0
    # NaNs stay 0 (black) but we'll mark in log; better: leave as mid gray for missing
    hard_img = hard.copy()
    hard_img[~known] = 0.5

    titles = {
        "clef": "Clef — blind Earth (Land vs Water)",
        "clef_flash": "Clef-flash — blind Earth (Land vs Water)",
        "jev": "Jev — blind Earth (Land vs Water)",
    }
    # Flip vertically so north is up: grid row 0 is lat=-89 (south)
    hard_show = np.flipud(hard_img)
    soft_show = np.flipud(np.where(known, grid, 0.5))

    # Primary hard B/W PNG (requested names at ROOT)
    name_map = {
        "clef": "blind_earth_clef.png",
        "clef_flash": "blind_earth_clef_flash.png",
        "jev": "blind_earth_jev.png",
    }
    primary = ROOT / name_map[model_key]
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.imshow(
        hard_show,
        # hard=1 means land; gray maps 1 to white and 0 to black.
        cmap="gray",
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
        extent=[-180, 180, -90, 90],
        origin="upper",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(titles[model_key] + " [hard B/W: white=Land]")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    fig.tight_layout()
    fig.savefig(primary, facecolor="white")
    # also copy under maps/
    fig.savefig(MAPS / name_map[model_key], facecolor="white")
    plt.close(fig)

    # Secondary soft grayscale
    soft = MAPS / f"{model_key}_soft_grayscale.png"
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    im = ax.imshow(
        soft_show,
        cmap="gray",
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
        extent=[-180, 180, -90, 90],
        origin="upper",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(titles[model_key] + " [P(Land) soft]")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="P(Land)")
    fig.tight_layout()
    fig.savefig(soft, facecolor="white")
    plt.close(fig)
    print(f"[{model_key}] wrote {primary} and {soft}", flush=True)


def sample_payload_doc() -> dict:
    pts = [(44, 53, -1, -73)]  # near equator? i,j dummy — use NYC-ish
    # find NYC-ish on grid: 41N ~ 41, -73W
    lats, lons, points = make_grid()
    # pick first point near 41,-73
    best = min(points, key=lambda p: abs(p[2] - 41) + abs(p[3] - (-73)))
    payload, _ = build_payload(get_models()["clef"], [best])
    # shrink to one Q for doc
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(get_models()) + ["all"], default="all")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    MODELS.clear()
    MODELS.update(get_models())
    models = list(MODELS) if args.model == "all" else [args.model]
    t0 = time.time()
    results = {}
    for m in models:
        if m == "jev" and not TS_KEY:
            print("[jev] SKIP: TYPESAFE_API_KEY not set", flush=True)
            results[m] = "skipped_no_key"
            continue
        if m != "jev" and not CF_TOKEN:
            print(f"[{m}] SKIP: CLOUDFLARE_AUTH_TOKEN not set", flush=True)
            results[m] = "skipped_no_token"
            continue
        try:
            if args.assemble_only:
                filled = assemble(m)
            else:
                filled = run_model(m, workers=args.workers)
            results[m] = f"filled={filled}"
        except Exception as e:
            traceback.print_exc()
            results[m] = f"error:{e}"
    print("DONE", json.dumps(results), f"wall_s={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
