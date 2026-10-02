# Blind-Earth land/water experiment — run log

## Method
Recipe: Henry / outsidetext — [How Does A Blind Model See The Earth?](https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth)

For each lat/lon on a 2° grid, ask a System One `choice` question (Land vs Water) and plot P(Land).

**Primary PNGs:** hard black-and-white equirectangular posters — **white = Land** (P(Land) > 0.5), **black = Water**. Soft grayscale P(Land) maps saved under `maps/*_soft_grayscale.png`.

## Grid definition
- Latitudes: `-89, -87, …, +89` step 2 → **90** values
- Longitudes: `-179, -177, …, +179` step 2 → **180** values
- Total points: **90 × 180 = 16,200**
- Equirectangular: x=lon, y=lat (north up in PNGs)
- Batch size: 64 questions/request → **254 batches/model**

## Sample API payload (Clef)
```json
{
  "model": "clef",
  "state": "Answer geographic land/water questions about Earth coordinates. ...",
  "questions": {
    "p_65_52": {
      "type": "choice",
      "instructions": "Is the location at 41°N, 73°W over land or over water?",
      "criteria": {
        "Land": "Over land, ice, or snow",
        "Water": "Over ocean, sea, or other water"
      }
    }
  }
}
```

Endpoints (credentials via env only):
- Clef: `POST https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/run/@cf/cloudflare/clef` with `model: "clef"`
- Clef-flash: same path `@cf/cloudflare/clef-flash` with `model: "clef-flash"`
- Jev: `POST https://api.typesafe.ai/v1/systemone` with `model: "jev-latest"` (`TYPESAFE_API_KEY`)

## Timing (wall clock, parallel HTTP workers=8)
| Model | Wall time | Batches | Filled |
|-------|-----------|---------|--------|
| clef | ~48.5 s | 254 | 16200/16200 |
| clef-flash | ~27.7 s | 254 | 16200/16200 |
| jev | ~5.0 s | 254 | 16200/16200 |

## Hard land fraction (P(Land) > 0.5) and mean P(Land)
| Model | Hard land fraction | Mean P(Land) |
|-------|-------------------:|-------------:|
| clef | **0.383** | 0.416 |
| clef-flash | **0.511** | 0.487 |
| jev | **0.487** | 0.484 |

Earth’s true land fraction is ~0.29; all three models over-predict land on this equirectangular grid (esp. poles / high latitudes).

## Usage totals (from API `usage` fields)
| Model | input_tokens | output_tokens | calls |
|-------|-------------:|--------------:|------:|
| clef | 1634899 | 0 | 254 |
| clef-flash | 1634899 | 0 | 254 |
| jev | 1294176 | 556062 | 254 |

## Outputs
- `blind_earth_clef.png` — hard B/W (recognizable continents; some ocean salt-and-pepper)
- `blind_earth_clef_flash.png` — hard B/W (noisier; rough landmass bands)
- `blind_earth_jev.png` — hard B/W (continents visible; more ocean false-land than Clef; Antarctica as a thick white band)
- `grids/*_land_probs.npz` — `land_probs` (90,180) + `lats`/`lons`
- `progress/*_progress.jsonl` — resume-friendly per-point records (regenerate; not shipped in the public how-to repo)
- `maps/*_soft_grayscale.png` — soft P(Land) grayscale

## Visual notes
- **Clef:** Americas, Africa, Eurasia, Australia clearly visible; Arctic over-predicted as land; scattered false land in Pacific. Lowest hard land fraction of the three (~38%).
- **Clef-flash:** More speckled; land concentrated in correct longitudinal bands; coastlines weaker. Highest hard land fraction (~51%).
- **Jev:** Continents recognizable; Afro-Eurasia tends to merge; Antarctica as a broad southern band; more Pacific false-land speckles than Clef. Hard land ~49%.

## Resume / re-run
```bash
export CLOUDFLARE_ACCOUNT_ID=...          # Cloudflare Workers AI
export CLOUDFLARE_AUTH_TOKEN=...
export TYPESAFE_API_KEY=...               # TypeSafe / Jev
cd blind-earth   # or clone of dylanler/blind-earth-clef-jev
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib requests
python3 run_blind_earth.py --model jev --workers 8
# or: --model all
```

Generated PT: 2026-10-02 ~01:17 America/Los_Angeles
