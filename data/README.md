# Data

This repository ships no data. Place your event JSON files here.

## Expected schema

Each event is a single JSON file. The pipeline expects this structure:

```json
{
  "id": 578987,
  "printer": "MACHINE_ID",
  "date_time_str": "20210705_083658",
  "date": "05.07.2021",
  "time": "08:36:58",
  "speed": "2.1 m/s",
  "grammage_weight": "52 g/m2",
  "web_width": 1630.0,
  "pap_len": "1639 m",
  "detector": 16,
  "grade": "...",
  "paper_supplier": "...",

  "videos": [
    {
      "camera_id": "a",
      "camera_name": "Seite A",
      "frames": [
        {
          "name": "0001.jpg",
          "label": "no_defect",
          "scores": {
            "no_defect": 0.9195,
            "Kantenfehler": 0.0490,
            "rollenwechsel": 0.0145,
            "defect": 0.0137,
            "tear": 0.0031
          }
        }
      ]
    }
  ]
}
```

## Field reference

| Field | Type | Required | Notes |
|---|---|---|---|
| `speed` | string or float | Yes | Parsed with unit stripping e.g. `"2.1 m/s"` → `2.1` |
| `grammage_weight` | string or float | Yes | e.g. `"52 g/m2"` → `52.0` |
| `web_width` | float | Yes | Raw float, no units |
| `pap_len` | string or float | Yes | e.g. `"1639 m"` → `1639.0` |
| `detector` | int | Yes | Detector position identifier |
| `printer` | string | Yes | Machine identifier — one-hot encoded |
| `grade` | string | Yes | Paper grade — one-hot encoded |
| `paper_supplier` | string | Yes | Supplier name — one-hot encoded |
| `videos` | array | Yes | At least one camera with ≥10 valid frames |
| `videos[].frames[].scores` | object | Yes | Must contain all 5 keys: `no_defect`, `defect`, `rollenwechsel`, `Kantenfehler`, `tear` |

## Label file format (training only)

For training, each ZIP must contain a `labels.txt` file:

```
1779100886202_FO52_20210705_083658_event  0
1779100886203_FO52_20210705_091234_event  1
```

Format: `<event_id>  <label>` where `0 = machine_problem`, `1 = paper_problem`.
Delimiters: space, comma, semicolon, or colon — all accepted.

## Directory layout

```
data/
├── README.md           ← you are here
├── events/             ← place your *_event.json files here (gitignored)
│   ├── 2021_07_events.zip
│   └── loose_event.json
└── new_events/
    ├── input/          ← drop monthly ZIPs or loose JSONs for scoring
    └── output/         ← predictions written here automatically
```

## Score-mode input

Drop monthly ZIP batches or loose JSON files into `data/new_events/input/` and run:

```bash
python -m src.cli score
```

Both formats are supported simultaneously in the same folder.
