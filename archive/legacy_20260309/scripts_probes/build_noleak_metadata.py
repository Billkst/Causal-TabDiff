import argparse
import json
from pathlib import Path


LEAKY_EXACT = {
    "can_scr",
}

LEAKY_PREFIX = (
    "scr_",
    "loc",
    "lesion",
)


def is_leaky(name: str) -> bool:
    if name in LEAKY_EXACT:
        return True
    return any(name.startswith(prefix) for prefix in LEAKY_PREFIX)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default="src/data/dataset_metadata.json")
    parser.add_argument("--dst", type=str, default="src/data/dataset_metadata_noleak.json")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    meta = json.loads(src.read_text(encoding="utf-8"))

    removed = [col["name"] for col in meta["columns"] if is_leaky(col["name"])]
    keep = [col["name"] for col in meta["columns"] if not is_leaky(col["name"])]

    meta_new = {
        "columns": [c for c in meta["columns"] if c["name"] in keep],
        "continuous": [c for c in meta.get("continuous", []) if c["name"] in keep],
        "categorical": [c for c in meta.get("categorical", []) if c["name"] in keep],
        "y_col": meta["y_col"],
        "feature_policy": {
            "name": "noleak_v1",
            "removed_exact": sorted(LEAKY_EXACT),
            "removed_prefix": list(LEAKY_PREFIX),
            "removed_columns": removed,
            "kept_columns": keep,
        },
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(meta_new, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    print(f"Wrote: {dst}")
    print(f"kept={len(keep)}, removed={len(removed)}")
    print("removed:", ", ".join(removed))


if __name__ == "__main__":
    main()
