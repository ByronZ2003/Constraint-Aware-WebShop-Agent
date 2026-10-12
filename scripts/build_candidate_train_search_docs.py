#!/usr/bin/env python3

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web_agent_site.engine import engine


PRODUCTS = (
    ROOT
    / "data/candidate_train/"
      "items_shuffle_candidate_train_1000.json"
)

ATTRS = (
    ROOT
    / "data/candidate_train/"
      "items_ins_v2_candidate_train_1000.json"
)

HUMAN_ATTRS = (
    ROOT
    / "data/items_human_ins.json"
)

OUT_DIR = (
    ROOT
    / "data/candidate_train/search_resources"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

engine.DEFAULT_ATTR_PATH = str(ATTRS)
engine.HUMAN_ATTR_PATH = str(HUMAN_ATTRS)

products, *_ = engine.load_products(
    filepath=str(PRODUCTS),
    num_products=1000,
    human_goals=False,
)

if len(products) != 1000:
    raise RuntimeError(
        f"Expected 1000 products, got {len(products)}"
    )

out_path = OUT_DIR / "documents.jsonl"

with out_path.open(
    "w",
    encoding="utf-8",
) as f:

    for p in products:

        option_texts = []

        for option_name, values in (
            p.get("options") or {}
        ).items():

            option_texts.append(
                f"{option_name}: {', '.join(values)}"
            )

        option_text = ", and ".join(
            option_texts
        )

        bullet_points = (
            p.get("BulletPoints")
            or [""]
        )

        first_bullet = (
            bullet_points[0]
            if bullet_points
            else ""
        )

        contents = " ".join([
            str(p.get("Title") or ""),
            str(p.get("Description") or ""),
            str(first_bullet or ""),
            str(option_text or ""),
        ]).lower()

        doc = {
            "id": p["asin"],
            "contents": contents,
            "product": p,
        }

        f.write(
            json.dumps(
                doc,
                ensure_ascii=False,
            )
            + "\n"
        )

print("SEARCH DOCUMENTS CREATED")
print("products:", len(products))
print("output:", out_path)
