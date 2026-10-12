#!/usr/bin/env python3

import argparse
import hashlib
import json
import random
from pathlib import Path

import ijson


def valid_instruction(meta):
    if not isinstance(meta, dict):
        return False

    instruction = meta.get("instruction")
    attrs = meta.get("instruction_attributes")

    return bool(
        isinstance(instruction, str)
        and instruction.strip()
        and isinstance(attrs, list)
        and len(attrs) > 0
    )


def reservoir_add(reservoir, value, seen, k, rng):
    if len(reservoir) < k:
        reservoir.append(value)
        return

    j = rng.randrange(seen)

    if j < k:
        reservoir[j] = value


def iter_attrs(path):
    with open(path, "rb") as f:
        yield from ijson.kvitems(f, "", use_float=True)


def iter_products(path):
    with open(path, "rb") as f:
        yield from ijson.items(f, "item", use_float=True)


def product_asin(item):
    if not isinstance(item, dict):
        return ""

    return str(
        item.get("asin")
        or item.get("ASIN")
        or ""
    ).upper()


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--full-products", required=True)
    ap.add_argument("--full-attrs", required=True)
    ap.add_argument("--small-attrs", required=True)
    ap.add_argument("--eval-tasks", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--targets", type=int, default=415)
    ap.add_argument("--products", type=int, default=1000)

    args = ap.parse_args()

    distractors_needed = args.products - args.targets

    if distractors_needed < 0:
        raise SystemExit("products must be >= targets")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    small = json.loads(
        Path(args.small_attrs).read_text(
            encoding="utf-8"
        )
    )

    eval_tasks = json.loads(
        Path(args.eval_tasks).read_text(
            encoding="utf-8"
        )
    )

    small_asins = {
        str(x).upper()
        for x in small.keys()
    }

    eval_asins = {
        str(x["task_id"]).upper()
        for x in eval_tasks
    }

    excluded = small_asins | eval_asins

    print("Seed:", args.seed)
    print("Excluded small ASINs:", len(small_asins))
    print("Excluded eval ASINs:", len(eval_asins))
    print("Combined excluded:", len(excluded))
    print()

    # ========================================================
    # PASS 1
    #
    # Reservoir-sample:
    #   415 instruction-bearing targets
    #   585 non-instruction distractors
    # ========================================================

    targets = []
    distractors = []

    target_seen = 0
    distractor_seen = 0
    scanned = 0

    print("PASS 1: sampling ASINs from full attributes")

    for asin, meta in iter_attrs(args.full_attrs):
        scanned += 1

        asin = str(asin).upper()

        if scanned % 100000 == 0:
            print(
                "  scanned:",
                scanned,
                "targets_seen:",
                target_seen,
                "distractors_seen:",
                distractor_seen,
                flush=True,
            )

        if asin in excluded:
            continue

        if valid_instruction(meta):
            target_seen += 1

            reservoir_add(
                targets,
                asin,
                target_seen,
                args.targets,
                rng,
            )

        else:
            distractor_seen += 1

            reservoir_add(
                distractors,
                asin,
                distractor_seen,
                distractors_needed,
                rng,
            )

    print()
    print("Full attribute rows scanned:", scanned)
    print("Eligible target rows:", target_seen)
    print("Eligible distractor rows:", distractor_seen)

    if len(targets) != args.targets:
        raise SystemExit(
            f"Need {args.targets} targets, "
            f"got {len(targets)}"
        )

    if len(distractors) != distractors_needed:
        raise SystemExit(
            f"Need {distractors_needed} distractors, "
            f"got {len(distractors)}"
        )

    target_set = set(targets)
    distractor_set = set(distractors)

    if target_set & distractor_set:
        raise SystemExit(
            "Target/distractor overlap detected"
        )

    selected = target_set | distractor_set

    if len(selected) != args.products:
        raise SystemExit(
            f"Expected {args.products} unique selected ASINs, "
            f"got {len(selected)}"
        )

    # ========================================================
    # PASS 2
    # Collect complete metadata for selected 1000 ASINs
    # ========================================================

    print()
    print("PASS 2: collecting selected attribute records")

    attrs_out = {}

    scanned = 0

    for asin, meta in iter_attrs(args.full_attrs):
        scanned += 1

        asin = str(asin).upper()

        if asin in selected:
            attrs_out[asin] = meta

        if len(attrs_out) == len(selected):
            break

        if scanned % 200000 == 0:
            print(
                "  scanned:",
                scanned,
                "found:",
                len(attrs_out),
                flush=True,
            )

    if set(attrs_out.keys()) != selected:
        missing = sorted(
            selected - set(attrs_out.keys())
        )

        raise SystemExit(
            "Missing attrs: "
            + repr(missing[:20])
        )

    # ========================================================
    # PASS 3
    # Scan 5.2GB full product file and collect selected records
    # ========================================================

    print()
    print("PASS 3: collecting selected full product records")

    products_out = []
    found_products = set()

    scanned_products = 0

    for item in iter_products(args.full_products):
        scanned_products += 1

        if scanned_products % 100000 == 0:
            print(
                "  scanned:",
                scanned_products,
                "found:",
                len(found_products),
                flush=True,
            )

        asin = product_asin(item)

        if (
            asin in selected
            and asin not in found_products
        ):
            products_out.append(item)
            found_products.add(asin)

            if len(found_products) == len(selected):
                break

    if found_products != selected:
        missing = sorted(
            selected - found_products
        )

        raise SystemExit(
            "Missing products: "
            + repr(missing[:20])
        )

    # ========================================================
    # Build exact 415 task list
    # ========================================================

    tasks = []

    for asin in targets:
        meta = attrs_out[asin]

        tasks.append({
            "task_id": asin,
            "instruction": meta["instruction"].strip(),
            "instruction_attributes": list(
                meta["instruction_attributes"]
            ),
        })

    rng.shuffle(products_out)
    rng.shuffle(tasks)

    # ========================================================
    # Strict validation
    # ========================================================

    product_asins = {
        product_asin(x)
        for x in products_out
    }

    attr_asins = set(attrs_out.keys())

    task_asins = {
        x["task_id"]
        for x in tasks
    }

    instruction_asins = {
        asin
        for asin, meta in attrs_out.items()
        if valid_instruction(meta)
    }

    small_overlap = (
        product_asins
        & small_asins
    )

    eval_overlap = (
        product_asins
        & eval_asins
    )

    assert len(products_out) == 1000
    assert len(product_asins) == 1000
    assert len(attrs_out) == 1000
    assert len(tasks) == 415

    assert product_asins == attr_asins

    assert task_asins == target_set

    # Because distractors were sampled only from
    # non-instruction records, this must be exactly 415.
    assert instruction_asins == target_set

    assert len(instruction_asins) == 415

    assert not small_overlap
    assert not eval_overlap

    # ========================================================
    # Output files
    # ========================================================

    products_path = (
        out
        / "items_shuffle_candidate_train_1000.json"
    )

    attrs_path = (
        out
        / "items_ins_v2_candidate_train_1000.json"
    )

    tasks_path = (
        out
        / "webshop_candidate_train_415.json"
    )

    products_path.write_text(
        json.dumps(
            products_out,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    attrs_path.write_text(
        json.dumps(
            attrs_out,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tasks_path.write_text(
        json.dumps(
            tasks,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "seed": args.seed,

        "sampling": {
            "type": "deterministic_stratified_reservoir",
            "targets": 415,
            "distractors": 585,
            "total_products": 1000,
        },

        "counts": {
            "products": len(products_out),
            "attributes": len(attrs_out),
            "tasks": len(tasks),
            "instruction_bearing_products": len(
                instruction_asins
            ),
        },

        "source_pool": {
            "eligible_instruction_products": target_seen,
            "eligible_non_instruction_products": distractor_seen,
        },

        "overlap": {
            "with_webshop_small": len(small_overlap),
            "with_eval_415": len(eval_overlap),
            "target_with_eval_415": len(
                target_set & eval_asins
            ),
        },

        "files": {
            products_path.name: sha256(products_path),
            attrs_path.name: sha256(attrs_path),
            tasks_path.name: sha256(tasks_path),
        },
    }

    manifest_path = (
        out
        / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("CANDIDATE TRAIN DATASET CREATED")
    print("=" * 60)
    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
