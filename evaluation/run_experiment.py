import argparse
import json
import shutil
import statistics
import traceback
from pathlib import Path

from agent.gated_agent import GatedAgent


DATASET = "evaluation/webshop_test_100.json"


def str_to_bool(value):
    value = value.strip().lower()

    if value in {"on", "true", "1", "yes"}:
        return True

    if value in {"off", "false", "0", "no"}:
        return False

    raise argparse.ArgumentTypeError(
        "Expected one of: on/off, true/false, 1/0, yes/no"
    )


def load_tasks(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []

    # 当前 WebShop evaluation 文件主要是 dict:
    # {
    #   "ASIN": {
    #       "instruction": "...",
    #       "instruction_attributes": [...]
    #   }
    # }
    if isinstance(data, dict):

        for task_id, item in data.items():

            if not isinstance(item, dict):
                continue

            instruction = item.get("instruction")

            if not isinstance(instruction, str):
                continue

            instruction = instruction.strip()

            if not instruction:
                continue

            tasks.append(
                {
                    "task_id": str(task_id),
                    "instruction": instruction,

                    # 只保存在 runner 侧，用于离线记录。
                    # 不会传给 Agent / Gate。
                    "instruction_attributes": item.get(
                        "instruction_attributes",
                        [],
                    ),
                }
            )

    elif isinstance(data, list):

        for i, item in enumerate(data):

            if not isinstance(item, dict):
                continue

            instruction = item.get("instruction")

            if not isinstance(instruction, str):
                continue

            instruction = instruction.strip()

            if not instruction:
                continue

            task_id = (
                item.get("task_id")
                or item.get("asin")
                or item.get("ASIN")
                or str(i)
            )

            tasks.append(
                {
                    "task_id": str(task_id),
                    "instruction": instruction,
                    "instruction_attributes": item.get(
                        "instruction_attributes",
                        [],
                    ),
                }
            )

    return tasks


def action_list(trajectory):
    result = []

    for step in trajectory:
        action = step.get("action")

        if isinstance(action, str):
            result.append(action.strip().lower())

    return result


def target_was_visited(task_id, trajectory):
    target = f"click[{task_id.lower()}]"

    return any(
        action == target
        for action in action_list(trajectory)
    )


def count_action(trajectory, target):
    target = target.lower()

    return sum(
        1
        for action in action_list(trajectory)
        if action == target
    )


def aggregate_gate_stats(records):
    totals = {}

    for record in records:

        stats = record.get("gate_stats") or {}

        for key, value in stats.items():

            if isinstance(value, bool):
                continue

            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value

    return totals


def compute_summary(records):
    valid = [
        r for r in records
        if r.get("outcome") != "ERROR"
    ]

    errors = [
        r for r in records
        if r.get("outcome") == "ERROR"
    ]

    full = [
        r for r in valid
        if float(r.get("reward", 0.0)) == 1.0
    ]

    partial = [
        r for r in valid
        if 0.0 < float(r.get("reward", 0.0)) < 1.0
    ]

    zero = [
        r for r in valid
        if float(r.get("reward", 0.0)) == 0.0
    ]

    reward_gt_0 = [
        r for r in valid
        if float(r.get("reward", 0.0)) > 0.0
    ]

    target_visited = [
        r for r in valid
        if r.get("target_visited")
    ]

    target_visited_but_zero = [
        r for r in zero
        if r.get("target_visited")
    ]

    target_not_visited_and_zero = [
        r for r in zero
        if not r.get("target_visited")
    ]

    rewards = [
        float(r.get("reward", 0.0))
        for r in valid
    ]

    steps = [
        int(r.get("steps", 0))
        for r in valid
    ]

    summary = {
        "tasks": len(records),
        "valid_tasks": len(valid),
        "errors": len(errors),

        "full_success": len(full),
        "partial_success": len(partial),
        "zero": len(zero),
        "reward_gt_0": len(reward_gt_0),

        "average_reward": (
            sum(rewards) / len(rewards)
            if rewards
            else 0.0
        ),

        "average_steps": (
            sum(steps) / len(steps)
            if steps
            else 0.0
        ),

        "median_steps": (
            statistics.median(steps)
            if steps
            else 0.0
        ),

        "target_visited": len(target_visited),

        "target_visited_but_zero":
            len(target_visited_but_zero),

        "target_not_visited_and_zero":
            len(target_not_visited_and_zero),

        "total_buy_now": sum(
            int(r.get("buy_now_count", 0))
            for r in valid
        ),

        "total_back_to_search": sum(
            int(r.get("back_to_search_count", 0))
            for r in valid
        ),

        "gate_stats": aggregate_gate_stats(valid),
    }

    return summary


def print_summary(summary, title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


def save_summary(output_dir, summary, title):
    output_dir = Path(output_dir)

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        output_dir / "summary.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write("=" * 80 + "\n")
        f.write(title + "\n")
        f.write("=" * 80 + "\n")

        f.write(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )

        f.write("\n")


def make_title(
    use_query_gate,
    use_product_gate,
    use_candidate_gate=False,
):
    if (
        use_query_gate
        and use_candidate_gate
        and use_product_gate
    ):
        return "QUERY + CANDIDATE + PRODUCT GATE"

    if (
        use_candidate_gate
        and use_product_gate
    ):
        return "CANDIDATE + PRODUCT GATE"

    if use_query_gate and use_product_gate:
        return "QUERY + PRODUCT GATE"

    if use_candidate_gate:
        return "CANDIDATE GATE ONLY"

    if use_query_gate:
        return "QUERY GATE ONLY"

    if use_product_gate:
        return "PRODUCT GATE ONLY"

    return "NO GATES — DIAGNOSTIC RUN"


def snapshot_code(output_dir):
    snapshot = Path(output_dir) / "code_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)

    files = [
        "agent/gated_agent.py",
        "agent/baseline_agent.py",
        "gates/constraint_manager.py",
        "gates/query_gate.py",
        "gates/candidate_gate.py",
        "gates/product_gate.py",
    ]

    for filename in files:
        src = Path(filename)

        if src.exists():
            shutil.copy2(
                src,
                snapshot / src.name,
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query-gate",
        type=str_to_bool,
        default=False,
        help="on/off",
    )

    parser.add_argument(
        "--candidate-gate",
        type=str_to_bool,
        default=False,
        help="on/off",
    )

    parser.add_argument(
        "--product-gate",
        type=str_to_bool,
        default=False,
        help="on/off",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--num-products",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--webshop-file-path",
        default=None,
        help="Custom WebShop product JSON file.",
    )

    parser.add_argument(
        "--webshop-attr-path",
        default=None,
        help="Custom WebShop attribute JSON file.",
    )

    parser.add_argument(
        "--webshop-human-attr-path",
        default=None,
        help="Optional custom human-goal attribute file.",
    )

    parser.add_argument(
        "--webshop-index-path",
        default=None,
        help="Custom Lucene search index directory.",
    )

    parser.add_argument(
        "--dataset",
        default=DATASET,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Optional Candidate online-weight learning.
    #
    # When enabled, install the learned type/requirement
    # weights into CandidateGate scoring before any task runs.
    # The Candidate selection policy itself remains:
    # repeat-only + hard-contradiction veto + no threshold.
    # --------------------------------------------------------

    import os

    if (
        args.candidate_gate
        and os.environ.get(
            "CANDIDATE_ONLINE_WEIGHTS",
            "0",
        ) == "1"
    ):
        from gates.candidate_online_weights import (
            reset as reset_candidate_weights,
            install as install_candidate_weights,
            get_state as get_candidate_weight_state,
        )

        reset_candidate_weights()
        install_candidate_weights()

        print(
            "Candidate online weights installed:",
            get_candidate_weight_state(),
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_dir / "results.jsonl"
    )

    if results_path.exists():

        if not args.overwrite:
            raise RuntimeError(
                f"{results_path} already exists. "
                "Use --overwrite if you really want "
                "to replace the previous run."
            )

        results_path.unlink()

    # 删除旧 summary，避免误读
    for filename in [
        "summary.json",
        "summary.txt",
    ]:
        p = output_dir / filename

        if p.exists():
            p.unlink()

    snapshot_code(output_dir)

    tasks = load_tasks(args.dataset)

    title = make_title(
        args.query_gate,
        args.product_gate,
        args.candidate_gate,
    )

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print("dataset:", args.dataset)
    print("instruction tasks:", len(tasks))
    print("max_steps:", args.max_steps)
    print("Query Gate:", "ON" if args.query_gate else "OFF")
    print(
        "Candidate Gate:",
        "ON" if args.candidate_gate else "OFF",
    )
    print(
        "Product Gate:",
        "ON" if args.product_gate else "OFF",
    )
    print("output:", output_dir)
    print("=" * 80)

    records = []

    for index, task in enumerate(
        tasks,
        start=1,
    ):
        task_id = task["task_id"]
        instruction = task["instruction"]

        print()
        print()
        print("#" * 80)
        print(
            f"TASK {index} / {len(tasks)}"
        )
        print("ASIN:", task_id)
        print("Instruction:", instruction)
        print("#" * 80)

        # ----------------------------------------------------
        # 非常重要：
        #
        # Agent 运行时只拿 task_id + instruction。
        # instruction_attributes 等 hidden evaluation 信息
        # 不传给 Agent / Query Gate / Product Gate。
        # ----------------------------------------------------
        agent_task = {
            "task_id": task_id,
            "instruction": instruction,
        }

        try:
            agent = GatedAgent(
                num_products=args.num_products,
                max_steps=args.max_steps,
                use_query_gate=args.query_gate,
                use_candidate_gate=args.candidate_gate,
                use_product_gate=args.product_gate,
                webshop_file_path=args.webshop_file_path,
                webshop_attr_path=args.webshop_attr_path,
                webshop_human_attr_path=(
                    args.webshop_human_attr_path
                ),
                webshop_index_path=args.webshop_index_path,
            )

            result = agent.run(
                agent_task
            )

            trajectory = (
                result.get("trajectory")
                or []
            )

            reward = float(
                result.get(
                    "reward",
                    0.0,
                )
            )

            if reward == 1.0:
                outcome = "FULL"

            elif reward > 0.0:
                outcome = "PARTIAL"

            else:
                outcome = "ZERO"

            target_visited = (
                target_was_visited(
                    task_id,
                    trajectory,
                )
            )

            record = {
                "task_index": index,
                "task_id": task_id,
                "instruction": instruction,

                # offline evaluation metadata only
                "instruction_attributes":
                    task.get(
                        "instruction_attributes",
                        [],
                    ),

                "reward": reward,

                "success":
                    bool(
                        result.get(
                            "success",
                            reward > 0,
                        )
                    ),

                "full_success":
                    reward == 1.0,

                "partial_success":
                    0.0 < reward < 1.0,

                "zero":
                    reward == 0.0,

                "outcome":
                    outcome,

                "steps":
                    len(trajectory),

                "target_visited":
                    target_visited,

                "buy_now_count":
                    count_action(
                        trajectory,
                        "click[buy now]",
                    ),

                "back_to_search_count":
                    count_action(
                        trajectory,
                        "click[back to search]",
                    ),

                "gate_stats":
                    result.get(
                        "gate_stats",
                        {},
                    ),

                "constraint_schema":
                    result.get(
                        "constraint_schema",
                    ),

                "trajectory":
                    trajectory,
            }

        except Exception as exc:
            print()
            print("=" * 80)
            print("TASK ERROR")
            print("=" * 80)
            traceback.print_exc()

            record = {
                "task_index": index,
                "task_id": task_id,
                "instruction": instruction,
                "instruction_attributes":
                    task.get(
                        "instruction_attributes",
                        [],
                    ),
                "reward": 0.0,
                "success": False,
                "full_success": False,
                "partial_success": False,
                "zero": False,
                "outcome": "ERROR",
                "steps": 0,
                "target_visited": False,
                "buy_now_count": 0,
                "back_to_search_count": 0,
                "gate_stats": {},
                "constraint_schema": None,
                "trajectory": [],
                "error": repr(exc),
                "traceback":
                    traceback.format_exc(),
            }

        records.append(record)


        # ONLINE CANDIDATE WEIGHT UPDATE
        if (
            __import__("os").environ.get(
                "CANDIDATE_ONLINE_WEIGHTS",
                "0",
            )
            == "1"
        ):
            from gates.candidate_online_weights import (
                update_from_result,
            )
            update_from_result(
                record
            )

        # 每个 task 立即写盘
        with open(
            results_path,
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        print()
        print("=" * 80)
        print(
            f"RESULT {index} / {len(tasks)}"
        )
        print("=" * 80)
        print("ASIN:", task_id)
        print(
            "OUTCOME:",
            record["outcome"],
        )
        print(
            "REWARD:",
            record["reward"],
        )
        print(
            "STEPS:",
            record["steps"],
        )
        print(
            "TARGET VISITED:",
            record["target_visited"],
        )
        print(
            "BUY NOW COUNT:",
            record["buy_now_count"],
        )
        print(
            "BACK TO SEARCH:",
            record["back_to_search_count"],
        )

        stats = (
            record.get("gate_stats")
            or {}
        )

        if args.candidate_gate:

            print(
                "CANDIDATE GATE CALLS:",
                stats.get(
                    "candidate_gate_calls",
                    0,
                ),
            )

            print(
                "CANDIDATE GATE CHANGES:",
                stats.get(
                    "candidate_gate_changes",
                    0,
                ),
            )

        if args.query_gate:
            print(
                "QUERY GATE CALLS:",
                stats.get(
                    "query_gate_calls",
                    0,
                ),
            )
            print(
                "QUERY GATE PASSES:",
                stats.get(
                    "query_gate_passes",
                    0,
                ),
            )
            print(
                "QUERY GATE REVISIONS:",
                stats.get(
                    "query_gate_revisions",
                    0,
                ),
            )

        if args.product_gate:
            print(
                "PRODUCT GATE READY:",
                stats.get(
                    "product_gate_ready",
                    0,
                ),
            )
            print(
                "PRODUCT GATE INSPECT:",
                stats.get(
                    "product_gate_inspects",
                    0,
                ),
            )
            print(
                "PRODUCT GATE REJECT:",
                stats.get(
                    "product_gate_rejections",
                    0,
                ),
            )

        print("=" * 80)

        current_summary = (
            compute_summary(records)
        )

        print()
        print("-" * 80)
        print(
            f"CURRENT PROGRESS: "
            f"{index} / {len(tasks)}"
        )

        print(
            f"FULL={current_summary['full_success']}  "
            f"PARTIAL={current_summary['partial_success']}  "
            f"ZERO={current_summary['zero']}  "
            f"ERROR={current_summary['errors']}  "
            f"AVG_REWARD="
            f"{current_summary['average_reward']:.4f}"
        )

        print(
            f"TARGET_VISITED="
            f"{current_summary['target_visited']}  "
            f"VISITED_BUT_ZERO="
            f"{current_summary['target_visited_but_zero']}  "
            f"UNSEEN_ZERO="
            f"{current_summary['target_not_visited_and_zero']}"
        )

        print("-" * 80)

    summary = compute_summary(
        records
    )

    print_summary(
        summary,
        title,
    )

    print()
    print("=" * 80)
    print("PER-TASK RESULTS")
    print("=" * 80)

    for record in records:
        print(
            f"{record['task_index']:02d}/"
            f"{len(records)} "
            f"{record['task_id']} "
            f"{record['outcome']} "
            f"reward={record['reward']} "
            f"steps={record['steps']} "
            f"target={record['target_visited']} "
            f"buy={record['buy_now_count']} "
            f"back={record['back_to_search_count']}"
        )

    print()
    print("=" * 80)
    print("TARGET VISITED BUT REWARD=0")
    print("=" * 80)

    for record in records:

        if (
            record["outcome"] == "ZERO"
            and record["target_visited"]
        ):
            print(
                record["task_id"],
                "steps=",
                record["steps"],
                "back_to_search=",
                record["back_to_search_count"],
            )

    print()
    print("=" * 80)
    print("TARGET NEVER VISITED + REWARD=0")
    print("=" * 80)

    for record in records:

        if (
            record["outcome"] == "ZERO"
            and not record["target_visited"]
        ):
            print(
                record["task_id"],
                "steps=",
                record["steps"],
            )

    save_summary(
        output_dir,
        summary,
        title,
    )

    print()
    print("=" * 80)
    print("ALL TASKS FINISHED")
    print("=" * 80)
    print(
        "Detailed results:",
        results_path,
    )
    print(
        "Summary JSON:",
        output_dir / "summary.json",
    )
    print(
        "Summary TXT:",
        output_dir / "summary.txt",
    )


if __name__ == "__main__":
    main()
