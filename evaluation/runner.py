import json
import os


def run_agent_on_tasks(agent, tasks, output_path):
    """
    用同一批 tasks 跑任意 Agent。

    已经存在的 task 会自动跳过，
    所以中途断掉以后可以继续跑。
    """

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True,
    )

    completed = set()

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    result = json.loads(line)
                    completed.add(result["task_id"])
                except Exception:
                    continue

    total = len(tasks)

    with open(output_path, "a", encoding="utf-8") as f:

        for i, task in enumerate(tasks, start=1):

            task_id = task["task_id"]

            if task_id in completed:
                print(
                    f"[{i}/{total}] "
                    f"{task_id} already completed, skip."
                )
                continue

            print(
                f"\n[{i}/{total}] Running "
                f"{task_id}"
            )

            try:
                result = agent.run(task)

                f.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                f.flush()

                print(
                    f"reward={result.get('reward')} "
                    f"success={result.get('success')}"
                )

            except Exception as e:

                error_result = {
                    "task_id": task_id,
                    "instruction": task["instruction"],
                    "success": False,
                    "reward": 0.0,
                    "error": str(e),
                }

                f.write(
                    json.dumps(
                        error_result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                f.flush()

                print(
                    f"ERROR: {e}"
                )
