def make_result(
    task_id,
    instruction,
    success=False,
    reward=0.0,
    trajectory=None,
    constraints=None,
    query=None,
    query_gate=None,
    product_gate=None,
):
    """
    所有 Agent 最终统一返回的数据格式。
    """

    return {
        "task_id": task_id,
        "instruction": instruction,
        "success": success,
        "reward": reward,
        "trajectory": trajectory or [],
        "constraints": constraints,
        "query": query,
        "query_gate": query_gate,
        "product_gate": product_gate,
    }
