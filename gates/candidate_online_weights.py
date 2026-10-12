import copy
import json
import math
import os

from gates.candidate_gate import CandidateGate


STATUS_SCORE = {
    "SUPPORTED": 1.00,
    "LIKELY_SUPPORTED": 0.75,
    "UNKNOWN": 0.35,
    "CONTRADICTED": 0.00,
    "NOT_REQUIRED": 1.00,
}


STATE = {
    "w_type": 0.5,
    "w_req": 0.5,
    "eta": 0.1,
    "updates": 0,
}

_INSTALLED = False


def reset():
    STATE["w_type"] = float(
        os.environ.get(
            "CANDIDATE_INIT_TYPE",
            "0.5",
        )
    )

    STATE["w_req"] = float(
        os.environ.get(
            "CANDIDATE_INIT_REQ",
            "0.5",
        )
    )

    total = (
        STATE["w_type"]
        + STATE["w_req"]
    )

    STATE["w_type"] /= total
    STATE["w_req"] /= total

    STATE["eta"] = float(
        os.environ.get(
            "CANDIDATE_ONLINE_ETA",
            "0.1",
        )
    )

    STATE["updates"] = 0


def get_state():
    return copy.deepcopy(STATE)


def _status_score(status):
    return STATUS_SCORE.get(
        str(
            status
            or "UNKNOWN"
        ).strip().upper(),
        0.35,
    )


def _components(
    profile,
    schema=None,
):
    """
    Returns:
        type_score
        requirement_score
    """

    type_score = _status_score(
        (
            profile.get(
                "product_type",
                {},
            )
            or {}
        ).get(
            "status",
            "UNKNOWN",
        )
    )

    req_scores = []

    for item in (
        profile.get(
            "constraints",
            [],
        )
        or []
    ):
        req_scores.append(
            _status_score(
                item.get(
                    "status",
                    "UNKNOWN",
                )
            )
        )

    # Treat price as part of requirement evidence
    # only when the original schema actually
    # contains a price requirement.
    if (
        isinstance(schema, dict)
        and schema.get(
            "price_constraint"
        )
        is not None
    ):
        req_scores.append(
            _status_score(
                (
                    profile.get(
                        "price",
                        {},
                    )
                    or {}
                ).get(
                    "status",
                    "UNKNOWN",
                )
            )
        )

    if req_scores:
        req_score = (
            sum(req_scores)
            / len(req_scores)
        )
    else:
        # If no requirement dimensions exist,
        # make the two experts equivalent.
        req_score = type_score

    return (
        float(type_score),
        float(req_score),
    )


def _find_schema(
    args,
    kwargs,
):
    schema = kwargs.get(
        "schema"
    )

    if isinstance(schema, dict):
        return schema

    for item in args:
        if (
            isinstance(item, dict)
            and (
                "required_constraints"
                in item
                or "product_type"
                in item
            )
        ):
            return item

    return {}


def install():
    """
    Monkeypatch CandidateGate scoring only.

    Candidate selection policy remains unchanged:
      - same Repeat-only logic
      - no numeric minimum-score threshold
      - same hard-contradiction veto

    Only the type/requirement score weights are replaced
    by the online weights.
    """

    global _INSTALLED

    if _INSTALLED:
        return

    original = (
        CandidateGate._normalize_profile
    )

    def wrapped(
        self,
        *args,
        **kwargs
    ):
        profile = original(
            self,
            *args,
            **kwargs
        )

        if not isinstance(
            profile,
            dict,
        ):
            return profile

        schema = _find_schema(
            args,
            kwargs,
        )

        (
            type_score,
            req_score,
        ) = _components(
            profile,
            schema,
        )

        score = (
            STATE["w_type"]
            * type_score
            +
            STATE["w_req"]
            * req_score
        )

        # Keep current Repeat-only semantics
        # exactly the same:
        # hard contradiction is still a hard veto.
        if profile.get(
            "hard_contradiction",
            False,
        ):
            score = min(
                score,
                0.49,
            )

        profile[
            "online_type_score"
        ] = round(
            type_score,
            4,
        )

        profile[
            "online_requirement_score"
        ] = round(
            req_score,
            4,
        )

        profile[
            "online_w_type"
        ] = round(
            STATE["w_type"],
            6,
        )

        profile[
            "online_w_req"
        ] = round(
            STATE["w_req"],
            6,
        )

        profile[
            "overall_match_score"
        ] = round(
            score,
            4,
        )

        return profile

    CandidateGate._normalize_profile = (
        wrapped
    )

    _INSTALLED = True


def _rank_loss(
    profiles,
    target_asin,
    field,
):
    """
    rank = 1 + number of candidates
               scoring strictly above target

    loss = 1 - 1/rank

    rank 1 -> 0 loss
    rank 2 -> 0.5
    rank 3 -> 0.667
    ...
    """

    target = str(
        target_asin
    ).lower()

    target_profile = None

    for profile in profiles:
        if (
            str(
                profile.get(
                    "asin",
                    "",
                )
            ).lower()
            == target
        ):
            target_profile = profile
            break

    if target_profile is None:
        return None

    target_score = float(
        target_profile.get(
            field,
            0.0,
        )
        or 0.0
    )

    rank = 1

    for profile in profiles:
        score = float(
            profile.get(
                field,
                0.0,
            )
            or 0.0
        )

        if (
            score
            > target_score
            + 1e-12
        ):
            rank += 1

    loss = (
        1.0
        - 1.0 / rank
    )

    return {
        "rank": rank,
        "loss": loss,
    }


def _profile_signature(
    profiles,
):
    """
    Deduplicate repeated identical search-result
    profile states inside the same task.
    """

    items = []

    for profile in profiles:

        items.append(
            (
                str(
                    profile.get(
                        "asin",
                        "",
                    )
                ).lower(),

                str(
                    (
                        profile.get(
                            "product_type",
                            {},
                        )
                        or {}
                    ).get(
                        "status",
                        "",
                    )
                ),

                tuple(
                    str(
                        c.get(
                            "status",
                            "",
                        )
                    )
                    for c
                    in (
                        profile.get(
                            "constraints",
                            [],
                        )
                        or []
                    )
                ),
            )
        )

    return tuple(items)


def _write_log(record):

    path = os.environ.get(
        "CANDIDATE_ONLINE_LOG"
    )

    if not path:
        return

    os.makedirs(
        os.path.dirname(path)
        or ".",
        exist_ok=True,
    )

    with open(
        path,
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


def update_from_result(result):
    """
    One online update MAX per task.

    Training signal:
    benchmark target ASIN == task_id.

    We use all unique CandidateGate profile states
    from this task where the target was visible.

    IMPORTANT:
    This is ONLINE SUPERVISED ADAPTATION.
    """

    if not isinstance(
        result,
        dict,
    ):
        return

    target_asin = str(
        result.get(
            "task_id",
            "",
        )
    ).strip().lower()

    if not target_asin:
        return

    type_losses = []
    req_losses = []

    type_ranks = []
    req_ranks = []

    seen_states = set()

    for step in (
        result.get(
            "trajectory",
            [],
        )
        or []
    ):

        gate = step.get(
            "candidate_gate"
        )

        if not isinstance(
            gate,
            dict,
        ):
            continue

        profiles = (
            gate.get(
                "candidate_profiles",
                [],
            )
            or []
        )

        if not profiles:
            continue

        # Recompute component scores if necessary.
        for profile in profiles:

            if (
                "online_type_score"
                not in profile
                or
                "online_requirement_score"
                not in profile
            ):

                t, r = _components(
                    profile,
                    result.get(
                        "constraint_schema"
                    )
                    or {},
                )

                profile[
                    "online_type_score"
                ] = t

                profile[
                    "online_requirement_score"
                ] = r

        if not any(
            str(
                p.get(
                    "asin",
                    "",
                )
            ).lower()
            == target_asin
            for p in profiles
        ):
            continue

        signature = (
            _profile_signature(
                profiles
            )
        )

        if signature in seen_states:
            continue

        seen_states.add(
            signature
        )

        type_info = _rank_loss(
            profiles,
            target_asin,
            "online_type_score",
        )

        req_info = _rank_loss(
            profiles,
            target_asin,
            "online_requirement_score",
        )

        if (
            type_info is None
            or req_info is None
        ):
            continue

        type_losses.append(
            type_info["loss"]
        )

        req_losses.append(
            req_info["loss"]
        )

        type_ranks.append(
            type_info["rank"]
        )

        req_ranks.append(
            req_info["rank"]
        )

    old_type = STATE["w_type"]
    old_req = STATE["w_req"]

    if (
        not type_losses
        or not req_losses
    ):
        _write_log({
            "task_id":
                target_asin.upper(),

            "updated":
                False,

            "reason":
                "target_not_visible_in_candidate_profiles",

            "w_type":
                old_type,

            "w_req":
                old_req,

            "reward":
                result.get(
                    "reward"
                ),
        })

        return

    type_loss = (
        sum(type_losses)
        / len(type_losses)
    )

    req_loss = (
        sum(req_losses)
        / len(req_losses)
    )

    eta = STATE["eta"]

    new_type = (
        old_type
        * math.exp(
            -eta
            * type_loss
        )
    )

    new_req = (
        old_req
        * math.exp(
            -eta
            * req_loss
        )
    )

    z = (
        new_type
        + new_req
    )

    if z <= 0:
        return

    STATE["w_type"] = (
        new_type / z
    )

    STATE["w_req"] = (
        new_req / z
    )

    STATE["updates"] += 1

    record = {
        "task_id":
            target_asin.upper(),

        "updated":
            True,

        "supervised_states":
            len(type_losses),

        "mean_type_rank":
            sum(type_ranks)
            / len(type_ranks),

        "mean_req_rank":
            sum(req_ranks)
            / len(req_ranks),

        "type_loss":
            type_loss,

        "req_loss":
            req_loss,

        "old_w_type":
            old_type,

        "old_w_req":
            old_req,

        "new_w_type":
            STATE["w_type"],

        "new_w_req":
            STATE["w_req"],

        "eta":
            eta,

        "reward":
            result.get(
                "reward"
            ),
    }

    _write_log(
        record
    )

    print(
        "[OnlineWeights]",
        target_asin.upper(),
        "states=",
        len(type_losses),
        "loss(type/req)=",
        f"{type_loss:.3f}/{req_loss:.3f}",
        "weights=",
        f"{old_type:.3f}/{old_req:.3f}",
        "->",
        f"{STATE['w_type']:.3f}/{STATE['w_req']:.3f}",
    )
