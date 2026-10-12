import json
import re

import openai

from gates.constraint_manager import ConstraintManager


class ProductGate:

    FINAL_DECISIONS = {
        "ACCEPT",
        "INSPECT",
        "REJECT",
    }

    def __init__(
        self,
        model="deepseek-chat",
        constraint_manager=None,
    ):

        self.model = model

        self.constraint_manager = (
            constraint_manager
            if constraint_manager is not None
            else ConstraintManager(
                model=model
            )
        )

        # Compatibility with earlier diagnostics.
        self.constraint_extractor = (
            self.constraint_manager
        )

        self.rule_checks = 0
        self.semantic_fallback_calls = 0

        self.final_audit_calls = 0
        self.final_audit_errors = 0

        # Old names.
        self.semantic_calls = 0
        self.semantic_errors = 0

    # =========================================================
    # Observation helpers
    # =========================================================

    @classmethod
    def _strip_instruction(
        cls,
        instruction,
        observation,
    ):

        instruction_norm = (
            ConstraintManager.normalize_text(
                instruction
            )
        )

        parts = [
            x.strip()
            for x in str(
                observation or ""
            ).split("[SEP]")
            if x.strip()
        ]

        result = []

        skip_next = False

        for part in parts:

            norm = (
                ConstraintManager
                .normalize_text(
                    part
                )
            )

            if skip_next:

                if norm == instruction_norm:
                    skip_next = False
                    continue

                skip_next = False

            if norm == "instruction":
                skip_next = True
                continue

            if norm == instruction_norm:
                continue

            if norm.startswith(
                "instruction "
            ):

                remainder = norm[
                    len("instruction "):
                ].strip()

                if (
                    remainder
                    == instruction_norm
                ):
                    continue

            result.append(part)

        return " [SEP] ".join(
            result
        )

    @staticmethod
    def _clickables(
        available_actions,
    ):

        if not isinstance(
            available_actions,
            dict,
        ):
            return []

        values = available_actions.get(
            "clickables",
            [],
        )

        if not isinstance(values, list):
            return []

        return [
            str(x).strip()
            for x in values
            if str(x).strip()
        ]

    @staticmethod
    def _preferred_text(
        constraint,
    ):

        return str(
            constraint.get(
                "canonical",
                constraint.get(
                    "source_text",
                    "",
                ),
            )
        ).strip()

    # =========================================================
    # Price
    # =========================================================

    @staticmethod
    def _visible_price(text):

        match = re.search(
            r"\$\s*([0-9]+(?:\.[0-9]+)?)",
            str(text),
        )

        if not match:
            return None

        try:
            return float(
                match.group(1)
            )
        except Exception:
            return None

    @classmethod
    def _check_price(
        cls,
        price_constraint,
        evidence,
    ):

        price = cls._visible_price(
            evidence
        )

        if not price_constraint:

            return {
                "status": "NOT_REQUIRED",
                "visible_price": price,
            }

        if price is None:

            return {
                "status": "MISSING",
                "visible_price": None,
            }

        text = (
            ConstraintManager
            .normalize_text(
                price_constraint
            )
        )

        numbers = re.findall(
            r"\d+(?:\.\d+)?",
            text,
        )

        if not numbers:

            return {
                "status": "MISSING",
                "visible_price": price,
            }

        limit = float(
            numbers[-1]
        )

        upper_words = (
            "under",
            "below",
            "less than",
            "at most",
            "no more than",
        )

        lower_words = (
            "over",
            "above",
            "more than",
            "at least",
            "greater than",
        )

        if any(
            x in text
            for x in upper_words
        ):

            return {
                "status": (
                    "SUPPORTED"
                    if price <= limit
                    else "CONTRADICTED"
                ),
                "visible_price": price,
                "limit": limit,
            }

        if any(
            x in text
            for x in lower_words
        ):

            return {
                "status": (
                    "SUPPORTED"
                    if price >= limit
                    else "CONTRADICTED"
                ),
                "visible_price": price,
                "limit": limit,
            }

        return {
            "status": "MISSING",
            "visible_price": price,
            "limit": limit,
        }

    # =========================================================
    # Option handling
    # =========================================================

    @classmethod
    def _option_candidates(
        cls,
        constraint,
    ):

        values = [
            cls._preferred_text(
                constraint
            ),
            constraint.get(
                "source_text",
                "",
            ),
        ]

        values.extend(
            constraint.get(
                "aliases",
                [],
            )
        )

        result = []
        seen = set()

        for value in values:

            value = str(
                value
            ).strip()

            if not value:
                continue

            stripped = re.sub(
                r"^\s*"
                r"(color|size|style|scent|"
                r"pack|count|variant)"
                r"\s*[:=-]?\s*",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip()

            for candidate in [
                value,
                stripped,
            ]:

                norm = (
                    ConstraintManager
                    .normalize_text(
                        candidate
                    )
                )

                if (
                    norm
                    and norm not in seen
                ):

                    seen.add(norm)
                    result.append(
                        candidate
                    )

        return result

    @classmethod
    def _option_selected(
        cls,
        constraint,
        history,
    ):

        candidates = (
            cls._option_candidates(
                constraint
            )
        )

        for item in reversed(
            history[-12:]
            if isinstance(history, list)
            else []
        ):

            action = str(
                item.get(
                    "action",
                    "",
                )
            ).strip()

            match = re.fullmatch(
                r"click\[(.*?)\]",
                action,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            clicked = (
                ConstraintManager
                .normalize_text(
                    match.group(1)
                )
            )

            for candidate in candidates:

                candidate = (
                    ConstraintManager
                    .normalize_text(
                        candidate
                    )
                )

                if (
                    candidate
                    and clicked
                    and (
                        candidate == clicked
                        or candidate in clicked
                        or clicked in candidate
                    )
                ):

                    return True, match.group(1)

        return False, None

    @classmethod
    def _option_available(
        cls,
        constraint,
        available_actions,
    ):

        candidates = (
            cls._option_candidates(
                constraint
            )
        )

        for action in cls._clickables(
            available_actions
        ):

            action_norm = (
                ConstraintManager
                .normalize_text(
                    action
                )
            )

            for candidate in candidates:

                candidate_norm = (
                    ConstraintManager
                    .normalize_text(
                        candidate
                    )
                )

                if (
                    candidate_norm
                    and (
                        candidate_norm
                        == action_norm
                        or candidate_norm
                        in action_norm
                        or action_norm
                        in candidate_norm
                    )
                ):

                    return True, action

        return False, None

    # =========================================================
    # Rule first, semantic fallback only for unresolved
    # =========================================================

    def _resolve_product_type(
        self,
        evidence,
        constraint,
    ):
        """
        Product type is handled separately from ordinary
        attributes.

        A clearly different product category should become
        CONTRADICTED rather than merely MISSING.
        """

        deterministic = (
            self.constraint_manager
            .match_product_type(
                text=evidence,
                constraint=constraint,
                use_semantic=False,
                context=(
                    "visible product evidence: "
                    "requested product category"
                ),
            )
        )


        if (
            deterministic["status"]
            == "SUPPORTED"
        ):

            return deterministic


        self.semantic_fallback_calls += 1
        self.semantic_calls += 1


        semantic = (
            self.constraint_manager
            .semantic_match_product_type(
                text=evidence,
                constraint=constraint,
                context=(
                    "visible product evidence: "
                    "requested product category"
                ),
            )
        )


        if semantic.get("error"):

            self.semantic_errors += 1


        return semantic


    def _resolve_constraint(
        self,
        evidence,
        constraint,
        context,
    ):

        deterministic = (
            self.constraint_manager
            .match_constraint(
                text=evidence,
                constraint=constraint,
                use_semantic=False,
                context=context,
            )
        )

        if (
            deterministic["status"]
            == "SUPPORTED"
        ):
            return deterministic

        self.semantic_fallback_calls += 1
        self.semantic_calls += 1

        semantic = (
            self.constraint_manager
            .semantic_match(
                text=evidence,
                constraint=constraint,
                context=context,
            )
        )

        if semantic.get("error"):
            self.semantic_errors += 1

        return semantic

    # =========================================================
    # Final audit
    # Only called when every requirement is resolved SUPPORTED
    # =========================================================

    def _final_audit(
        self,
        instruction,
        schema,
        evidence,
    ):

        prompt = f"""
You are the final semantic auditor for a Product Gate.

All lower-level constraint checks have already found support.

Your only task is to make sure those matches are genuinely
semantically correct and preserve user intent.

User instruction:
{instruction}

Validated constraint schema:
{json.dumps(schema, ensure_ascii=False)}

Visible product evidence:
{evidence}

Use ONLY the visible evidence.
Do not use hidden ASINs, attributes, reward, or outside knowledge.

Preserve semantic polarity.

Examples:

animal testing != cruelty free
with alcohol != alcohol free
contains fragrance != fragrance free
cruelty free may equal not tested on animals

Return ONLY JSON:

{{
  "decision": "ACCEPT or REJECT",
  "problematic_constraints": [],
  "reason": "short explanation"
}}

This is a FINAL audit: all lower-level checks already found support.
Do NOT return INSPECT. Do not demand exact wording when the visible
evidence clearly expresses the same product category or requirement.
Return REJECT only when there is a concrete semantic contradiction or
a genuine mismatch with the user's intent. Otherwise return ACCEPT.
"""

        try:

            self.final_audit_calls += 1
            self.semantic_calls += 1

            response = (
                openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    temperature=0,
                )
            )

            raw = (
                response["choices"][0]["message"]
                ["content"]
                .strip()
            )

            parsed = (
                ConstraintManager
                ._extract_json(
                    raw
                )
            )

            if not isinstance(
                parsed,
                dict,
            ):
                raise ValueError(
                    "Invalid final audit JSON"
                )

            decision = str(
                parsed.get(
                    "decision",
                    "INSPECT",
                )
            ).strip().upper()

            if decision not in self.FINAL_DECISIONS:
                decision = "REJECT"

            problematic = parsed.get(
                "problematic_constraints",
                [],
            )

            if not isinstance(
                problematic,
                list,
            ):
                problematic = []

            # FINAL means terminal.
            # Older/local models may still emit INSPECT despite the
            # prompt. Never let that create an infinite inspection loop.
            #
            # All lower-level checks already support the candidate:
            # - no concrete problem -> ACCEPT
            # - concrete problem    -> REJECT
            if decision == "INSPECT":
                decision = (
                    "REJECT"
                    if problematic
                    else "ACCEPT"
                )

            return {
                "decision": decision,
                "problematic_constraints": [
                    str(x).strip()
                    for x in problematic
                    if str(x).strip()
                ],
                "reason": str(
                    parsed.get(
                        "reason",
                        "",
                    )
                ).strip(),
                "raw_output": raw,
                "error": None,
            }

        except Exception as e:

            self.final_audit_errors += 1
            self.semantic_errors += 1

            # Never loop forever because the final audit itself failed.
            # Conservative terminal fallback: reject this candidate.
            return {
                "decision": "REJECT",
                "problematic_constraints": [
                    "final_audit_error"
                ],
                "reason": (
                    "Final audit failed; reject this candidate "
                    "instead of repeating inspection."
                ),
                "raw_output": None,
                "error": str(e),
            }

    # =========================================================
    # Recommendation only
    # =========================================================

    def _recommend_action(
        self,
        decision,
        option_states,
        available_actions,
        history,
    ):

        clickables = self._clickables(
            available_actions
        )

        lower = {
            x.lower(): x
            for x in clickables
        }

        if decision in {
            "REJECT",
            "EXHAUSTED",
        }:

            if "back to search" in lower:
                return (
                    "click["
                    + lower["back to search"]
                    + "]"
                )

            if "< prev" in lower:
                return (
                    "click["
                    + lower["< prev"]
                    + "]"
                )

            return None

        for option in option_states:

            if (
                not option["selected"]
                and option["available"]
            ):

                return (
                    "click["
                    + option[
                        "available_action"
                    ]
                    + "]"
                )

        if decision == "READY":

            if "buy now" in lower:
                return (
                    "click["
                    + lower["buy now"]
                    + "]"
                )

            return None

        recent = {
            str(
                x.get(
                    "action",
                    "",
                )
            ).lower()
            for x in (
                history[-8:]
                if isinstance(history, list)
                else []
            )
        }

        for name in [
            "features",
            "description",
            "reviews",
        ]:

            if (
                name in lower
                and (
                    f"click[{name}]"
                    not in recent
                )
            ):

                return (
                    "click["
                    + lower[name]
                    + "]"
                )

        for name in [
            "features",
            "description",
            "reviews",
        ]:

            if name in lower:
                return (
                    "click["
                    + lower[name]
                    + "]"
                )

        return None

    # =========================================================
    # Main evaluate
    # =========================================================

    def evaluate(
        self,
        instruction,
        observation,
        history=None,
        available_actions=None,
        inspection_state=None,
    ):

        self.rule_checks += 1

        history = (
            history
            if isinstance(history, list)
            else []
        )

        available_actions = (
            available_actions
            if isinstance(
                available_actions,
                dict,
            )
            else {}
        )

        schema = (
            self.constraint_manager
            .get_constraints(
                instruction
            )
        )

        evidence = self._strip_instruction(
            instruction,
            observation,
        )

        if schema.get(
            "extraction_error"
        ):

            return {
                "decision": "PASS_THROUGH",

                "ready_to_buy": False,

                # No reliable constraint schema means Product
                # Gate has no basis for forcing the candidate
                # to stay or leave.
                "should_keep_candidate": False,

                "rule_passed": False,

                "reason": (
                    "Constraint extraction did not pass "
                    "validation; Product Gate fails open "
                    "and does not intervene."
                ),

                "matched_constraints": [],

                "missing_constraints": [],

                "contradicted_constraints": [],

                "recommended_action": None,

                "llm_audit_used": False,

                "semantic_fallback_used": False,

                "constraint_evidence": [],

                "inspection_state": (
                    inspection_state
                ),

                "gate_error": schema.get(
                    "extraction_error"
                ),
            }


        semantic_before = (
            self.semantic_fallback_calls
        )

        # -----------------------------------------------------
        # Product type
        # -----------------------------------------------------

        product_type = schema.get(
            "product_type",
            {},
        )

        product_result = (
            self._resolve_product_type(
                evidence=evidence,
                constraint=product_type,
            )
        )


        # =====================================================
        # PRODUCT TYPE EARLY STOP
        # =====================================================
        #
        # Different product category:
        #     reject immediately.
        #
        # Unclear category:
        #     inspect immediately.
        #
        # Only if product type is SUPPORTED do we spend calls
        # checking detailed attributes.
        # =====================================================

        if (
            product_result["status"]
            == "CONTRADICTED"
        ):

            decision = "REJECT"

            return {
                "decision": decision,
                "ready_to_buy": False,
                "should_keep_candidate": False,
                "rule_passed": False,

                "reason": (
                    "Visible product evidence identifies "
                    "a different product category from "
                    "the one requested by the user. "
                    + product_result.get(
                        "reason",
                        ""
                    )
                ).strip(),

                "matched_constraints": [],

                "missing_constraints": [],

                "contradicted_constraints": [
                    "product_type"
                ],

                "constraint_evidence": [],

                "product_type_evidence": (
                    product_result
                ),

                "post_selection": [],

                "price_evidence": {},

                "recommended_action": (
                    self._recommend_action(
                        decision,
                        [],
                        available_actions,
                        history,
                    )
                ),

                "semantic_fallback_used": True,

                "llm_audit_used": False,

                "llm_audit": None,

                "visible_product_evidence": (
                    evidence
                ),

                "gate_error": (
                    product_result.get(
                        "error"
                    )
                ),
            }


        if (
            product_result["status"]
            == "MISSING"
        ):

            decision = "INSPECT"

            return {
                "decision": decision,
                "ready_to_buy": False,
                "should_keep_candidate": True,
                "rule_passed": False,

                "reason": (
                    "The visible evidence is not yet "
                    "specific enough to verify the "
                    "requested product category. "
                    + product_result.get(
                        "reason",
                        ""
                    )
                ).strip(),

                "matched_constraints": [],

                "missing_constraints": [
                    "product_type"
                ],

                "contradicted_constraints": [],

                "constraint_evidence": [],

                "product_type_evidence": (
                    product_result
                ),

                "post_selection": [],

                "price_evidence": {},

                "recommended_action": (
                    self._recommend_action(
                        decision,
                        [],
                        available_actions,
                        history,
                    )
                ),

                "semantic_fallback_used": True,

                "llm_audit_used": False,

                "llm_audit": None,

                "visible_product_evidence": (
                    evidence
                ),

                "gate_error": (
                    product_result.get(
                        "error"
                    )
                ),
            }


        # -----------------------------------------------------
        # Product type is supported.
        # Now evaluate detailed required constraints.
        # -----------------------------------------------------

        constraint_evidence = []

        matched = []
        missing = []
        contradicted = []

        for constraint in schema.get(
            "required_constraints",
            [],
        ):

            result = (
                self._resolve_constraint(
                    evidence=evidence,
                    constraint=constraint,
                    context=(
                        "visible product evidence: "
                        "required shopping constraint"
                    ),
                )
            )

            name = self._preferred_text(
                constraint
            )

            constraint_evidence.append({
                "constraint": name,
                "source_text": constraint.get(
                    "source_text"
                ),
                "status": result[
                    "status"
                ],
                "match_type": result[
                    "match_type"
                ],
                "matched_by": result[
                    "matched_by"
                ],
                "reason": result[
                    "reason"
                ],
            })

            if result["status"] == "SUPPORTED":
                matched.append(name)

            elif result["status"] == "CONTRADICTED":
                contradicted.append(name)

            else:
                missing.append(name)

        # -----------------------------------------------------
        # Options
        # -----------------------------------------------------

        option_states = []
        unselected_options = []

        for constraint in schema.get(
            "post_selection_constraints",
            [],
        ):

            selected, selected_action = (
                self._option_selected(
                    constraint,
                    history,
                )
            )

            available, available_action = (
                self._option_available(
                    constraint,
                    available_actions,
                )
            )

            name = self._preferred_text(
                constraint
            )

            state = {
                "constraint": name,
                "selected": selected,
                "selected_action": selected_action,
                "available": available,
                "available_action": available_action,
            }

            # -------------------------------------------------
            # REAL OPTION
            #
            # A post-selection constraint is a true selectable
            # option only when:
            #   1. it was actually selected before, or
            #   2. a matching clickable option exists now.
            # -------------------------------------------------

            if selected or available:

                state["mode"] = "option"

                option_states.append(
                    state
                )

                if not selected:

                    unselected_options.append(
                        name
                    )

                continue

            # -------------------------------------------------
            # ATTRIBUTE FALLBACK
            #
            # WebShop contains many phrases that the extractor
            # may classify as post-selection constraints even
            # though no selectable variant exists:
            #
            #   for sensitive skin
            #   for dry hair
            #   for dead skin
            #
            # If there is no corresponding clickable option,
            # validate the phrase against accumulated visible
            # product evidence instead of waiting forever for
            # an impossible option click.
            # -------------------------------------------------

            result = self._resolve_constraint(
                evidence=evidence,
                constraint=constraint,
                context=(
                    "visible product evidence: "
                    "post-selection shopping requirement "
                    "with no matching selectable option"
                ),
            )

            state["mode"] = "evidence"
            state["evidence_status"] = (
                result["status"]
            )

            option_states.append(
                state
            )

            constraint_evidence.append({
                "constraint": name,
                "source_text": constraint.get(
                    "source_text"
                ),
                "status": result["status"],
                "match_type": result[
                    "match_type"
                ],
                "matched_by": result[
                    "matched_by"
                ],
                "reason": result[
                    "reason"
                ],
            })

            if (
                result["status"]
                == "SUPPORTED"
            ):

                matched.append(
                    name
                )

            elif (
                result["status"]
                == "CONTRADICTED"
            ):

                contradicted.append(
                    name
                )

            else:

                missing.append(
                    name
                )

        # -----------------------------------------------------
        # Price
        # -----------------------------------------------------

        price_result = self._check_price(
            schema.get(
                "price_constraint"
            ),
            evidence,
        )

        if (
            price_result["status"]
            == "CONTRADICTED"
        ):

            contradicted.append(
                "price"
            )

        elif (
            price_result["status"]
            == "MISSING"
        ):

            missing.append(
                "price"
            )

        semantic_used = (
            self.semantic_fallback_calls
            > semantic_before
        )

        # -----------------------------------------------------
        # Confirmed contradiction
        # -----------------------------------------------------

        if contradicted:

            decision = "REJECT"

            return {
                "decision": decision,
                "ready_to_buy": False,
                "should_keep_candidate": False,
                "rule_passed": False,
                "reason": (
                    "Confirmed contradiction: "
                    + ", ".join(
                        contradicted
                    )
                ),
                "matched_constraints": matched,
                "missing_constraints": missing,
                "contradicted_constraints": contradicted,
                "constraint_evidence": (
                    constraint_evidence
                ),
                "product_type_evidence": (
                    product_result
                ),
                "post_selection": option_states,
                "price_evidence": price_result,
                "recommended_action": (
                    self._recommend_action(
                        decision,
                        option_states,
                        available_actions,
                        history,
                    )
                ),
                "semantic_fallback_used": (
                    semantic_used
                ),
                "llm_audit_used": False,
                "llm_audit": None,
                "visible_product_evidence": (
                    evidence
                ),
                "gate_error": False,
            }

        # -----------------------------------------------------
        # Still unresolved
        # -----------------------------------------------------

        if (
            missing
            or unselected_options
        ):

            has_inspection_state = isinstance(
                inspection_state,
                dict,
            )

            seen_sections = set()

            available_sections = set()


            if has_inspection_state:

                seen_sections = {
                    str(x).strip().lower()
                    for x in inspection_state.get(
                        "seen_sections",
                        [],
                    )
                    if str(x).strip()
                }

                available_sections = {
                    str(x).strip().lower()
                    for x in inspection_state.get(
                        "available_sections",
                        [],
                    )
                    if str(x).strip()
                }


            # -------------------------------------------------
            # Fallback:
            #
            # GatedAgent already stores accumulated evidence
            # with markers such as:
            #
            #   after click[features]
            #   after click[description]
            #   after click[reviews]
            #
            # Therefore ProductGate can reconstruct inspection
            # progress itself even if inspection_state was not
            # explicitly passed by the agent.
            # -------------------------------------------------

            evidence_lower = str(
                evidence
            ).lower()


            standard_sections = {
                "features",
                "description",
                "reviews",
            }


            for section in standard_sections:

                marker = (
                    "after click["
                    + section
                    + "]"
                )

                if marker in evidence_lower:

                    seen_sections.add(
                        section
                    )


            # WebShop product pages expose these three standard
            # evidence surfaces. If no explicit inspection state
            # was supplied, use them as the candidate inspection
            # space.
            if not has_inspection_state:

                available_sections = set(
                    standard_sections
                )


            # -------------------------------------------------
            # Fallback:
            #
            # GatedAgent already stores accumulated evidence
            # with markers such as:
            #
            #   after click[features]
            #   after click[description]
            #   after click[reviews]
            #
            # Therefore ProductGate can reconstruct inspection
            # progress itself even if inspection_state was not
            # explicitly passed by the agent.
            # -------------------------------------------------

            evidence_lower = str(
                evidence
            ).lower()


            standard_sections = {
                "features",
                "description",
                "reviews",
            }


            for section in standard_sections:

                marker = (
                    "after click["
                    + section
                    + "]"
                )

                if marker in evidence_lower:

                    seen_sections.add(
                        section
                    )


            # WebShop product pages expose these three standard
            # evidence surfaces. If no explicit inspection state
            # was supplied, use them as the candidate inspection
            # space.
            if not has_inspection_state:

                available_sections = set(
                    standard_sections
                )


            # -------------------------------------------------
            # EXHAUSTED
            #
            # All useful evidence pages for THIS candidate have
            # already been inspected, but required evidence is
            # still missing.
            #
            # Missing != contradiction, so keep this separate
            # from REJECT.
            # -------------------------------------------------

            inspections_exhausted = (
                bool(missing)
                and not unselected_options
                and (
                    not available_sections
                    or available_sections.issubset(
                        seen_sections
                    )
                )
            )


            if inspections_exhausted:

                decision = "EXHAUSTED"

                return {
                    "decision": decision,

                    "ready_to_buy": False,

                    "should_keep_candidate": False,

                    "rule_passed": False,

                    "reason": (
                        "All available product evidence "
                        "sections have been inspected, but "
                        "required evidence is still unresolved: "
                        + ", ".join(
                            missing
                        )
                    ),

                    "matched_constraints": matched,

                    "missing_constraints": missing,

                    "contradicted_constraints": [],

                    "constraint_evidence": (
                        constraint_evidence
                    ),

                    "product_type_evidence": (
                        product_result
                    ),

                    "post_selection": (
                        option_states
                    ),

                    "price_evidence": (
                        price_result
                    ),

                    "recommended_action": (
                        self._recommend_action(
                            decision,
                            option_states,
                            available_actions,
                            history,
                        )
                    ),

                    "semantic_fallback_used": (
                        semantic_used
                    ),

                    "llm_audit_used": False,

                    "llm_audit": None,

                    "inspection_state": (
                        inspection_state
                    ),

                    "visible_product_evidence": (
                        evidence
                    ),

                    "gate_error": False,
                }


            # -------------------------------------------------
            # More useful evidence is still available.
            # -------------------------------------------------

            decision = "INSPECT"

            reasons = []


            if missing:

                reasons.append(
                    "unresolved evidence: "
                    + ", ".join(
                        missing
                    )
                )


            if unselected_options:

                reasons.append(
                    "required options not selected: "
                    + ", ".join(
                        unselected_options
                    )
                )


            # Prefer an unseen evidence page when one is
            # currently clickable.
            recommended_action = None

            clickables = {
                str(x).strip().lower(): str(x).strip()
                for x in self._clickables(
                    available_actions
                )
            }


            for section in [
                "features",
                "description",
                "reviews",
            ]:

                if (
                    section
                    in available_sections
                    and section
                    not in seen_sections
                    and section
                    in clickables
                ):

                    recommended_action = (
                        "click["
                        + clickables[
                            section
                        ]
                        + "]"
                    )

                    break


            if recommended_action is None:

                recommended_action = (
                    self._recommend_action(
                        decision,
                        option_states,
                        available_actions,
                        history,
                    )
                )


            return {
                "decision": decision,

                "ready_to_buy": False,

                "should_keep_candidate": True,

                "rule_passed": False,

                "reason": "; ".join(
                    reasons
                ),

                "matched_constraints": matched,

                "missing_constraints": missing,

                "contradicted_constraints": [],

                "constraint_evidence": (
                    constraint_evidence
                ),

                "product_type_evidence": (
                    product_result
                ),

                "post_selection": option_states,

                "price_evidence": price_result,

                "recommended_action": (
                    recommended_action
                ),

                "semantic_fallback_used": (
                    semantic_used
                ),

                "llm_audit_used": False,

                "llm_audit": None,

                "inspection_state": (
                    inspection_state
                ),

                "visible_product_evidence": (
                    evidence
                ),

                "gate_error": False,
            }


        # -----------------------------------------------------
        # Everything is supported.
        # NOW call final LLM audit.
        # -----------------------------------------------------

        audit = self._final_audit(
            instruction=instruction,
            schema=schema,
            evidence=evidence,
        )

        if audit["decision"] == "ACCEPT":

            decision = "READY"
            ready = True
            keep = True

        elif audit["decision"] == "REJECT":

            decision = "REJECT"
            ready = False
            keep = False

        else:

            decision = "INSPECT"
            ready = False
            keep = True

        return {
            "decision": decision,
            "ready_to_buy": ready,
            "should_keep_candidate": keep,
            "rule_passed": True,
            "reason": audit["reason"],
            "matched_constraints": matched,
            "missing_constraints": [],
            "contradicted_constraints": (
                audit[
                    "problematic_constraints"
                ]
                if decision == "REJECT"
                else []
            ),
            "constraint_evidence": (
                constraint_evidence
            ),
            "product_type_evidence": (
                product_result
            ),
            "post_selection": option_states,
            "price_evidence": price_result,
            "recommended_action": (
                self._recommend_action(
                    decision,
                    option_states,
                    available_actions,
                    history,
                )
            ),
            "semantic_fallback_used": (
                semantic_used
            ),
            "llm_audit_used": True,
            "llm_audit": audit,
            "visible_product_evidence": evidence,
            "gate_error": audit.get(
                "error"
            ),
        }
