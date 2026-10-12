import re

from gates.constraint_manager import ConstraintManager


class QueryGate:

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

    # Compatibility with existing gated_agent.py
    @property
    def extraction_calls(self):
        return (
            self.constraint_manager
            .extraction_calls
        )

    @property
    def extraction_cache_hits(self):
        return (
            self.constraint_manager
            .schema_cache_hits
        )

    @staticmethod
    def _clean_query(query):

        if query is None:
            return ""

        query = str(query).strip()

        match = re.fullmatch(
            r"search\[(.*?)\]",
            query,
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        if match:
            query = match.group(1)

        return re.sub(
            r"\s+",
            " ",
            query,
        ).strip()

    def get_constraints(
        self,
        instruction,
    ):

        return (
            self.constraint_manager
            .get_constraints(
                instruction
            )
        )

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

    def validate_query(
        self,
        instruction,
        proposed_query,
    ):

        proposed_query = (
            self._clean_query(
                proposed_query
            )
        )

        schema = self.get_constraints(
            instruction
        )

        if schema.get(
            "extraction_error"
        ):

            return {
                "decision": "PASS",
                "reason": (
                    "Constraint extraction is unavailable; "
                    "falling back to the baseline query."
                ),
                "original_query": proposed_query,
                "revised_query": None,
                "product_type": schema.get(
                    "product_type"
                ),
                "required_constraints": schema.get(
                    "required_constraints",
                    [],
                ),
                "post_selection_constraints": schema.get(
                    "post_selection_constraints",
                    [],
                ),
                "price_constraint": schema.get(
                    "price_constraint"
                ),
                "product_type_covered": None,
                "constraint_coverage": [],
                "missing_constraints": [],
                "contradicted_constraints": [],
                "gate_error": schema.get(
                    "extraction_error"
                ),
                "validator": (
                    "constraint_manager_cascade"
                ),
            }

        product_type = schema.get(
            "product_type",
            {},
        )

        required = schema.get(
            "required_constraints",
            [],
        )

        product_result = (
            self.constraint_manager
            .match_constraint(
                text=proposed_query,
                constraint=product_type,
                use_semantic=True,
                context="search query product type",
            )
        )

        coverage = []

        missing = []
        contradicted = []

        for constraint in required:

            result = (
                self.constraint_manager
                .match_constraint(
                    text=proposed_query,
                    constraint=constraint,
                    use_semantic=True,
                    context="search query constraint",
                )
            )

            canonical = (
                self._preferred_text(
                    constraint
                )
            )

            coverage.append({
                "constraint": canonical,
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

            if result["status"] == "MISSING":
                missing.append(canonical)

            elif (
                result["status"]
                == "CONTRADICTED"
            ):
                contradicted.append(canonical)

        product_missing = (
            product_result["status"]
            == "MISSING"
        )

        product_contradicted = (
            product_result["status"]
            == "CONTRADICTED"
        )

        if (
            not product_missing
            and not product_contradicted
            and not missing
            and not contradicted
        ):

            return {
                "decision": "PASS",
                "reason": (
                    "Product type and all explicit "
                    "retrieval constraints are covered."
                ),
                "original_query": proposed_query,
                "revised_query": None,
                "product_type": product_type,
                "required_constraints": required,
                "post_selection_constraints": schema.get(
                    "post_selection_constraints",
                    [],
                ),
                "price_constraint": schema.get(
                    "price_constraint"
                ),
                "product_type_covered": True,
                "product_type_match": product_result,
                "constraint_coverage": coverage,
                "missing_constraints": [],
                "contradicted_constraints": [],
                "gate_error": False,
                "validator": (
                    "constraint_manager_cascade"
                ),
            }

        # -----------------------------------------------------
        # REVISE
        #
        # Missing constraints:
        # preserve original query and append them.
        #
        # Contradiction:
        # rebuild a clean query from validated schema.
        # -----------------------------------------------------

        if (
            product_contradicted
            or contradicted
        ):

            parts = []

            product_text = (
                self._preferred_text(
                    product_type
                )
            )

            if product_text:
                parts.append(
                    product_text
                )

            for constraint in required:

                text = self._preferred_text(
                    constraint
                )

                if text:
                    parts.append(text)

        else:

            parts = [
                proposed_query
            ]

            if product_missing:

                product_text = (
                    self._preferred_text(
                        product_type
                    )
                )

                if product_text:
                    parts.append(
                        product_text
                    )

            parts.extend(
                missing
            )

        revised_query = re.sub(
            r"\s+",
            " ",
            " ".join(parts),
        ).strip()

        reasons = []

        if product_missing:
            reasons.append(
                "missing product type"
            )

        if product_contradicted:
            reasons.append(
                "product type semantic conflict"
            )

        if missing:
            reasons.append(
                "missing constraints: "
                + ", ".join(missing)
            )

        if contradicted:
            reasons.append(
                "contradicted constraints: "
                + ", ".join(
                    contradicted
                )
            )

        return {
            "decision": "REVISE",
            "reason": "; ".join(
                reasons
            ),
            "original_query": proposed_query,
            "revised_query": revised_query,
            "product_type": product_type,
            "required_constraints": required,
            "post_selection_constraints": schema.get(
                "post_selection_constraints",
                [],
            ),
            "price_constraint": schema.get(
                "price_constraint"
            ),
            "product_type_covered": (
                not product_missing
                and not product_contradicted
            ),
            "product_type_match": (
                product_result
            ),
            "constraint_coverage": coverage,
            "missing_constraints": missing,
            "contradicted_constraints": (
                contradicted
            ),
            "gate_error": False,
            "validator": (
                "constraint_manager_cascade"
            ),
        }

    def evaluate(
        self,
        instruction,
        proposed_query,
        history,
    ):

        result = self.validate_query(
            instruction=instruction,
            proposed_query=proposed_query,
        )

        previous_queries = []

        if isinstance(history, list):

            for item in history:

                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                action = str(
                    item.get(
                        "action",
                        "",
                    )
                ).strip()

                match = re.fullmatch(
                    r"search\[(.*?)\]",
                    action,
                    flags=(
                        re.IGNORECASE
                        | re.DOTALL
                    ),
                )

                if match:

                    previous_queries.append(
                        self._clean_query(
                            match.group(1)
                        )
                    )

        proposed_norm = (
            ConstraintManager.normalize_text(
                proposed_query
            )
        )

        previous_norm = {
            ConstraintManager.normalize_text(q)
            for q in previous_queries
        }

        result["repeated_query"] = (
            proposed_norm
            in previous_norm
        )

        # Repetition is diagnostic only.
        # It never changes REVISE into PASS.
        return result
