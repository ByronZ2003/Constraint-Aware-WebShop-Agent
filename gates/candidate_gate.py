import json
import re
import time

import openai


class CandidateGate:
    """
    Candidate Selector V2

    Key ideas:
    1. Reuse the SAME request constraint schema.
    2. Build a schema-aligned Candidate Match Profile.
    3. Compute score deterministically from categorical judgments.
    4. On repeated product clicks, select the highest-scoring safe unseen candidate with no minimum threshold.
    5. Store selected profile on the shared ConstraintManager as WEAK prior.
    """

    STATUS_SCORE = {
        "SUPPORTED": 1.00,
        "LIKELY_SUPPORTED": 0.75,
        "UNKNOWN": 0.35,
        "CONTRADICTED": 0.00,
        "NOT_REQUIRED": 1.00,
    }

    VALID_STATUS = set(STATUS_SCORE)

    def __init__(
        self,
        model="deepseek-chat",
        constraint_manager=None,
    ):
        self.model = model
        self.constraint_manager = constraint_manager

        self.visited_candidates = set()
        self.search_queries = []

        if (
            self.constraint_manager is not None
            and not hasattr(
                self.constraint_manager,
                "_candidate_priors",
            )
        ):
            self.constraint_manager._candidate_priors = {}

    # =========================================================
    # Action helpers
    # =========================================================

    @staticmethod
    def _click(action):

        m = re.fullmatch(
            r"\s*click\[(.*?)\]\s*",
            str(action or ""),
            flags=re.I | re.S,
        )

        return (
            m.group(1).strip()
            if m
            else None
        )

    @staticmethod
    def _search(action):

        m = re.fullmatch(
            r"\s*search\[(.*?)\]\s*",
            str(action or ""),
            flags=re.I | re.S,
        )

        return (
            m.group(1).strip()
            if m
            else None
        )

    @staticmethod
    def _norm_query(query):

        return " ".join(
            str(query or "")
            .lower()
            .split()
        )

    @staticmethod
    def _is_pid(value):

        if not value:
            return False

        value = str(value).strip()

        # WebShop/Amazon ASIN:
        # exactly 10 alphanumeric characters.
        # Require at least one digit so UI labels such as
        # "attributes" are never mistaken for product IDs.
        return bool(
            re.fullmatch(
                r"[A-Za-z0-9]{10}",
                value,
            )
            and any(
                ch.isdigit()
                for ch in value
            )
        )

    @classmethod
    def _product_from_action(
        cls,
        action,
    ):

        value = cls._click(action)

        if cls._is_pid(value):
            return value

        return None

    @classmethod
    def _visible_products(
        cls,
        available_actions,
    ):

        if not isinstance(
            available_actions,
            dict,
        ):
            return []

        result = []
        seen = set()

        for x in (
            available_actions.get(
                "clickables",
                [],
            )
            or []
        ):
            x = str(x).strip()

            if not cls._is_pid(x):
                continue

            key = x.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(x)

        return result

    @staticmethod
    def _has_next(
        available_actions,
    ):

        if not isinstance(
            available_actions,
            dict,
        ):
            return False

        return any(
            str(x).strip().lower()
            == "next >"
            for x in (
                available_actions.get(
                    "clickables",
                    [],
                )
                or []
            )
        )

    # =========================================================
    # Schema
    # =========================================================

    @staticmethod
    def _constraint_name(c):

        if isinstance(c, dict):

            return str(
                c.get(
                    "canonical",
                    c.get(
                        "source_text",
                        "",
                    ),
                )
            ).strip()

        return str(c or "").strip()

    @classmethod
    def _schema_view(
        cls,
        schema,
    ):

        if not isinstance(
            schema,
            dict,
        ):
            return {}

        return {
            "product_type":
                schema.get(
                    "product_type"
                ),

            "required_constraints":
                schema.get(
                    "required_constraints",
                    [],
                ),

            "post_selection_constraints":
                schema.get(
                    "post_selection_constraints",
                    [],
                ),

            "price_constraint":
                schema.get(
                    "price_constraint"
                ),
        }

    # =========================================================
    # Search-result candidate text
    # =========================================================

    @classmethod
    def _candidate_blocks(
        cls,
        observation,
        visible,
    ):

        parts = [
            str(x).strip()
            for x in str(
                observation or ""
            ).split("[SEP]")
            if str(x).strip()
        ]

        visible_map = {
            str(x).lower(): str(x)
            for x in visible
        }

        blocks = {
            str(x).lower(): []
            for x in visible
        }

        current = None

        for part in parts:

            key = part.lower()

            if key in visible_map:

                current = key
                continue

            if (
                cls._is_pid(part)
                and key not in visible_map
            ):
                current = None
                continue

            if current is not None:

                blocks[current].append(
                    part
                )

        result = {}

        for asin in visible:

            text = " [SEP] ".join(
                blocks.get(
                    asin.lower(),
                    [],
                )
            )

            result[
                asin.lower()
            ] = text[:1800]

        return result

    # =========================================================
    # JSON parser
    # =========================================================

    @staticmethod
    def _parse_json_object(text):

        text = str(
            text or ""
        ).strip()

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.I,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        start = text.find("{")
        end = text.rfind("}")

        if (
            start < 0
            or end < start
        ):
            raise ValueError(
                "No JSON object found"
            )

        return json.loads(
            text[start:end + 1]
        )

    @classmethod
    def _clean_status(
        cls,
        value,
    ):

        value = str(
            value or "UNKNOWN"
        ).strip().upper()

        if value not in cls.VALID_STATUS:
            return "UNKNOWN"

        return value

    # =========================================================
    # Candidate profiling
    # =========================================================

    def _ask_profiles(
        self,
        instruction,
        constraint_schema,
        candidate_blocks,
    ):
        """
        Fast Candidate profiling.

        Important:
        - Semantic status judgments are unchanged.
        - Downstream profile structure is unchanged.
        - LLM only emits compact categorical codes.
        - Profiles are cached by:
          instruction + schema + ASIN + visible text.
        """

        # -----------------------------------------------------
        # Lazy cache. Key includes request + visible evidence,
        # so profiles from a different task/evidence cannot
        # accidentally be reused.
        # -----------------------------------------------------
        if not hasattr(self, "_profile_cache"):
            self._profile_cache = {}

        schema = constraint_schema or {}

        schema_key = json.dumps(
            schema,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

        required = (
            schema.get("required_constraints", [])
            or []
        )

        required_names = [
            self._constraint_name(x)
            for x in required
        ]

        product_type_obj = (
            schema.get("product_type")
            or {}
        )

        product_type_name = (
            self._constraint_name(
                product_type_obj
            )
        )

        price_constraint = (
            schema.get("price_constraint")
        )

        # -----------------------------------------------------
        # Cache lookup at individual candidate level.
        # -----------------------------------------------------
        merged = {}
        missing = {}
        cache_keys = {}

        for asin, visible_text in candidate_blocks.items():

            key = (
                str(instruction or ""),
                schema_key,
                str(asin).lower(),
                str(visible_text or ""),
            )

            cache_keys[str(asin).lower()] = key

            cached = self._profile_cache.get(key)

            if cached is not None:
                merged[str(asin).lower()] = cached
            else:
                missing[asin] = visible_text

        cache_hits = (
            len(candidate_blocks)
            - len(missing)
        )

        print(
            "[CandidateCache]",
            "hits=",
            cache_hits,
            "misses=",
            len(missing),
        )

        # -----------------------------------------------------
        # Everything already profiled -> NO LLM CALL.
        # -----------------------------------------------------
        if not missing:

            self._last_profile_llm_used = False

            payload = {
                "candidates": [
                    merged[str(asin).lower()]
                    for asin
                    in candidate_blocks
                    if str(asin).lower() in merged
                ]
            }

            print(
                "[CandidateLLM]",
                "cache_only=True",
                "time=0.00",
                "candidates=",
                len(candidate_blocks),
            )

            return (
                payload,
                "[PROFILE_CACHE]",
            )

        # -----------------------------------------------------
        # Compact request.
        #
        # Codes:
        # S = SUPPORTED
        # L = LIKELY_SUPPORTED
        # U = UNKNOWN
        # C = CONTRADICTED
        # N = NOT_REQUIRED (price only)
        #
        # No explanation/evidence generation.
        # -----------------------------------------------------
        compact_candidates = [
            [
                asin,
                str(text or ""),
            ]
            for asin, text in missing.items()
        ]

        prompt = f"""
You are a WebShop candidate matcher.

Judge only from the visible search-result text.
Do not use hidden product information.

STATUS CODES:
S = SUPPORTED
L = LIKELY_SUPPORTED
U = UNKNOWN
C = CONTRADICTED
N = NOT_REQUIRED

USER REQUEST:
{instruction}

PRODUCT TYPE:
{product_type_name}

REQUIRED CONSTRAINTS IN ORDER:
{json.dumps(required_names, ensure_ascii=False)}

PRICE CONSTRAINT:
{json.dumps(price_constraint, ensure_ascii=False)}

CANDIDATES:
{json.dumps(compact_candidates, ensure_ascii=False, separators=(",", ":"))}

RULES:
1. Use only visible candidate text.
2. Clear wrong product category = C.
3. Explicit support = S.
4. Strong but not explicit support = L.
5. Missing evidence = U, never C.
6. For REQUIRED CONSTRAINTS, return one status code per
   constraint, in exactly the same order.
7. If there is no price constraint, price code = N.
8. Do not provide explanations.
9. Do not provide evidence text.
10. Do not repeat product titles.
11. Return JSON only.

FORMAT:
{{
  "c": [
    ["ASIN", "TYPE_CODE", ["REQ1_CODE", "REQ2_CODE"], "PRICE_CODE"]
  ]
}}
"""

        # Enough room for compact codes, but nowhere near 2200.
        max_out = min(
            900,
            max(
                300,
                100
                + len(missing)
                * (
                    25
                    + 8 * len(required_names)
                ),
            ),
        )

        self._last_profile_llm_used = True

        t0 = time.perf_counter()

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.0,
            max_tokens=max_out,
        )

        dt = time.perf_counter() - t0

        usage = response.get("usage", {}) or {}

        content = (
            response["choices"][0]
            ["message"]
            ["content"]
        )

        content = str(content or "").strip()

        print(
            "[CandidateLLM]",
            "time=",
            round(dt, 2),
            "prompt_tokens=",
            usage.get("prompt_tokens"),
            "completion_tokens=",
            usage.get("completion_tokens"),
            "total_tokens=",
            usage.get("total_tokens"),
            "candidates=",
            len(missing),
            "cache_hits=",
            cache_hits,
            "max_tokens=",
            max_out,
        )

        # -----------------------------------------------------
        # Parse compact JSON.
        # -----------------------------------------------------
        json_start = content.find("{")
        json_end = content.rfind("}")

        if (
            json_start < 0
            or json_end < json_start
        ):
            raise ValueError(
                "Candidate compact JSON not found: "
                + content[:300]
            )

        compact = json.loads(
            content[
                json_start:
                json_end + 1
            ]
        )

        rows = compact.get("c", []) or []

        code_map = {
            "S": "SUPPORTED",
            "L": "LIKELY_SUPPORTED",
            "U": "UNKNOWN",
            "C": "CONTRADICTED",
            "N": "NOT_REQUIRED",

            # tolerate model returning full names
            "SUPPORTED": "SUPPORTED",
            "LIKELY_SUPPORTED": "LIKELY_SUPPORTED",
            "UNKNOWN": "UNKNOWN",
            "CONTRADICTED": "CONTRADICTED",
            "NOT_REQUIRED": "NOT_REQUIRED",
        }

        def decode_status(
            value,
            allow_not_required=False,
        ):
            key = str(
                value or "U"
            ).strip().upper()

            status = code_map.get(
                key,
                "UNKNOWN",
            )

            if (
                status == "NOT_REQUIRED"
                and not allow_not_required
            ):
                status = "UNKNOWN"

            return status

        returned = {}

        for row in rows:

            if (
                not isinstance(row, list)
                or len(row) < 3
            ):
                continue

            asin = str(
                row[0] or ""
            ).strip()

            if not asin:
                continue

            asin_key = asin.lower()

            if asin_key not in cache_keys:
                continue

            type_status = decode_status(
                row[1]
            )

            req_codes = (
                row[2]
                if isinstance(row[2], list)
                else []
            )

            if price_constraint:
                price_status = decode_status(
                    row[3]
                    if len(row) >= 4
                    else "U",
                    allow_not_required=True,
                )
            else:
                price_status = "NOT_REQUIRED"

            # Use grounded visible text as metadata evidence.
            # The LLM no longer wastes tokens generating prose.
            source_text = str(
                missing.get(
                    asin,
                    missing.get(
                        asin_key,
                        "",
                    ),
                )
                or ""
            )

            source_text = " ".join(
                source_text.split()
            )[:240]

            constraints = []

            for i, req_name in enumerate(
                required_names
            ):

                status = decode_status(
                    req_codes[i]
                    if i < len(req_codes)
                    else "U"
                )

                constraints.append({
                    "constraint": req_name,
                    "status": status,
                    "evidence": source_text,
                })

            profile = {
                "asin": asin,
                "product_type": {
                    "status": type_status,
                    "evidence": source_text,
                },
                "constraints": constraints,
                "price": {
                    "status": price_status,
                    "evidence": source_text,
                },
            }

            returned[asin_key] = profile

            self._profile_cache[
                cache_keys[asin_key]
            ] = profile

        # -----------------------------------------------------
        # If model unexpectedly omitted a candidate, use UNKNOWN
        # for this decision, but DO NOT cache the omission.
        # That candidate can be retried next time.
        # -----------------------------------------------------
        for asin, visible_text in missing.items():

            key = str(asin).lower()

            if key in returned:
                merged[key] = returned[key]
                continue

            source_text = " ".join(
                str(visible_text or "").split()
            )[:240]

            merged[key] = {
                "asin": asin,
                "product_type": {
                    "status": "UNKNOWN",
                    "evidence": source_text,
                },
                "constraints": [
                    {
                        "constraint": req_name,
                        "status": "UNKNOWN",
                        "evidence": source_text,
                    }
                    for req_name
                    in required_names
                ],
                "price": {
                    "status": (
                        "UNKNOWN"
                        if price_constraint
                        else "NOT_REQUIRED"
                    ),
                    "evidence": source_text,
                },
            }

        payload = {
            "candidates": [
                merged[str(asin).lower()]
                for asin
                in candidate_blocks
                if str(asin).lower() in merged
            ]
        }

        return payload, content


    def _normalize_profile(
        self,
        raw_profile,
        asin,
        schema,
        visible_text,
    ):

        raw_profile = (
            raw_profile
            if isinstance(
                raw_profile,
                dict,
            )
            else {}
        )

        required = (
            schema.get(
                "required_constraints",
                [],
            )
            if isinstance(
                schema,
                dict,
            )
            else []
        )

        expected = [
            self._constraint_name(x)
            for x in required
        ]

        returned = {}

        for item in (
            raw_profile.get(
                "constraints",
                [],
            )
            or []
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            name = str(
                item.get(
                    "constraint",
                    "",
                )
            ).strip()

            if not name:
                continue

            returned[
                name.lower()
            ] = {
                "constraint":
                    name,

                "status":
                    self._clean_status(
                        item.get(
                            "status"
                        )
                    ),

                "evidence":
                    str(
                        item.get(
                            "evidence",
                            "",
                        )
                    ).strip(),
            }

        normalized_constraints = []

        for name in expected:

            item = returned.get(
                name.lower()
            )

            if item is None:

                item = {
                    "constraint":
                        name,

                    "status":
                        "UNKNOWN",

                    "evidence":
                        "",
                }

            normalized_constraints.append(
                item
            )

        ptype = raw_profile.get(
            "product_type",
            {},
        )

        if not isinstance(
            ptype,
            dict,
        ):
            ptype = {}

        price = raw_profile.get(
            "price",
            {},
        )

        if not isinstance(
            price,
            dict,
        ):
            price = {}

        profile = {
            "asin":
                asin,

            "product_type": {
                "status":
                    self._clean_status(
                        ptype.get(
                            "status"
                        )
                    ),

                "evidence":
                    str(
                        ptype.get(
                            "evidence",
                            "",
                        )
                    ).strip(),
            },

            "constraints":
                normalized_constraints,

            "price": {
                "status":
                    self._clean_status(
                        price.get(
                            "status"
                        )
                    ),

                "evidence":
                    str(
                        price.get(
                            "evidence",
                            "",
                        )
                    ).strip(),
            },

            "visible_text":
                visible_text,
        }

        score, hard_contradiction = (
            self._score_profile(
                profile,
                schema,
            )
        )

        profile[
            "overall_match_score"
        ] = score

        profile[
            "hard_contradiction"
        ] = hard_contradiction

        return profile

    def _score_profile(
        self,
        profile,
        schema,
    ):

        schema = (
            schema
            if isinstance(
                schema,
                dict,
            )
            else {}
        )

        ptype_required = bool(
            schema.get(
                "product_type"
            )
        )

        required = (
            schema.get(
                "required_constraints",
                [],
            )
            or []
        )

        price_required = bool(
            schema.get(
                "price_constraint"
            )
        )

        ptype_status = (
            profile[
                "product_type"
            ][
                "status"
            ]
        )

        constraint_statuses = [
            x["status"]
            for x in profile[
                "constraints"
            ]
        ]

        price_status = (
            profile[
                "price"
            ][
                "status"
            ]
        )

        hard_contradiction = bool(
            (
                ptype_required
                and ptype_status
                == "CONTRADICTED"
            )
            or any(
                x == "CONTRADICTED"
                for x in constraint_statuses
            )
            or (
                price_required
                and price_status
                == "CONTRADICTED"
            )
        )

        components = []

        if ptype_required:

            components.append(
                (
                    0.40,
                    self.STATUS_SCORE[
                        ptype_status
                    ],
                )
            )

        if required:

            req_score = (
                sum(
                    self.STATUS_SCORE[x]
                    for x
                    in constraint_statuses
                )
                / max(
                    1,
                    len(
                        constraint_statuses
                    ),
                )
            )

            components.append(
                (
                    0.60,
                    req_score,
                )
            )

        if price_required:

            # Add price as explicit component.
            # Existing weights are renormalized below.
            components.append(
                (
                    0.15,
                    self.STATUS_SCORE[
                        price_status
                    ],
                )
            )

        if not components:
            return 0.50, False

        total_weight = sum(
            w
            for w, _
            in components
        )

        score = sum(
            w * value
            for w, value
            in components
        ) / total_weight

        if hard_contradiction:
            score = min(
                score,
                0.49,
            )

        return (
            round(
                float(score),
                4,
            ),
            hard_contradiction,
        )

    # =========================================================
    # Weak-evidence handoff
    # =========================================================

    def _store_prior(
        self,
        instruction,
        schema,
        profile,
    ):

        if (
            self.constraint_manager
            is None
        ):
            return

        priors = getattr(
            self.constraint_manager,
            "_candidate_priors",
            None,
        )

        if not isinstance(
            priors,
            dict,
        ):
            priors = {}
            self.constraint_manager._candidate_priors = (
                priors
            )

        asin = str(
            profile.get(
                "asin",
                "",
            )
        ).lower()

        if not asin:
            return

        priors[asin] = {
            "asin":
                asin,

            "source":
                "search_result",

            "evidence_strength":
                "WEAK",

            "instruction":
                instruction,

            "request_schema":
                self._schema_view(
                    schema
                ),

            "overall_match_score":
                profile.get(
                    "overall_match_score"
                ),
"profile":
                profile,
        }

    # =========================================================
    # Memory
    # =========================================================

    def record_executed_action(
        self,
        action,
    ):

        product = (
            self._product_from_action(
                action
            )
        )

        if product:

            self.visited_candidates.add(
                product.lower()
            )

        query = self._search(action)

        if query:

            self.search_queries.append(
                self._norm_query(
                    query
                )
            )

            self.search_queries = (
                self.search_queries[-50:]
            )

    # =========================================================
    # Main gate
    # =========================================================

    def evaluate(
        self,
        instruction,
        observation,
        available_actions,
        proposed_action,
        constraint_schema=None,
        active_product=None,
    ):

        result = {
            "engaged":
                False,

            "action":
                proposed_action,

            "decision":
                "PASS",

            "changed":
                False,

            "reason":
                None,
"selected_score":
                None,

            "selected_profile":
                None,

            "candidate_profiles":
                [],

            "visible_candidates":
                [],

            "unseen_candidates":
                [],

            "llm_used":
                False,

            "llm_error":
                False,

            "llm_output":
                None,

            "repeat_product_blocked":
                False,

            "repeat_search_blocked":
                False,
}

        # Product page belongs to Product Gate.
        if active_product:

            result[
                "reason"
            ] = (
                "Product page belongs "
                "to ProductGate."
            )

            return result

        visible = (
            self._visible_products(
                available_actions
            )
        )

        result[
            "visible_candidates"
        ] = visible

        # =====================================================
        # STALE_REPEAT_RECOVERY_V1
        #
        # A Base model can emit a previously visited ASIN after
        # returning to the search landing page. On that page
        # there are no visible product IDs, so the old code
        # returned before repeat detection and could loop on the
        # same invalid/stale click for the rest of the episode.
        #
        # Detect that repeated product BEFORE the no-visible
        # early return and recover to the latest executed search.
        # =====================================================
        early_proposed_product = (
            self._product_from_action(
                proposed_action
            )
        )

        early_repeated_product = bool(
            early_proposed_product
            and early_proposed_product.lower()
            in self.visited_candidates
        )

        if (
            not visible
            and early_repeated_product
        ):

            last_query = (
                self.search_queries[-1]
                if self.search_queries
                else None
            )

            has_search_bar = False
            clickables = []

            if isinstance(
                available_actions,
                dict,
            ):
                has_search_bar = bool(
                    available_actions.get(
                        "has_search_bar",
                        False,
                    )
                )

                clickables = (
                    available_actions.get(
                        "clickables",
                        [],
                    )
                    or []
                )

            # Preferred recovery:
            # replay the latest actually executed search.
            if (
                has_search_bar
                and last_query
            ):

                final_action = (
                    f"search[{last_query}]"
                )

                result.update({
                    "engaged": True,
                    "action": final_action,
                    "decision": "RECOVER_SEARCH",
                    "changed": (
                        final_action.strip().lower()
                        != str(
                            proposed_action
                        ).strip().lower()
                    ),
                    "reason": (
                        "Blocked a stale repeated product click "
                        "on a page with no visible candidates; "
                        "replayed the latest executed search."
                    ),
                    "repeat_product_blocked": True,
                    "repeat_search_blocked": False,
                })

                print(
                    "[CandidateRecovery]",
                    "stale_product=",
                    early_proposed_product,
                    "->",
                    final_action,
                )

                return result

            # Secondary safe recovery if we are not on a page
            # with a search box but Back to Search is available.
            lower_clickables = {
                str(x).strip().lower()
                for x in clickables
            }

            if "back to search" in lower_clickables:

                final_action = (
                    "click[back to search]"
                )

                result.update({
                    "engaged": True,
                    "action": final_action,
                    "decision": "RECOVER_BACK",
                    "changed": (
                        final_action.strip().lower()
                        != str(
                            proposed_action
                        ).strip().lower()
                    ),
                    "reason": (
                        "Blocked a stale repeated product click "
                        "with no visible candidates and returned "
                        "to search."
                    ),
                    "repeat_product_blocked": True,
                    "repeat_search_blocked": False,
                })

                print(
                    "[CandidateRecovery]",
                    "stale_product=",
                    early_proposed_product,
                    "->",
                    final_action,
                )

                return result

        if not visible:

            result[
                "reason"
            ] = (
                "No visible product IDs."
            )

            return result

        unseen = [
            x
            for x in visible
            if x.lower()
            not in self.visited_candidates
        ]

        result[
            "unseen_candidates"
        ] = unseen

        proposed_product = (
            self._product_from_action(
                proposed_action
            )
        )

        proposed_query = (
            self._search(
                proposed_action
            )
        )

        query_norm = (
            self._norm_query(
                proposed_query
            )
            if proposed_query
            else None
        )

        repeated_product = bool(
            proposed_product
            and proposed_product.lower()
            in self.visited_candidates
        )

        repeated_search = bool(
            query_norm
            and query_norm
            in self.search_queries
        )

        # New query stays untouched.
        if (
            proposed_query
            and not repeated_search
        ):

            result[
                "reason"
            ] = (
                "New query; CandidateGate "
                "does not rewrite it."
            )

            return result

        click_value = (
            self._click(
                proposed_action
            )
        )

        if (
            click_value
            and click_value.lower()
            == "next >"
        ):

            result[
                "reason"
            ] = (
                "Base already chose pagination."
            )

            return result

        if (
            not proposed_product
            and not repeated_search
        ):

            result[
                "reason"
            ] = (
                "Action outside CandidateGate scope."
            )

            return result


        # =====================================================
        # Candidate Selector V3: REPEAT-ONLY
        #
        # Core principle:
        #
        # 1. NEVER override a first-time Base product click.
        # 2. NEVER rewrite a normal search action.
        # 3. Intervene ONLY when Base tries to click a product
        #    that has already been visited in this task.
        # 4. On repetition, rerank unseen visible candidates
        #    using the shared request schema.
        # 5. Search-result evidence remains WEAK evidence.
        # =====================================================

        # -----------------------------------------------------
        # Search actions are completely outside this version.
        # -----------------------------------------------------

        if proposed_query:

            result[
                "decision"
            ] = "PASS"

            result[
                "reason"
            ] = (
                "Repeat-only CandidateGate does not "
                "modify search actions."
            )

            return result

        # -----------------------------------------------------
        # Non-product actions are untouched.
        # -----------------------------------------------------

        if not proposed_product:

            result[
                "decision"
            ] = "PASS"

            result[
                "reason"
            ] = (
                "Action is not a product click."
            )

            return result

        # -----------------------------------------------------
        # MOST IMPORTANT RULE:
        #
        # First-time product click is NEVER overridden.
        # No candidate profiling call is made here.
        # -----------------------------------------------------

        if not repeated_product:

            result[
                "decision"
            ] = "PASS"

            result[
                "action"
            ] = proposed_action

            result[
                "changed"
            ] = False

            result[
                "reason"
            ] = (
                "First-time product click; "
                "preserve Base selection."
            )

            return result

        # =====================================================
        # From here onward, Base is repeating a visited product.
        # CandidateGate is allowed to intervene.
        # =====================================================

        result[
            "engaged"
        ] = True

        result[
            "repeat_product_blocked"
        ] = True

        result[
            "repeat_search_blocked"
        ] = False

        # -----------------------------------------------------
        # If every product on the page has already been visited,
        # paginate rather than re-clicking the same product.
        # -----------------------------------------------------

        if not unseen:

            if self._has_next(
                available_actions
            ):

                final_action = (
                    "click[next >]"
                )

                result.update({
                    "action":
                        final_action,

                    "decision":
                        "NEXT",

                    "changed":
                        final_action.lower()
                        != str(
                            proposed_action
                        ).strip().lower(),

                    "reason":
                        (
                            "Repeated product blocked; "
                            "all visible candidates "
                            "have already been visited."
                        ),
                })

            else:

                result[
                    "decision"
                ] = "PASS"

                result[
                    "reason"
                ] = (
                    "Repeated product detected, but "
                    "all visible candidates are visited "
                    "and no next page is available."
                )

            return result

        # -----------------------------------------------------
        # Build search-result text blocks ONLY for unseen
        # candidates. The repeated candidate itself is excluded.
        # -----------------------------------------------------

        blocks = (
            self._candidate_blocks(
                observation,
                unseen,
            )
        )

        try:

            payload, raw = (
                self._ask_profiles(
                    instruction=instruction,
                    constraint_schema=(
                        constraint_schema
                    ),
                    candidate_blocks=blocks,
                )
            )

            result[
                "llm_used"
            ] = bool(
                getattr(
                    self,
                    "_last_profile_llm_used",
                    True,
                )
            )

            result[
                "llm_output"
            ] = raw

        except Exception as exc:

            result[
                "llm_used"
            ] = True

            result[
                "llm_error"
            ] = True

            result[
                "decision"
            ] = "PASS"

            result[
                "reason"
            ] = (
                "Repeat detected, but candidate "
                "profiling failed: "
                + str(exc)
            )

            return result

        raw_candidates = {}

        for item in (
            payload.get(
                "candidates",
                [],
            )
            or []
        ):

            if not isinstance(
                item,
                dict,
            ):
                continue

            asin = str(
                item.get(
                    "asin",
                    "",
                )
            ).strip().lower()

            if asin:
                raw_candidates[
                    asin
                ] = item

        profiles = []

        for asin in unseen:

            key = asin.lower()

            profile = (
                self._normalize_profile(
                    raw_profile=(
                        raw_candidates.get(
                            key,
                            {},
                        )
                    ),
                    asin=asin,
                    schema=(
                        constraint_schema
                        or {}
                    ),
                    visible_text=(
                        blocks.get(
                            key,
                            "",
                        )
                    ),
                )
            )

            profiles.append(
                profile
            )

        profiles.sort(
            key=lambda x: float(
                x.get(
                    "overall_match_score",
                    0.0,
                )
                or 0.0
            ),
            reverse=True,
        )

        result[
            "candidate_profiles"
        ] = profiles

        # -----------------------------------------------------
        # Safe unseen candidates may replace the repeated
        # click regardless of their numeric score.
        # -----------------------------------------------------

        # Repeat-only, no-threshold Candidate policy.
        #
        # profiles is already sorted by overall_match_score.
        # Keep the hard-contradiction veto, but use no
        # minimum numeric score threshold.
        eligible = [
            profile
            for profile in profiles
            if not profile.get(
                "hard_contradiction",
                False,
            )
        ]

        if eligible:

            best = eligible[0]

            asin = str(
                best[
                    "asin"
                ]
            )

            final_action = (
                f"click[{asin}]"
            )

            # Candidate -> Product weak evidence handoff.
            self._store_prior(
                instruction=instruction,
                schema=(
                    constraint_schema
                    or {}
                ),
                profile=best,
            )

            result.update({
                "action":
                    final_action,

                # Keep original stats-compatible label.
                "decision":
                    "SELECT",

                "changed":
                    final_action.lower()
                    != str(
                        proposed_action
                    ).strip().lower(),

                "reason":
                    (
                        "Blocked repeated product and "
                        "selected the highest-scoring "
                        "schema-aligned unseen candidate."
                    ),

                "selected_score":
                    best.get(
                        "overall_match_score"
                    ),

                "selected_profile":
                    best,
            })

            return result

        # -----------------------------------------------------
        # No safe unseen candidate remains on this page.
        # Prefer exploring the next results page.
        # -----------------------------------------------------
        if self._has_next(
            available_actions
        ):

            final_action = (
                "click[next >]"
            )

            result.update({
                "action":
                    final_action,

                # Keep original stats-compatible label.
                "decision":
                    "NEXT",

                "changed":
                    True,

                "reason":
                    (
                        "Repeated product blocked; "
                        "no safe unseen candidate remains "
                        "on this page, so paginate."
                    ),
            })

            return result

        # No safe alternative and no pagination available.
        # We do NOT invent a candidate.
        result[
            "decision"
        ] = "PASS"

        result[
            "action"
        ] = proposed_action

        result[
            "changed"
        ] = False

        result[
            "reason"
        ] = (
            "Repeated product detected, but no "
            "safe unseen candidate and no next "
            "page are available."
        )

        return result

