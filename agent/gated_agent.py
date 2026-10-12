import json
import re

import openai

from agent.baseline_agent import BaselineAgent
from evaluation.result_schema import make_result
from gates.constraint_manager import ConstraintManager
from gates.query_gate import QueryGate
from gates.candidate_gate import CandidateGate
from gates.product_gate import ProductGate
from webshop_wrapper.env import WebShopWrapper


class GatedAgent(BaselineAgent):

    def __init__(
        self,
        num_products=1000,
        max_steps=20,
        use_query_gate=True,
        use_candidate_gate=False,
        use_product_gate=True,
        webshop_file_path=None,
        webshop_attr_path=None,
        webshop_human_attr_path=None,
        webshop_index_path=None,
    ):
        super().__init__(
            num_products=num_products,
            max_steps=max_steps,
            webshop_file_path=webshop_file_path,
            webshop_attr_path=webshop_attr_path,
            webshop_human_attr_path=webshop_human_attr_path,
            webshop_index_path=webshop_index_path,
        )

        self.use_query_gate = use_query_gate
        self.use_candidate_gate = use_candidate_gate
        self.use_product_gate = use_product_gate

        # =====================================================
        # ONE shared ConstraintManager
        #
        # QueryGate + ProductGate share:
        # - instruction extraction
        # - extraction validation
        # - schema cache
        # - semantic matching
        # =====================================================

        self.constraint_manager = None
        self.query_gate = None
        self.candidate_gate = None
        self.product_gate = None

        if (
            use_query_gate
            or use_candidate_gate
            or use_product_gate
        ):

            self.constraint_manager = (
                ConstraintManager()
            )

        if use_query_gate:

            self.query_gate = QueryGate(
                constraint_manager=(
                    self.constraint_manager
                )
            )

        if use_candidate_gate:

            self.candidate_gate = CandidateGate(
                constraint_manager=(
                    self.constraint_manager
                )
            )

        if use_product_gate:

            self.product_gate = ProductGate(
                constraint_manager=(
                    self.constraint_manager
                )
            )


        if (
            use_query_gate
            and use_product_gate
        ):
            self.name = "query_product_gate"

        elif use_query_gate:
            self.name = "query_gate"

        elif use_product_gate:
            self.name = "product_gate"

        else:
            self.name = "baseline_gated_wrapper"


    # =========================================================
    # Action helpers
    # =========================================================

    @staticmethod
    def _extract_search_query(action):

        if not action:
            return None

        match = re.fullmatch(
            r"\s*search\[(.*?)\]\s*",
            str(action),
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        if not match:
            return None

        return (
            match.group(1)
            .strip()
        )


    @staticmethod
    def _extract_click_value(action):

        if not action:
            return None

        match = re.fullmatch(
            r"\s*click\[(.*?)\]\s*",
            str(action),
            flags=(
                re.IGNORECASE
                | re.DOTALL
            ),
        )

        if not match:
            return None

        return (
            match.group(1)
            .strip()
        )


    @classmethod
    def _is_search_action(
        cls,
        action,
    ):

        return (
            cls._extract_search_query(
                action
            )
            is not None
        )


    @staticmethod
    def _is_buy_action(action):

        return (
            str(action)
            .strip()
            .lower()
            == "click[buy now]"
        )


    @classmethod
    def _is_back_to_search(
        cls,
        action,
    ):

        value = (
            cls._extract_click_value(
                action
            )
        )

        return (
            value is not None
            and value.lower()
            == "back to search"
        )


    @classmethod
    def _is_product_click(
        cls,
        action,
    ):

        value = (
            cls._extract_click_value(
                action
            )
        )

        if not value:
            return False


        value_lower = (
            value.lower()
        )


        navigation = {
            "features",
            "description",
            "reviews",
            "buy now",
            "back to search",
            "< prev",
            "next >",
        }


        if value_lower in navigation:

            return False


        # WebShop product IDs are normally ASIN-like.
        return bool(
            re.fullmatch(
                r"[a-z0-9]{8,12}",
                value_lower,
            )
        )


    @classmethod
    def _is_candidate_inspection_action(
        cls,
        action,
    ):

        value = (
            cls._extract_click_value(
                action
            )
        )

        if not value:
            return False


        value_lower = (
            value.lower()
        )


        return value_lower in {
            "features",
            "description",
            "reviews",
            "< prev",
        }


    @classmethod
    def _is_candidate_option_action(
        cls,
        action,
    ):

        value = (
            cls._extract_click_value(
                action
            )
        )

        if not value:
            return False


        value_lower = (
            value.lower()
        )


        excluded = {
            "features",
            "description",
            "reviews",
            "buy now",
            "back to search",
            "< prev",
            "next >",
        }


        if value_lower in excluded:
            return False


        if cls._is_product_click(
            action
        ):
            return False


        # Anything else clicked while a candidate is active
        # may be a color / size / scent / pack option.
        return True


    # =========================================================
    # Candidate evidence memory
    # =========================================================

    def _clean_product_observation(
        self,
        instruction,
        observation,
    ):

        if not self.product_gate:

            return str(
                observation or ""
            )


        # ProductGate already contains the important
        # instruction-leakage remover.
        return (
            self.product_gate
            ._strip_instruction(
                instruction,
                observation,
            )
        )


    def _add_candidate_evidence(
        self,
        pages,
        instruction,
        observation,
        source,
    ):

        cleaned = (
            self._clean_product_observation(
                instruction,
                observation,
            )
        ).strip()


        if not cleaned:
            return False


        normalized = (
            ConstraintManager
            .normalize_text(
                cleaned
            )
        )


        if not normalized:
            return False


        # Avoid storing identical pages repeatedly.
        for item in pages:

            if (
                item.get(
                    "normalized"
                )
                == normalized
            ):

                return False


        pages.append({
            "source": source,
            "text": cleaned,
            "normalized": normalized,
        })


        # Enough for WebShop while avoiding unlimited prompt growth.
        if len(pages) > 10:

            del pages[:-10]


        return True


    @staticmethod
    def _build_candidate_evidence(
        pages,
    ):

        blocks = []


        for index, item in enumerate(
            pages,
            start=1,
        ):

            blocks.append(
                (
                    f"[CANDIDATE EVIDENCE {index}: "
                    f"{item.get('source', 'unknown')}]\n"
                    f"{item.get('text', '')}"
                )
            )


        evidence = "\n\n".join(
            blocks
        )


        # Keep prompt bounded.
        if len(evidence) > 16000:

            evidence = evidence[
                -16000:
            ]


        return evidence


    # =========================================================
    # Compact Product Gate feedback for Base LLM
    # =========================================================

    @staticmethod
    def _compact_product_feedback(
        gate_result,
    ):

        if not isinstance(
            gate_result,
            dict,
        ):
            return None


        return {
            "decision": (
                gate_result.get(
                    "decision"
                )
            ),

            "ready_to_buy": (
                gate_result.get(
                    "ready_to_buy"
                )
            ),

            "keep_candidate": (
                gate_result.get(
                    "should_keep_candidate"
                )
            ),

            "matched_constraints": (
                gate_result.get(
                    "matched_constraints",
                    [],
                )
            ),

            "missing_constraints": (
                gate_result.get(
                    "missing_constraints",
                    [],
                )
            ),

            "contradicted_constraints": (
                gate_result.get(
                    "contradicted_constraints",
                    [],
                )
            ),

            "recommended_action": (
                gate_result.get(
                    "recommended_action"
                )
            ),

            "reason": (
                gate_result.get(
                    "reason"
                )
            ),
        }


    # =========================================================
    # Base LLM with active Product Gate feedback
    # =========================================================

    def ask_deepseek_with_product_feedback(
        self,
        instruction,
        history,
        observation,
        available_actions,
        product_feedback,
        previous_proposal=None,
    ):

        compact_feedback = (
            self._compact_product_feedback(
                product_feedback
            )
        )


        proposal_text = ""

        if previous_proposal:

            proposal_text = f"""
Your previous proposed action was:
{previous_proposal}

You are being asked to reconsider that action using the
Product Gate feedback below.
"""


        prompt = f"""
You are a shopping agent operating WebShop.

User instruction:
{instruction}

Action history:
{json.dumps(history, ensure_ascii=False)}

Current observation:
{observation}

Available actions:
{json.dumps(available_actions, ensure_ascii=False)}

{proposal_text}

Product Gate feedback for the CURRENT product candidate:
{json.dumps(compact_feedback, ensure_ascii=False)}

Choose exactly ONE next action.

Allowed formats:
search[query]
click[item]


IMPORTANT PRODUCT GATE GUIDANCE:

If Product Gate says READY:
- The currently selected product already has enough accumulated
  visible evidence to satisfy the user's constraints.
- Do NOT restart search without a new visible contradiction.
- If "buy now" is currently available, strongly prefer:
  click[buy now]
- If "buy now" is not available because you are on Features,
  Description, or Reviews, navigate back to the product page
  (for example click[< prev]) while keeping this candidate.

If Product Gate says INSPECT:
- The current product is still a valid candidate.
- Do NOT abandon it merely because one page lacks evidence.
- Inspect the missing evidence using the recommended action,
  Features, Description, Reviews, or required product options.

If Product Gate says REJECT:
- The current product should not be purchased.
- Leave it and consider another product.

If Product Gate says EXHAUSTED:
- All useful evidence pages for this candidate have already
  been inspected.
- Some required evidence is still missing.
- Do NOT inspect Features, Description, or Reviews again.
- Leave this candidate and search for another product.
- Prefer click[back to search] when it is available.

If Product Gate says PASS_THROUGH:
- Product Gate does not have a reliable constraint schema.
- Ignore Product Gate and follow the normal Base policy.

Important:
- Product Gate feedback is advisory evidence, not hidden ground truth.
- Do not invent product properties.
- Preserve the user's original constraint meaning.
- Do not repeat useless actions.
- Output ONLY one action with no explanation.
"""


        response = (
            openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )
        )


        return (
            response[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]
            .strip()
        )


    # =========================================================
    # Product feedback action reconsideration
    # =========================================================

    def _should_reconsider_with_product_gate(
        self,
        action,
        product_feedback,
    ):

        if not isinstance(
            product_feedback,
            dict,
        ):
            return False


        decision = str(
            product_feedback.get(
                "decision",
                "",
            )
        ).upper()


        keep_candidate = bool(
            product_feedback.get(
                "should_keep_candidate"
            )
        )


        # Prevent a clearly premature purchase from simply
        # bypassing the Product Gate.
        if (
            self._is_buy_action(
                action
            )
            and decision != "READY"
        ):

            return True


        # Main failure mode:
        # Product Gate says keep this candidate,
        # while Base wants to abandon it.
        abandoning = (
            self._is_search_action(
                action
            )
            or self._is_back_to_search(
                action
            )
        )


        if (
            keep_candidate
            and decision in {
                "READY",
                "INSPECT",
            }
            and abandoning
        ):

            return True


        return False


    def _recover_from_product_feedback(
        self,
        instruction,
        history,
        observation,
        available_actions,
        proposed_action,
        product_feedback,
    ):

        try:

            llm_output = (
                self.ask_deepseek_with_product_feedback(
                    instruction=instruction,
                    history=history,
                    observation=observation,
                    available_actions=available_actions,
                    product_feedback=product_feedback,
                    previous_proposal=proposed_action,
                )
            )

            candidate_action = (
                self.parse_action(
                    llm_output
                )
            )


        except Exception as e:

            return {
                "action": proposed_action,
                "llm_output": None,
                "changed": False,
                "reason": (
                    "Product feedback reconsideration "
                    "API error: "
                    + str(e)
                ),
            }


        if candidate_action is None:

            return {
                "action": proposed_action,
                "llm_output": llm_output,
                "changed": False,
                "reason": (
                    "Product feedback reconsideration "
                    "returned invalid action."
                ),
            }


        decision = str(
            product_feedback.get(
                "decision",
                "",
            )
        ).upper()


        # Product is not READY:
        # recovery is not allowed to buy it.
        if (
            self._is_buy_action(
                candidate_action
            )
            and decision != "READY"
        ):

            recommended = (
                product_feedback.get(
                    "recommended_action"
                )
            )

            recommended_action = (
                self.parse_action(
                    recommended
                )
                if recommended
                else None
            )


            if (
                recommended_action is not None
                and not self._is_buy_action(
                    recommended_action
                )
            ):

                candidate_action = (
                    recommended_action
                )

            else:

                return {
                    "action": proposed_action,
                    "llm_output": llm_output,
                    "changed": False,
                    "reason": (
                        "Recovery still proposed buy "
                        "before Product Gate was READY."
                    ),
                }


        changed = (
            candidate_action.strip().lower()
            != proposed_action.strip().lower()
        )


        return {
            "action": candidate_action,
            "llm_output": llm_output,
            "changed": changed,
            "reason": (
                "Base reconsidered its action using "
                "Product Gate feedback."
            ),
        }


    # =========================================================
    # Query Gate recovery
    # =========================================================

    def ask_after_query_reject(
        self,
        instruction,
        history,
        observation,
        available_actions,
        original_action,
        gate_result,
    ):

        prompt = f"""
You are a shopping agent operating WebShop.

User instruction:
{instruction}

Action history:
{json.dumps(history, ensure_ascii=False)}

Current observation:
{observation}

Available actions:
{json.dumps(available_actions, ensure_ascii=False)}

Your previously proposed search action was:
{original_action}

The Query Gate found that this search query failed to preserve
one or more user constraints.

Query Gate feedback:
{json.dumps(gate_result, ensure_ascii=False)}

Generate ONE corrected search action.

Important:
- Preserve the requested product type.
- Restore missing explicit constraints.
- Preserve semantic direction.
- Do not invent requirements.
- Output ONLY:
search[query]
"""


        response = (
            openai.ChatCompletion.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0,
            )
        )


        return (
            response[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]
            .strip()
        )


    def _recover_query_action(
        self,
        instruction,
        history,
        observation,
        available_actions,
        original_action,
        gate_result,
    ):

        gate_query = (
            gate_result.get(
                "revised_query"
            )
        )


        fallback_action = (
            f"search[{gate_query}]"
            if gate_query
            else original_action
        )


        try:

            llm_output = (
                self.ask_after_query_reject(
                    instruction=instruction,
                    history=history,
                    observation=observation,
                    available_actions=available_actions,
                    original_action=original_action,
                    gate_result=gate_result,
                )
            )


            candidate_action = (
                self.parse_action(
                    llm_output
                )
            )


        except Exception as e:

            return {
                "action": fallback_action,
                "llm_output": None,
                "candidate_action": None,
                "candidate_validation": None,
                "fallback_used": True,
                "fallback_reason": (
                    "Query recovery API error: "
                    + str(e)
                ),
            }


        candidate_query = (
            self._extract_search_query(
                candidate_action
            )
        )


        if not candidate_query:

            return {
                "action": fallback_action,
                "llm_output": llm_output,
                "candidate_action": candidate_action,
                "candidate_validation": None,
                "fallback_used": True,
                "fallback_reason": (
                    "Query recovery did not return "
                    "a valid search action."
                ),
            }


        validation = (
            self.query_gate
            .validate_query(
                instruction=instruction,
                proposed_query=candidate_query,
            )
        )


        if (
            validation.get(
                "decision"
            )
            == "PASS"
        ):

            return {
                "action": (
                    f"search[{candidate_query}]"
                ),
                "llm_output": llm_output,
                "candidate_action": candidate_action,
                "candidate_validation": validation,
                "fallback_used": False,
                "fallback_reason": None,
            }


        deterministic_query = (
            validation.get(
                "revised_query"
            )
            or gate_query
        )


        final_action = (
            f"search[{deterministic_query}]"
            if deterministic_query
            else fallback_action
        )


        return {
            "action": final_action,
            "llm_output": llm_output,
            "candidate_action": candidate_action,
            "candidate_validation": validation,
            "fallback_used": True,
            "fallback_reason": (
                "Recovery query still failed "
                "constraint validation."
            ),
        }


    # =========================================================
    # Main run
    # =========================================================

    def run(self, task):

        instruction = (
            task["instruction"]
        )

        task_id = (
            task["task_id"]
        )


        env = WebShopWrapper(
            num_products=self.num_products
        )


        observation = env.reset(
            task
        )


        # Same shape as baseline.
        # Gate feedback is NOT inserted permanently here.
        policy_history = []


        trajectory = []


        final_reward = 0.0
        success = False


        # =====================================================
        # Active candidate state
        # =====================================================

        active_product = None

        candidate_evidence_pages = []

        current_product_feedback = None


        stats = {
            "use_query_gate": (
                self.use_query_gate
            ),

            "use_candidate_gate": (
                self.use_candidate_gate
            ),

            "use_product_gate": (
                self.use_product_gate
            ),

            "candidate_gate_calls": 0,
            "candidate_gate_changes": 0,
            "candidate_gate_selects": 0,
            "candidate_gate_passes": 0,
            "candidate_gate_next_pages": 0,
            "candidate_gate_llm_calls": 0,
            "candidate_gate_llm_errors": 0,
            "candidate_repeat_product_blocks": 0,
            "candidate_repeat_search_blocks": 0,

            "query_gate_calls": 0,
            "query_gate_passes": 0,
            "query_gate_revisions": 0,
            "query_gate_feedback_calls": 0,
            "query_gate_recovery_fallbacks": 0,

            "product_gate_calls": 0,
            "product_gate_accepts": 0,
            "product_gate_ready": 0,
            "product_gate_inspects": 0,
            "product_gate_rejections": 0,

            "product_feedback_injected_steps": 0,

            "product_reconsider_calls": 0,
            "product_reconsider_changes": 0,

            "premature_buy_reconsiderations": 0,
            "candidate_abandon_reconsiderations": 0,

            "candidate_resets": 0,
            "candidate_evidence_pages_added": 0,
        }


        # =====================================================
        # Extract and validate constraints ONCE
        # =====================================================

        constraint_schema = None


        if self.constraint_manager:

            constraint_schema = (
                self.constraint_manager
                .get_constraints(
                    instruction
                )
            )


            print()
            print(
                "[ConstraintManager]"
            )

            print(
                json.dumps(
                    {
                        "product_type": (
                            constraint_schema.get(
                                "product_type"
                            )
                        ),

                        "required_constraints": (
                            constraint_schema.get(
                                "required_constraints"
                            )
                        ),

                        "post_selection_constraints": (
                            constraint_schema.get(
                                "post_selection_constraints"
                            )
                        ),

                        "price_constraint": (
                            constraint_schema.get(
                                "price_constraint"
                            )
                        ),

                        "validation": (
                            constraint_schema.get(
                                "validation"
                            )
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


        try:

            for step in range(
                1,
                self.max_steps + 1,
            ):

                available_actions = (
                    env.get_available_actions()
                )


                # Save feedback state BEFORE this action.
                product_feedback_before_action = (
                    current_product_feedback
                )


                # =================================================
                # BASE PROPOSAL
                #
                # If a candidate is active, Product Gate feedback
                # is continuously visible to Base.
                # =================================================

                if (
                    self.use_product_gate
                    and active_product
                    and current_product_feedback
                ):

                    stats[
                        "product_feedback_injected_steps"
                    ] += 1


                    llm_output = (
                        self.ask_deepseek_with_product_feedback(
                            instruction=instruction,
                            history=policy_history,
                            observation=observation,
                            available_actions=available_actions,
                            product_feedback=(
                                current_product_feedback
                            ),
                        )
                    )


                else:

                    llm_output = (
                        self.ask_deepseek(
                            instruction,
                            policy_history,
                            observation,
                            available_actions,
                        )
                    )


                proposed_action = (
                    self.parse_action(
                        llm_output
                    )
                )


                if proposed_action is None:

                    print(
                        f"[Base] Step {step}: "
                        f"invalid output: "
                        f"{llm_output}"
                    )

                    break


                print()
                print(
                    f"[Base] Step {step} proposed -> "
                    f"{proposed_action}"
                )


                action = proposed_action


                # =================================================
                # PRODUCT FEEDBACK PRE-ACTION RECONSIDERATION
                #
                # Product Gate still NEVER directly buys.
                #
                # We only ask Base to reconsider when:
                # - it tries to abandon a candidate Gate says keep
                # - it tries to buy before Gate is READY
                # =================================================

                product_reconsider_llm_output = None

                product_reconsider_reason = None

                product_reconsider_changed = False


                if (
                    self.use_product_gate
                    and active_product
                    and current_product_feedback
                    and self._should_reconsider_with_product_gate(
                        action,
                        current_product_feedback,
                    )
                ):

                    stats[
                        "product_reconsider_calls"
                    ] += 1


                    if self._is_buy_action(
                        action
                    ):

                        stats[
                            "premature_buy_reconsiderations"
                        ] += 1

                    else:

                        stats[
                            "candidate_abandon_reconsiderations"
                        ] += 1


                    recovery = (
                        self._recover_from_product_feedback(
                            instruction=instruction,
                            history=policy_history,
                            observation=observation,
                            available_actions=available_actions,
                            proposed_action=action,
                            product_feedback=(
                                current_product_feedback
                            ),
                        )
                    )


                    product_reconsider_llm_output = (
                        recovery.get(
                            "llm_output"
                        )
                    )

                    product_reconsider_reason = (
                        recovery.get(
                            "reason"
                        )
                    )


                    new_action = (
                        recovery.get(
                            "action"
                        )
                    )


                    if new_action:

                        action = new_action


                    product_reconsider_changed = (
                        recovery.get(
                            "changed",
                            False,
                        )
                    )


                    if product_reconsider_changed:

                        stats[
                            "product_reconsider_changes"
                        ] += 1


                    print(
                        "[ProductGate Feedback]"
                    )

                    print(
                        "  previous proposal:",
                        proposed_action,
                    )

                    print(
                        "  decision:",
                        current_product_feedback.get(
                            "decision"
                        ),
                    )

                    print(
                        "  recommended:",
                        current_product_feedback.get(
                            "recommended_action"
                        ),
                    )

                    print(
                        "  reconsider output:",
                        product_reconsider_llm_output,
                    )

                    print(
                        "  action after reconsider:",
                        action,
                    )


                # =================================================
                # =================================================
                # =================================================
                # CANDIDATE GATE
                #
                # Runs only on search-result candidate selection.
                # =================================================

                candidate_gate_result = None

                candidate_gate_input_action = (
                    action
                )


                if self.use_candidate_gate:

                    candidate_gate_result = (
                        self.candidate_gate
                        .evaluate(
                            instruction=instruction,
                            observation=observation,
                            available_actions=(
                                available_actions
                            ),
                            proposed_action=action,
                            constraint_schema=(
                                constraint_schema
                            ),
                            active_product=(
                                active_product
                            ),
                        )
                    )


                    engaged = (
                        candidate_gate_result
                        .get(
                            "engaged",
                            False,
                        )
                    )


                    if engaged:

                        stats[
                            "candidate_gate_calls"
                        ] += 1


                        decision = str(
                            candidate_gate_result
                            .get(
                                "decision",
                                "PASS",
                            )
                        ).upper()


                        if decision.startswith(
                            "SELECT"
                        ):

                            stats[
                                "candidate_gate_selects"
                            ] += 1


                        elif decision == "NEXT":

                            stats[
                                "candidate_gate_next_pages"
                            ] += 1


                        else:

                            stats[
                                "candidate_gate_passes"
                            ] += 1


                        if (
                            candidate_gate_result
                            .get(
                                "repeat_product_blocked",
                                False,
                            )
                        ):

                            stats[
                                "candidate_repeat_product_blocks"
                            ] += 1


                        if (
                            candidate_gate_result
                            .get(
                                "repeat_search_blocked",
                                False,
                            )
                        ):

                            stats[
                                "candidate_repeat_search_blocks"
                            ] += 1


                    if (
                        candidate_gate_result
                        .get(
                            "llm_used",
                            False,
                        )
                    ):

                        stats[
                            "candidate_gate_llm_calls"
                        ] += 1


                    if (
                        candidate_gate_result
                        .get(
                            "llm_error",
                            False,
                        )
                    ):

                        stats[
                            "candidate_gate_llm_errors"
                        ] += 1


                    new_action = (
                        candidate_gate_result
                        .get(
                            "action"
                        )
                    )


                    if new_action:

                        action = new_action


                    if (
                        candidate_gate_result
                        .get(
                            "changed",
                            False,
                        )
                    ):

                        stats[
                            "candidate_gate_changes"
                        ] += 1


                    if engaged:

                        print(
                            "[CandidateGate]"
                        )

                        print(
                            "  input:",
                            candidate_gate_input_action,
                        )

                        print(
                            "  decision:",
                            candidate_gate_result
                            .get(
                                "decision"
                            ),
                        )

                        print(
                            "  visible:",
                            candidate_gate_result
                            .get(
                                "visible_candidates"
                            ),
                        )

                        print(
                            "  unseen:",
                            candidate_gate_result
                            .get(
                                "unseen_candidates"
                            ),
                        )

                        print(
                            "  final:",
                            action,
                        )

                        print(
                            "  reason:",
                            candidate_gate_result
                            .get(
                                "reason"
                            ),
                        )


                # =================================================
                # QUERY GATE
                #
                # Runs on the action that is ACTUALLY about to
                # become a search action.
                # =================================================

                query_gate_result = None

                query_recovery_llm_output = None

                query_recovery_candidate_action = None

                query_recovery_validation = None

                query_recovery_fallback_used = False

                query_recovery_fallback_reason = None


                query_input_action = action

                original_query = (
                    self._extract_search_query(
                        query_input_action
                    )
                )


                if (
                    self.use_query_gate
                    and original_query
                ):

                    stats[
                        "query_gate_calls"
                    ] += 1


                    query_gate_result = (
                        self.query_gate.evaluate(
                            instruction=instruction,
                            proposed_query=original_query,
                            history=policy_history,
                        )
                    )


                    print(
                        "[QueryGate]"
                    )

                    print(
                        "  query:",
                        original_query,
                    )

                    print(
                        "  decision:",
                        query_gate_result.get(
                            "decision"
                        ),
                    )

                    print(
                        "  missing:",
                        query_gate_result.get(
                            "missing_constraints"
                        ),
                    )


                    if (
                        query_gate_result.get(
                            "decision"
                        )
                        == "PASS"
                    ):

                        stats[
                            "query_gate_passes"
                        ] += 1


                    else:

                        stats[
                            "query_gate_revisions"
                        ] += 1

                        stats[
                            "query_gate_feedback_calls"
                        ] += 1


                        recovery = (
                            self._recover_query_action(
                                instruction=instruction,
                                history=policy_history,
                                observation=observation,
                                available_actions=available_actions,
                                original_action=query_input_action,
                                gate_result=query_gate_result,
                            )
                        )


                        action = (
                            recovery.get(
                                "action",
                                action,
                            )
                        )


                        query_recovery_llm_output = (
                            recovery.get(
                                "llm_output"
                            )
                        )

                        query_recovery_candidate_action = (
                            recovery.get(
                                "candidate_action"
                            )
                        )

                        query_recovery_validation = (
                            recovery.get(
                                "candidate_validation"
                            )
                        )

                        query_recovery_fallback_used = (
                            recovery.get(
                                "fallback_used",
                                False,
                            )
                        )

                        query_recovery_fallback_reason = (
                            recovery.get(
                                "fallback_reason"
                            )
                        )


                        if query_recovery_fallback_used:

                            stats[
                                "query_gate_recovery_fallbacks"
                            ] += 1


                        print(
                            "  final search action:",
                            action,
                        )


                # =================================================
                # EXECUTE
                # =================================================

                action_changed_by_gate = (
                    action.strip().lower()
                    != proposed_action.strip().lower()
                )


                print(
                    "[Execute]"
                )

                print(
                    "  original:",
                    proposed_action,
                )

                print(
                    "  final:   ",
                    action,
                )

                print(
                    "  changed_by_gate:",
                    action_changed_by_gate,
                )


                (
                    new_observation,
                    reward,
                    done,
                    info,
                ) = env.step(
                    action
                )


                if self.use_candidate_gate:

                    self.candidate_gate.record_executed_action(
                        action
                    )


                print(
                    "  reward:",
                    reward,
                )

                print(
                    "  done:",
                    done,
                )


                current_history_item = {
                    "step": step,
                    "action": action,
                    "reward": reward,
                    "done": done,
                }


                # =================================================
                # UPDATE CANDIDATE MEMORY AFTER ACTION
                # =================================================

                product_gate_after_action = None

                candidate_evidence_added = False

                product_gate_trigger = None


                if self.use_product_gate:

                    # ------------------------------------------------
                    # Search / Back to Search:
                    # candidate lifecycle ends.
                    # ------------------------------------------------

                    if (
                        self._is_search_action(
                            action
                        )
                        or self._is_back_to_search(
                            action
                        )
                    ):

                        if active_product:

                            stats[
                                "candidate_resets"
                            ] += 1


                        active_product = None

                        candidate_evidence_pages = []

                        current_product_feedback = None


                    # ------------------------------------------------
                    # Enter a product:
                    # NEW candidate lifecycle.
                    # ------------------------------------------------

                    elif self._is_product_click(
                        action
                    ):

                        active_product = (
                            self._extract_click_value(
                                action
                            )
                        )


                        candidate_evidence_pages = []

                        current_product_feedback = None


                        candidate_evidence_added = (
                            self._add_candidate_evidence(
                                pages=(
                                    candidate_evidence_pages
                                ),
                                instruction=instruction,
                                observation=new_observation,
                                source=(
                                    "product entry: "
                                    + str(
                                        active_product
                                    )
                                ),
                            )
                        )


                        if candidate_evidence_added:

                            stats[
                                "candidate_evidence_pages_added"
                            ] += 1


                        product_gate_trigger = (
                            "product_entry"
                        )


                    # ------------------------------------------------
                    # Current candidate is still active.
                    #
                    # Add evidence from:
                    # - features
                    # - description
                    # - reviews
                    # - returning with < prev>
                    # - selectable product option
                    # ------------------------------------------------

                    elif active_product:

                        if (
                            self._is_candidate_inspection_action(
                                action
                            )
                            or self._is_candidate_option_action(
                                action
                            )
                        ):

                            clicked_value = (
                                self._extract_click_value(
                                    action
                                )
                            )


                            candidate_evidence_added = (
                                self._add_candidate_evidence(
                                    pages=(
                                        candidate_evidence_pages
                                    ),
                                    instruction=instruction,
                                    observation=new_observation,
                                    source=(
                                        "after click["
                                        + str(
                                            clicked_value
                                        )
                                        + "]"
                                    ),
                                )
                            )


                            if candidate_evidence_added:

                                stats[
                                    "candidate_evidence_pages_added"
                                ] += 1


                            # Even if < prev> returns a duplicate
                            # product page, re-evaluate because the
                            # available actions changed and buy now
                            # may now be available.
                            product_gate_trigger = (
                                "candidate_inspection"
                            )


                    # ------------------------------------------------
                    # Evaluate Product Gate using ALL accumulated
                    # evidence for this candidate.
                    # ------------------------------------------------

                    if (
                        active_product
                        and product_gate_trigger
                        and not done
                    ):

                        accumulated_evidence = (
                            self._build_candidate_evidence(
                                candidate_evidence_pages
                            )
                        )


                        next_available_actions = (
                            env.get_available_actions()
                        )


                        product_gate_after_action = (
                            self.product_gate.evaluate(
                                instruction=instruction,
                                observation=(
                                    accumulated_evidence
                                ),
                                history=(
                                    policy_history
                                    + [
                                        current_history_item
                                    ]
                                ),
                                available_actions=(
                                    next_available_actions
                                ),
                            )
                        )


                        # =====================================
                        # PRODUCT GATE POST-DECISION GUARDS
                        #
                        # These guards operate only on:
                        # - user instruction / extracted schema
                        # - visible candidate evidence
                        # - Product Gate's own result
                        #
                        # They never use target ASIN, hidden attrs,
                        # or reward.
                        # =====================================

                        guard_decision = str(
                            product_gate_after_action.get(
                                "decision",
                                "",
                            )
                        ).upper()


                        # =====================================
                        # 1. INSPECTION EXHAUSTION
                        #
                        # Candidate Evidence Memory already
                        # labels pages like:
                        #
                        # after click[features]
                        # after click[description]
                        # after click[reviews]
                        #
                        # Once all three visible evidence
                        # surfaces have been inspected and some
                        # required constraint is STILL missing,
                        # do not inspect them forever.
                        # =====================================

                        if guard_decision == "INSPECT":

                            guard_missing = list(
                                product_gate_after_action.get(
                                    "missing_constraints",
                                    [],
                                )
                                or []
                            )


                            evidence_lower = str(
                                accumulated_evidence
                            ).lower()


                            all_evidence_seen = all(
                                (
                                    "after click["
                                    + section
                                    + "]"
                                )
                                in evidence_lower

                                for section in (
                                    "features",
                                    "description",
                                    "reviews",
                                )
                            )


                            if (
                                guard_missing
                                and all_evidence_seen
                            ):

                                product_gate_after_action = dict(
                                    product_gate_after_action
                                )


                                clickables = []

                                if isinstance(
                                    next_available_actions,
                                    dict,
                                ):

                                    clickables = (
                                        next_available_actions
                                        .get(
                                            "clickables",
                                            [],
                                        )
                                        or []
                                    )


                                lower_clickables = {
                                    str(x).strip().lower():
                                    str(x).strip()

                                    for x in clickables
                                }


                                recommended = None


                                if (
                                    "back to search"
                                    in lower_clickables
                                ):

                                    recommended = (
                                        "click["
                                        + lower_clickables[
                                            "back to search"
                                        ]
                                        + "]"
                                    )

                                elif (
                                    "< prev"
                                    in lower_clickables
                                ):

                                    recommended = (
                                        "click["
                                        + lower_clickables[
                                            "< prev"
                                        ]
                                        + "]"
                                    )


                                product_gate_after_action.update(
                                    {
                                        "decision":
                                            "EXHAUSTED",

                                        "ready_to_buy":
                                            False,

                                        "should_keep_candidate":
                                            False,

                                        "rule_passed":
                                            False,

                                        "recommended_action":
                                            recommended,

                                        "reason": (
                                            "All standard visible "
                                            "product evidence "
                                            "surfaces have been "
                                            "inspected, but required "
                                            "evidence is still "
                                            "missing: "
                                            + ", ".join(
                                                guard_missing
                                            )
                                        ),

                                        "exhaustion_guard":
                                            True,
                                    }
                                )


                                guard_decision = (
                                    "EXHAUSTED"
                                )


                                print()
                                print(
                                    "[ProductGate Guard]"
                                )

                                print(
                                    "  INSPECT -> EXHAUSTED"
                                )

                                print(
                                    "  still missing:",
                                    guard_missing,
                                )

                                print(
                                    "  recommended:",
                                    recommended,
                                )


                        # =====================================
                        # 2. BROAD CATEGORY EARLY-VETO GUARD
                        #
                        # WebShop taxonomy contains umbrella
                        # categories such as:
                        #
                        # accessories
                        # treatments
                        # care
                        # supplies
                        # tools
                        # kits
                        # equipment
                        #
                        # A specific product can belong to the
                        # umbrella without literally repeating
                        # its category label.
                        #
                        # Therefore:
                        #
                        # all explicit required constraints pass
                        # +
                        # the ONLY contradiction is product type
                        # +
                        # requested type is an umbrella category
                        #
                        # => do not hard reject at Early Product
                        #    Gate.
                        # =====================================

                        if guard_decision == "REJECT":

                            guard_missing = list(
                                product_gate_after_action.get(
                                    "missing_constraints",
                                    [],
                                )
                                or []
                            )

                            guard_contradicted = [
                                str(x).strip().lower()

                                for x in (
                                    product_gate_after_action
                                    .get(
                                        "contradicted_constraints",
                                        [],
                                    )
                                    or []
                                )
                            ]


                            schema = (
                                self.constraint_manager
                                .extract_from_instruction(
                                    instruction
                                )
                            )


                            product_type = (
                                schema.get(
                                    "product_type",
                                    {},
                                )
                                or {}
                            )


                            source_text = str(
                                product_type.get(
                                    "source_text",
                                    "",
                                )
                            ).strip()


                            canonical = str(
                                product_type.get(
                                    "canonical",
                                    "",
                                )
                            ).strip()


                            product_type_text = (
                                source_text
                                + " "
                                + canonical
                            ).lower()


                            broad_terms = (
                                "accessories",
                                "accessory",
                                "treatments",
                                "treatment",
                                "care",
                                "supplies",
                                "supply",
                                "tools",
                                "tool",
                                "kits",
                                "kit",
                                "equipment",
                            )


                            broad_category = any(
                                term
                                in product_type_text

                                for term in broad_terms
                            )


                            allowed_product_type_labels = {
                                "product_type",
                                "product type",
                                source_text.lower(),
                                canonical.lower(),
                            }


                            only_product_type_conflict = (
                                len(
                                    guard_contradicted
                                )
                                == 1

                                and (
                                    guard_contradicted[0]
                                    in
                                    allowed_product_type_labels
                                )
                            )


                            if (
                                broad_category
                                and not guard_missing
                                and
                                only_product_type_conflict
                            ):

                                product_gate_after_action = dict(
                                    product_gate_after_action
                                )


                                product_type_evidence = dict(
                                    product_gate_after_action.get(
                                        "product_type_evidence",
                                        {},
                                    )
                                    or {}
                                )


                                product_type_evidence.update(
                                    {
                                        "status":
                                            "SUPPORTED",

                                        "match_type":
                                            "broad_category_guard",

                                        "matched_by":
                                            (
                                                source_text
                                                or canonical
                                            ),

                                        "reason": (
                                            "Requested product "
                                            "type is an umbrella "
                                            "category. A semantic "
                                            "product-type mismatch "
                                            "alone is not allowed "
                                            "to hard-reject a "
                                            "candidate when all "
                                            "explicit required "
                                            "constraints are "
                                            "supported."
                                        ),
                                    }
                                )


                                product_gate_after_action.update(
                                    {
                                        "decision":
                                            "READY",

                                        "ready_to_buy":
                                            True,

                                        "should_keep_candidate":
                                            True,

                                        "rule_passed":
                                            True,

                                        "contradicted_constraints":
                                            [],

                                        "product_type_evidence":
                                            product_type_evidence,

                                        "recommended_action":
                                            None,

                                        "reason": (
                                            "All explicit required "
                                            "constraints are "
                                            "supported. Product "
                                            "type belongs to a "
                                            "broad requested "
                                            "umbrella category, so "
                                            "the early product-type "
                                            "veto is suppressed."
                                        ),

                                        "broad_category_guard":
                                            True,
                                    }
                                )


                                guard_decision = "READY"


                                print()
                                print(
                                    "[ProductGate Guard]"
                                )

                                print(
                                    "  broad product-type "
                                    "REJECT -> READY"
                                )

                                print(
                                    "  requested type:",
                                    source_text
                                    or canonical,
                                )


                        stats[
                            "product_gate_calls"
                        ] += 1


                        decision = str(
                            product_gate_after_action.get(
                                "decision",
                                "",
                            )
                        ).upper()


                        # -----------------------------------------
                        # FAIL OPEN
                        #
                        # PASS_THROUGH must not become persistent
                        # Product Gate feedback. Otherwise the
                        # next Base proposal may still be forced
                        # through reconsideration even though the
                        # gate explicitly said it cannot judge.
                        # -----------------------------------------

                        if decision == "PASS_THROUGH":

                            current_product_feedback = None

                            stats[
                                "product_gate_pass_through"
                            ] = (
                                stats.get(
                                    "product_gate_pass_through",
                                    0,
                                )
                                + 1
                            )

                        else:

                            current_product_feedback = (
                                product_gate_after_action
                            )


                        if decision == "EXHAUSTED":

                            stats[
                                "product_gate_exhausted"
                            ] = (
                                stats.get(
                                    "product_gate_exhausted",
                                    0,
                                )
                                + 1
                            )


                        if decision == "READY":

                            stats[
                                "product_gate_ready"
                            ] += 1

                            # Compatibility with old stats name.
                            stats[
                                "product_gate_accepts"
                            ] += 1


                        elif decision == "INSPECT":

                            stats[
                                "product_gate_inspects"
                            ] += 1


                        elif decision == "REJECT":

                            stats[
                                "product_gate_rejections"
                            ] += 1


                        elif decision == "EXHAUSTED":

                            stats[
                                "product_gate_exhausted"
                            ] = (
                                stats.get(
                                    "product_gate_exhausted",
                                    0,
                                )
                                + 1
                            )


                        print()
                        print(
                            "[ProductGate AFTER ACTION]"
                        )

                        print(
                            "  active product:",
                            active_product,
                        )

                        print(
                            "  accumulated pages:",
                            len(
                                candidate_evidence_pages
                            ),
                        )

                        print(
                            "  decision:",
                            product_gate_after_action.get(
                                "decision"
                            ),
                        )

                        print(
                            "  matched:",
                            product_gate_after_action.get(
                                "matched_constraints"
                            ),
                        )

                        print(
                            "  missing:",
                            product_gate_after_action.get(
                                "missing_constraints"
                            ),
                        )

                        print(
                            "  contradicted:",
                            product_gate_after_action.get(
                                "contradicted_constraints"
                            ),
                        )

                        print(
                            "  recommended:",
                            product_gate_after_action.get(
                                "recommended_action"
                            ),
                        )

                        print(
                            "  reason:",
                            product_gate_after_action.get(
                                "reason"
                            ),
                        )


                # =================================================
                # TRAJECTORY
                # =================================================

                trajectory.append({
                    "step": step,

                    "llm_output": llm_output,

                    "proposed_action": (
                        proposed_action
                    ),

                    "action": action,

                    "action_changed_by_gate": (
                        action_changed_by_gate
                    ),

                    "reward": reward,

                    "done": done,

                    "observation": str(
                        new_observation
                    ),

                    "available_actions": (
                        available_actions
                    ),

                    # Product Gate BEFORE action
                    "active_product_before_action": (
                        active_product
                        if product_feedback_before_action
                        else None
                    ),

                    "product_feedback_before_action": (
                        product_feedback_before_action
                    ),

                    "product_reconsider_llm_output": (
                        product_reconsider_llm_output
                    ),

                    "product_reconsider_reason": (
                        product_reconsider_reason
                    ),

                    "product_reconsider_changed": (
                        product_reconsider_changed
                    ),

                    # Candidate Gate
                    "candidate_gate_input_action": (
                        candidate_gate_input_action
                    ),

                    "candidate_gate": (
                        candidate_gate_result
                    ),

                    # Query Gate
                    "original_query": (
                        original_query
                    ),

                    "final_query": (
                        self._extract_search_query(
                            action
                        )
                    ),

                    "query_gate": (
                        query_gate_result
                    ),

                    "query_recovery_llm_output": (
                        query_recovery_llm_output
                    ),

                    "query_recovery_candidate_action": (
                        query_recovery_candidate_action
                    ),

                    "query_recovery_validation": (
                        query_recovery_validation
                    ),

                    "query_recovery_fallback_used": (
                        query_recovery_fallback_used
                    ),

                    "query_recovery_fallback_reason": (
                        query_recovery_fallback_reason
                    ),

                    # Product Gate AFTER action
                    "product_gate_trigger": (
                        product_gate_trigger
                    ),

                    "candidate_evidence_added": (
                        candidate_evidence_added
                    ),

                    "candidate_evidence_page_count": (
                        len(
                            candidate_evidence_pages
                        )
                        if active_product
                        else 0
                    ),

                    "product_gate_after_action": (
                        product_gate_after_action
                    ),
                })


                # =================================================
                # Baseline-shaped policy history
                # =================================================

                policy_history.append(
                    current_history_item
                )


                observation = (
                    new_observation
                )

                final_reward = reward


                if done:

                    # Keep baseline definition frozen.
                    success = (
                        reward > 0
                    )

                    break


        finally:

            env.close()


        # =====================================================
        # Result
        # =====================================================

        result = make_result(
            task_id=task_id,
            instruction=instruction,
            success=success,
            reward=final_reward,
            trajectory=trajectory,
        )


        if self.constraint_manager:

            stats[
                "constraint_extraction_calls"
            ] = (
                self.constraint_manager
                .extraction_calls
            )

            stats[
                "constraint_validation_calls"
            ] = (
                self.constraint_manager
                .validation_calls
            )

            stats[
                "constraint_schema_cache_hits"
            ] = (
                self.constraint_manager
                .schema_cache_hits
            )

            stats[
                "semantic_match_calls"
            ] = (
                self.constraint_manager
                .semantic_match_calls
            )

            stats[
                "product_type_semantic_calls"
            ] = (
                self.constraint_manager
                .product_type_semantic_calls
            )


        if self.use_product_gate:

            stats[
                "product_final_audit_calls"
            ] = (
                self.product_gate
                .final_audit_calls
            )


        result[
            "gate_stats"
        ] = stats


        result[
            "constraint_schema"
        ] = (
            constraint_schema
        )


        return result
