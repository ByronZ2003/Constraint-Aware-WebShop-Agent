import os
import re
import sys
import copy

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import web_agent_site.envs.web_agent_text_env as web_text_env
import web_agent_site.engine.engine as web_engine


class WebShopWrapper:

    def __init__(
        self,
        num_products=1000,
        attr_path=None,
        human_attr_path=None,
        search_index_path=None,
        **kwargs,
    ):
        """
        WebShop wrapper with optional custom dataset paths.

        Standard WebShop-small runs use the upstream defaults.

        Candidate-weight training can explicitly provide:
        - product file_path
        - attr_path
        - human_attr_path
        - search_index_path

        This keeps WebShop-small and the independent
        Candidate training subset isolated.
        """

        self.env_kwargs = {
            "observation_mode": "text",
            "num_products": num_products,
        }

        self.env_kwargs.update(kwargs)

        self.attr_path = (
            os.path.abspath(attr_path)
            if attr_path
            else None
        )

        self.human_attr_path = (
            os.path.abspath(human_attr_path)
            if human_attr_path
            else None
        )

        self.search_index_path = (
            os.path.abspath(search_index_path)
            if search_index_path
            else None
        )

        self.env = None


    def _create_env(self):
        """
        Construct WebShop with optional custom attribute
        files and Lucene search index.
        """

        # Standard WebShop-small:
        # use upstream defaults unchanged.
        if (
            not self.attr_path
            and not self.human_attr_path
            and not self.search_index_path
        ):
            return web_text_env.WebAgentTextEnv(
                **self.env_kwargs
            )

        original_load_products = (
            web_text_env.load_products
        )

        original_init_search = (
            web_text_env.init_search_engine
        )

        original_attr_path = (
            web_engine.DEFAULT_ATTR_PATH
        )

        original_human_attr_path = (
            web_engine.HUMAN_ATTR_PATH
        )


        def load_products_custom(
            filepath,
            num_products=None,
            human_goals=True,
        ):
            old_attr = (
                web_engine.DEFAULT_ATTR_PATH
            )

            old_human = (
                web_engine.HUMAN_ATTR_PATH
            )

            try:

                if self.attr_path:
                    web_engine.DEFAULT_ATTR_PATH = (
                        self.attr_path
                    )

                if self.human_attr_path:
                    web_engine.HUMAN_ATTR_PATH = (
                        self.human_attr_path
                    )

                return web_engine.load_products(
                    filepath=filepath,
                    num_products=num_products,
                    human_goals=human_goals,
                )

            finally:

                web_engine.DEFAULT_ATTR_PATH = (
                    old_attr
                )

                web_engine.HUMAN_ATTR_PATH = (
                    old_human
                )


        def init_search_custom(
            num_products=None,
        ):

            if self.search_index_path:

                if not os.path.isdir(
                    self.search_index_path
                ):
                    raise FileNotFoundError(
                        "Custom WebShop search index "
                        "not found: "
                        + self.search_index_path
                    )

                return web_engine.LuceneSearcher(
                    self.search_index_path
                )

            return web_engine.init_search_engine(
                num_products=num_products
            )


        web_text_env.load_products = (
            load_products_custom
        )

        web_text_env.init_search_engine = (
            init_search_custom
        )

        try:

            return web_text_env.WebAgentTextEnv(
                **self.env_kwargs
            )

        finally:

            web_text_env.load_products = (
                original_load_products
            )

            web_text_env.init_search_engine = (
                original_init_search
            )

            web_engine.DEFAULT_ATTR_PATH = (
                original_attr_path
            )

            web_engine.HUMAN_ATTR_PATH = (
                original_human_attr_path
            )


    @staticmethod
    def _normalize(text):
        if text is None:
            return ""

        text = str(text).strip()

        if text.lower().startswith("instruction:"):
            text = text[len("instruction:"):].strip()

        return " ".join(text.lower().split())


    @staticmethod
    def _extract_price_upper(instruction):
        """
        只有原始 instruction 明确写了价格限制时才使用。

        如果 instruction 没有价格限制，
        返回一个很大的数字，相当于不限制价格。
        """

        text = str(instruction).lower()

        patterns = [
            r"price\s+lower\s+than\s+\$?\s*(\d+(?:\.\d+)?)",
            r"under\s+\$?\s*(\d+(?:\.\d+)?)",
            r"less\s+than\s+\$?\s*(\d+(?:\.\d+)?)",
            r"below\s+\$?\s*(\d+(?:\.\d+)?)",
            r"lower\s+than\s+\$?\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return float(match.group(1))

        return 1000000.0


    def _find_matches(self, asin):
        """
        找出当前 ASIN 在 WebShop goals 中对应的所有 goal。
        """

        asin = str(asin).upper()

        matches = []

        for idx, goal in enumerate(
            self.env.server.goals
        ):

            goal_asin = str(
                goal.get("asin", "")
            ).upper()

            if goal_asin == asin:
                matches.append(
                    (idx, goal)
                )

        return matches


    def _choose_base_goal(
        self,
        matches,
        instruction,
    ):
        """
        同一个 ASIN 可能有多个 WebShop goal，
        例如不同 size / color。

        这里只选一个作为商品 metadata 模板。

        真正用于 evaluation 的：
        - instruction
        - attributes
        - options
        - price

        后面都会重新构造。
        """

        target = self._normalize(
            instruction
        )

        target_words = set(
            target.split()
        )

        best = None
        best_score = -1

        for idx, goal in matches:

            goal_text = self._normalize(
                goal.get(
                    "instruction_text",
                    ""
                )
            )

            goal_words = set(
                goal_text.split()
            )

            score = 0

            # 当前 task instruction 是完整 goal 的前半部分
            # 时给予最高优先级
            if target in goal_text:
                score += 1000

            # fallback：
            # 根据文本单词重合度判断
            score += len(
                target_words & goal_words
            )

            if score > best_score:

                best_score = score

                best = (
                    idx,
                    goal,
                )

        return best


    @staticmethod
    def _option_value_is_explicit(
        instruction,
        value,
    ):
        """
        判断一个 option value 是否真的明确出现在
        原始 instruction 中。

        重点：
        不能直接：

            value in instruction

        因为例如：

            value = "c"

        会错误命中：

            "cases"

        从而把：

            color = c

        错误当成用户要求。

        这里使用 word boundary 风格检查。
        """

        if value is None:
            return False

        instruction = (
            str(instruction)
            .strip()
            .lower()
        )

        value = (
            str(value)
            .strip()
            .lower()
        )

        if not value:
            return False

        # 对非常短的单字符 option 特别严格。
        #
        # c 只能匹配独立的 "c"
        # 不能匹配 cases / cruelty 等。
        pattern = (
            r"(?<!\w)"
            + re.escape(value)
            + r"(?!\w)"
        )

        return (
            re.search(
                pattern,
                instruction,
                flags=re.IGNORECASE,
            )
            is not None
        )


    def _get_explicit_options(
        self,
        task,
        matches,
    ):
        """
        只保留 evaluation task 明确要求的 options。

        优先级：

        1. task 自己有 instruction_options
        2. task 自己有 goal_options
        3. 从 instruction 文本里判断 option value
           是否明确出现
        4. 否则 {}

        不允许 WebShop 自动生成的隐藏
        size / color / scent 条件偷偷进入 reward。
        """

        explicit = (
            task.get(
                "instruction_options"
            )
            or task.get(
                "goal_options"
            )
            or {}
        )

        # 如果 task 文件自己明确保存了 options，
        # 直接采用。
        if (
            isinstance(explicit, dict)
            and explicit
        ):

            return copy.deepcopy(
                explicit
            )


        instruction = self._normalize(
            task.get(
                "instruction",
                ""
            )
        )

        best = {}


        for _, candidate_goal in matches:

            candidate_options = (
                candidate_goal.get(
                    "goal_options",
                    {}
                )
            )

            if not isinstance(
                candidate_options,
                dict
            ):
                continue


            found = {}


            for (
                key,
                value,
            ) in candidate_options.items():

                if self._option_value_is_explicit(
                    instruction,
                    value,
                ):

                    found[key] = value


            # 如果多个 goal 都有可能，
            # 选择显式匹配 option 数最多的。
            if len(found) > len(best):

                best = found


        return copy.deepcopy(best)


    def reset(
        self,
        task=None,
        instruction=None,
    ):

        if task is None:

            raise ValueError(
                "WebShopWrapper.reset() "
                "requires task"
            )


        # ==========================================
        # Evaluation task
        # ==========================================

        asin = str(
            task["task_id"]
        ).upper()


        if instruction is None:

            instruction = (
                task["instruction"]
            )


        instruction = str(
            instruction
        ).strip()


        # ==========================================
        # 1. 创建 WebShop environment
        # ==========================================

        self.env = self._create_env()


        # ==========================================
        # 2. 找当前 ASIN 对应的所有 WebShop goals
        # ==========================================

        matches = self._find_matches(
            asin
        )


        if not matches:

            raise RuntimeError(
                "Cannot find WebShop goal.\n"
                f"ASIN: {asin}\n"
                f"Instruction: {instruction}"
            )


        # ==========================================
        # 3. 选一个 base goal
        # ==========================================

        (
            goal_idx,
            base_goal,
        ) = self._choose_base_goal(
            matches,
            instruction,
        )


        goal = copy.deepcopy(
            base_goal
        )


        # ==========================================
        # 4. 构造真正 evaluation 使用的 goal
        # ==========================================

        # instruction 必须严格等于
        # evaluation task instruction
        goal["instruction_text"] = (
            instruction
        )


        # ==========================================
        # Attributes
        # ==========================================

        if (
            task.get(
                "instruction_attributes"
            )
            is not None
        ):

            goal["attributes"] = (
                copy.deepcopy(
                    task.get(
                        "instruction_attributes"
                    )
                    or []
                )
            )

        elif (
            task.get("attributes")
            is not None
        ):

            goal["attributes"] = (
                copy.deepcopy(
                    task.get(
                        "attributes"
                    )
                    or []
                )
            )


        # ==========================================
        # Options
        #
        # 只允许 task 明确要求的 option。
        # ==========================================

        goal["goal_options"] = (
            self._get_explicit_options(
                task,
                matches,
            )
        )


        # ==========================================
        # Price
        #
        # 原 task 不写 price
        # => 不增加 WebShop 随机 price constraint
        # ==========================================

        if (
            task.get("price_upper")
            is not None
        ):

            goal["price_upper"] = float(
                task["price_upper"]
            )

        else:

            goal["price_upper"] = (
                self._extract_price_upper(
                    instruction
                )
            )


        # ==========================================
        # 5. 创建合法 WebShop session
        #
        # WebAgentTextEnv.reset() 参数叫 session。
        #
        # 当 session 是 int：
        #
        # session_int = session
        #
        # 最终 SimServer 会：
        #
        # goal = self.goals[session_int]
        #
        # 所以这里传 goal_idx。
        # ==========================================

        obs, _ = self.env.reset(
            session=goal_idx
        )


        # ==========================================
        # 6. 替换 session 中真正的 evaluation goal
        # ==========================================

        session = (
            self.env.server
            .user_sessions[
                self.env.session
            ]
        )


        session["goal"] = goal


        # ==========================================
        # 7. 重新生成首页
        #
        # 注意：
        #
        # 必须调用 receive()
        #
        # 不能直接 server.index()
        #
        # 因为 receive() 内部负责 Flask context。
        # ==========================================

        (
            html,
            url,
            status,
        ) = self.env.server.receive(
            self.env.session,
            self.env.browser.current_url,
        )


        # ==========================================
        # 8. 同步 browser
        # ==========================================

        self.env.browser.page_source = (
            html
        )

        self.env.browser.current_url = (
            url
        )

        self.env.browser.session_id = (
            self.env.session
        )


        # ==========================================
        # 9. 同步 env instruction
        # ==========================================

        self.env.instruction_text = (
            goal["instruction_text"]
        )


        # ==========================================
        # 10. 获得新的 observation
        # ==========================================

        obs = self.env.observation


        # ==========================================
        # 11. Sanity checks
        # ==========================================

        actual_goal = (
            self.env.server
            .user_sessions[
                self.env.session
            ]["goal"]
        )


        actual_asin = str(
            actual_goal.get(
                "asin",
                ""
            )
        ).upper()


        if actual_asin != asin:

            raise RuntimeError(
                "ASIN / goal desync!\n"
                f"Expected: {asin}\n"
                f"Actual: {actual_asin}"
            )


        observed_instruction = (
            self.env.get_instruction_text()
        )


        if (
            self._normalize(
                observed_instruction
            )
            !=
            self._normalize(
                goal["instruction_text"]
            )
        ):

            raise RuntimeError(
                "Observation / goal desync!\n"
                f"Observed: "
                f"{observed_instruction}\n"
                f"Goal: "
                f"{goal['instruction_text']}"
            )


        # ==========================================
        # 12. Debug information
        # ==========================================

        print(
            f"[WebShop] "
            f"asin={asin} "
            f"base_goal_index={goal_idx}"
        )


        print(
            f"[WebShop] "
            f"instruction="
            f"{goal['instruction_text']}"
        )


        print(
            f"[WebShop] "
            f"attributes="
            f"{goal.get('attributes', [])}"
        )


        print(
            f"[WebShop] "
            f"options="
            f"{goal.get('goal_options', {})}"
        )


        price_upper = goal.get(
            "price_upper",
            1000000.0,
        )


        if price_upper >= 1000000:

            print(
                "[WebShop] "
                "price_upper=NONE"
            )

        else:

            print(
                "[WebShop] "
                f"price_upper={price_upper}"
            )


        return obs


    def get_available_actions(self):

        return (
            self.env
            .get_available_actions()
        )


    def get_instruction(self):

        return (
            self.env
            .get_instruction_text()
        )


    def step(self, action):

        return self.env.step(
            action
        )


    def close(self):

        if self.env is not None:

            try:
                self.env.close()

            except Exception:
                pass