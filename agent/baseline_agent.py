import json
import os
import re

import openai

from agent.base_agent import BaseAgent
from evaluation.result_schema import make_result
from webshop_wrapper.env import WebShopWrapper


class BaselineAgent(BaseAgent):

    def __init__(
        self,
        num_products=1000,
        max_steps=20,
        webshop_file_path=None,
        webshop_attr_path=None,
        webshop_human_attr_path=None,
        webshop_index_path=None,
    ):
        super().__init__("baseline")

        self.num_products = num_products
        self.max_steps = max_steps

        self.webshop_file_path = (
            webshop_file_path
        )

        self.webshop_attr_path = (
            webshop_attr_path
        )

        self.webshop_human_attr_path = (
            webshop_human_attr_path
        )

        self.webshop_index_path = (
            webshop_index_path
        )

        api_key = (
            os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )

        if not api_key:
            raise RuntimeError(
                "Set DEEPSEEK_API_KEY or OPENAI_API_KEY."
            )

        openai.api_key = api_key

        openai.api_base = (
            os.environ.get("DEEPSEEK_API_BASE")
            or os.environ.get("OPENAI_API_BASE")
            or "https://api.deepseek.com"
        )

    def ask_deepseek(self, instruction, history, observation, available_actions):

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

Choose exactly ONE next action.

Allowed formats:
search[query]
click[item]

Important:
- Do not repeat an action that has already been taken.
- Use the current observation to decide the next action.
- Search only when necessary.
- After finding a relevant product, inspect it and verify the required attributes.
- If the current product satisfies the user's instruction and click[buy now] is available, choose click[buy now] immediately.
- Do not continue searching after finding a product that satisfies all constraints.
- The task is completed only after selecting click[buy now].
- Output ONLY the action, with no explanation.
"""

        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=128,
        )

        return response["choices"][0]["message"]["content"].strip()

    def parse_action(self, text):
        match = re.search(
            r"(search|click)\[(.*?)\]",
            text,
            re.IGNORECASE,
        )

        if not match:
            return None

        action_type = match.group(1).lower()
        value = match.group(2).strip()

        return f"{action_type}[{value}]"

    def run(self, task):

        instruction = task["instruction"]
        task_id = task["task_id"]

        env_kwargs = {}

        if self.webshop_file_path:
            env_kwargs["file_path"] = (
                self.webshop_file_path
            )

        if self.webshop_attr_path:
            env_kwargs["attr_path"] = (
                self.webshop_attr_path
            )

        if self.webshop_human_attr_path:
            env_kwargs["human_attr_path"] = (
                self.webshop_human_attr_path
            )

        if self.webshop_index_path:
            env_kwargs["search_index_path"] = (
                self.webshop_index_path
            )

        env = WebShopWrapper(
            num_products=self.num_products,
            **env_kwargs,
        )

        observation = env.reset(task)

        history = []
        trajectory = []

        final_reward = 0.0
        success = False

        try:

            for step in range(1, self.max_steps + 1):

                available_actions = env.get_available_actions()

                llm_output = self.ask_deepseek(
                    instruction,
                    history,
                    observation,
                    available_actions,
                )

                action = self.parse_action(llm_output)

                if action is None:
                    print(
                        f"Step {step}: invalid LLM output: "
                        f"{llm_output}"
                    )
                    break

                print(
                    f"Step {step}: {action}"
                )

                observation, reward, done, info = env.step(action)

                trajectory.append({
                    "step": step,
                    "llm_output": llm_output,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "observation": str(observation),
                    "available_actions": available_actions,
                })

                history.append({
                    "step": step,
                    "action": action,
                    "reward": reward,
                    "done": done,
                })

                final_reward = reward

                if done:
                    success = reward > 0
                    break

        finally:
            env.close()

        return make_result(
            task_id=task_id,
            instruction=instruction,
            success=success,
            reward=final_reward,
            trajectory=trajectory,
        )
