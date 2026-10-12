from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    所有 Shopping Agent 的统一接口。

    后面 Baseline / Query-Gated / Product-Gated / Full Agent
    都实现同一个 run(task) 方法。
    """

    def __init__(self, name="agent"):
        self.name = name

    @abstractmethod
    def run(self, task):
        """
        输入一个 task，返回统一格式的 result。
        """
        raise NotImplementedError
