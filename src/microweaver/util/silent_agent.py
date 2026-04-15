from agentscope.agent import ReActAgent
from typing import override


class SilentReActAgent(ReActAgent):
    @override
    async def print(self, *args, **kwargs):
        pass