import json
import os
from typing import List, Dict

from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit
from pydantic import ValidationError

from microweaver.evaluation.model import CompareResult
from microweaver.util.silent_agent import SilentReActAgent


def create_agent(agent_name: str, system_prompt: str, toolkit: Toolkit) -> SilentReActAgent:
    """Create agent object based on given information."""
    if system_prompt is None or system_prompt.strip() == "":
        system_prompt = f"You are {agent_name}, a professional software engineer."

    return SilentReActAgent(
        name=agent_name,
        sys_prompt=system_prompt,
        model=DashScopeChatModel(
            model_name="qwen3-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
        ),
        toolkit=toolkit,
        formatter=DashScopeMultiAgentFormatter(),
    )


async def run_evaluate_agent(splits: List[Dict]) -> CompareResult | None:
    evaluate_agent = create_agent("evaluate_agent",
                                  "You are a software engineer. I will give you several different microservice partitioning results for the same monolithic system. I hope you can compare and analyze the rationality of these microservice partitioning results, and score each partitioning result from three aspects: semantic consistency, semantic coupling, and service boundary clarity, and give your reasons.",
                                  toolkit=Toolkit())

    prompt = f"Please compare and analyze the following microservice partitioning results:"
    for idx, split in enumerate(splits):
        prompt += f"\nPartition result {idx + 1}: {json.dumps(split, indent=4, ensure_ascii=False)}"

    res = await evaluate_agent(Msg(
        "user",
        prompt,
        "user",
    ),
        structured_model=CompareResult
    )

    if not hasattr(res, "metadata") or res.metadata is None:
        print("Warning: No valid metadata in model response, returning empty result")
        return None
    try:
        analyze_result = CompareResult.model_validate(res.metadata)
        return analyze_result

    except ValidationError as e:
        print(f"Warning: Structured data conversion failed (invalid model response format): {e}")
        print(f"Original metadata returned by model: {json.dumps(res.metadata, indent=4, ensure_ascii=False)}")
        return None
