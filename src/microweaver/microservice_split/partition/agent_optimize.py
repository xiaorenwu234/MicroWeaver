import json
import os
from typing import List, Tuple, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel

from microweaver.util.silent_agent import SilentReActAgent

optimize_agent = SilentReActAgent(
    name="Optimizer",
    sys_prompt="""You are a helpful software engineer. I will provide you with a microservice partitioning result and expert optimization suggestions. Please optimize the partitioning result to ensure the following requirements are met:
1. Are there any semantically unreasonable partitions? Should any functions be split out or merged?
2. Please give clear must-link or cannot-link suggestions.
3. must-link suggestions indicate nodes that must be placed in the same service; cannot-link suggestions indicate two nodes that cannot be placed in the same service.

Note:
must-link is a list of lists of node names, indicating these nodes must be placed in the same service;
cannot-link is a list of tuples of node names, ensuring clarity.
Don't be influenced only by test functions like Ping; consider business functions and responsibilities comprehensively.
Try to separate data processing logic of different businesses to reduce coupling between services.
Try to place different businesses in different services to improve service cohesion and reduce coupling.
""",
    model=DashScopeChatModel(
        model_name="qwen3-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    ),
    formatter=DashScopeChatFormatter(),
)

analyze_agent = SilentReActAgent(
name="Analyzer",
    sys_prompt="""You are a helpful software engineer. I will provide you with a microservice partitioning result. Please analyze the partitioning result to ensure the following requirements are met:
1. Is the partitioning result reasonable? Does it need further optimization?
2. If optimization is needed, please give clear optimization suggestions.

Note:
1. Coupling between services should be as low as possible; cohesion within services should be as high as possible.
2. Try to separate data processing logic of different businesses to reduce coupling between services.
3. Try to place different businesses in different services to improve service cohesion and reduce coupling.
4. Services should be semantically reasonable; related functions should be partitioned into the same service.
5. Don't be influenced only by test functions like Ping; consider business functions and responsibilities comprehensively.
""",
    model=DashScopeChatModel(
        model_name="qwen3-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        temperature=0.1
    ),
    formatter=DashScopeChatFormatter(),
)

class AnalyzeResult(BaseModel):
    """Data model for Agent analysis results"""
    needs_optimization: bool = Field(
        default=True,
        description="Whether the current partitioning result needs optimization")
    suggestions: Optional[str] = Field(
        default=None,
        description="Agent's analysis suggestion explanation")


class OptimizeResult(BaseModel):
    """Data model for Agent optimization results"""
    must_links: List[List[str]] = Field(
        default_factory=list,
        description="List of node pairs that must be placed in the same service, each element is a tuple representing node name or index")
    cannot_link: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="List of node pairs that cannot be placed in the same service, each element is a tuple representing node name or index")
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning process and suggestion explanation")

async def agent_analyze(partitions: Dict, safe_upper: int) -> Optional[AnalyzeResult]:
    """
    Use Agent to analyze microservice partitioning results.

    Args:
        partitions: Partition dictionary, format is {service_id: [node_ids]}
        safe_upper: Maximum number of nodes per service

    Returns:
        AnalyzeResult object, containing whether optimization is needed and suggestion explanation;
        Returns None if analysis fails or returns None
    """
    try:
        print("Starting to call LLM to analyze microservice partitioning results, waiting for response...")
        res = await analyze_agent(
            Msg(
                "user",
                f"Microservice partitioning result: {json.dumps(partitions, ensure_ascii=False)}. Maximum node limit per service: {safe_upper}"
                "Please analyze this partitioning and tell me if optimization is needed. If optimization is needed, please give suggestions.",
                "user"
            ),
            structured_model=AnalyzeResult,
        )
        print("Model has fully returned, starting to process results...")

        if not hasattr(res, "metadata") or res.metadata is None:
            print("Warning: No valid metadata in model response, returning empty result")
            return AnalyzeResult(needs_optimization=True, suggestions=None)

        try:
            analyze_result = AnalyzeResult.model_validate(res.metadata)
            print(f"Agent analysis completed: needs_optimization={analyze_result.needs_optimization}")
            if analyze_result.suggestions:
                print(f"  - Suggestion: {analyze_result.suggestions[:100]}...")
            return analyze_result
        except ValidationError as e:
            print(f"Warning: Structured data conversion failed (invalid model response format): {e}")
            print(f"Original metadata returned by model: {json.dumps(res.metadata, indent=4, ensure_ascii=False)}")
            # Return empty result instead of throwing exception, allowing analysis to continue
            return AnalyzeResult(needs_optimization=True, suggestions=None)

    except Exception as e:
        print(f"Error: Exception occurred during Agent analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


async def agent_optimize(partitions: Dict, advice: str) -> Optional[OptimizeResult]:

    """
    Use Agent to optimize microservice partitioning results.
    
    Args:
        partitions: Partition dictionary, format is {service_id: [node_ids]}
        advice: Optimization suggestion explanation obtained from Agent analysis
    
    Returns:
        OptimizeResult object, containing must_links and cannot_link suggestions;
        Returns None if optimization fails or returns None
    """
    try:
        print("Starting to call LLM to optimize microservice partitioning results, waiting for response...")
        if partitions is None or len(partitions) == 0:
            res = await optimize_agent(
                Msg(
                    "user",
                    advice,
                    "user"
                ),
                structured_model=OptimizeResult
            )
        else:
            res = await optimize_agent(
                Msg(
                    "user",
                    f"Microservice partitioning result: {json.dumps(partitions, ensure_ascii=False)}. Expert opinion: {advice}."
                    "Please analyze this partitioning and give your must-link and cannot-link results.",
                    "user"
                ),
                structured_model=OptimizeResult,
            )
        print("Model has fully returned, starting to process results...")

        if not hasattr(res, "metadata") or res.metadata is None:
            print("Warning: No valid metadata in model response, returning empty result")
            return OptimizeResult(must_links=[], cannot_link=[])

        try:
            optimize_result = OptimizeResult.model_validate(res.metadata)
            print(f"Agent optimization completed:")
            print(f"  - must_links: {len(optimize_result.must_links)} items")
            print(f"  - cannot_link: {len(optimize_result.cannot_link)} items")
            if optimize_result.reasoning:
                print(f"  - Reasoning: {optimize_result.reasoning[:100]}...")
            return optimize_result
        except ValidationError as e:
            print(f"Warning: Structured data conversion failed (invalid model response format): {e}")
            print(f"Original metadata returned by model: {json.dumps(res.metadata, indent=4, ensure_ascii=False)}")
            return OptimizeResult(must_links=[], cannot_link=[])
    
    except Exception as e:
        print(f"Error: Exception occurred during Agent optimization: {e}")
        import traceback
        traceback.print_exc()
        return None
