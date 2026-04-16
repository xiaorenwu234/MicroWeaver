import os
import json

import asyncio
from agentscope.formatter import DashScopeChatFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm
from microweaver.input_builder.config import InputConfig

from microweaver.util.silent_agent import SilentReActAgent


def create_new_cohesion_agent() -> SilentReActAgent:
    """
    Create a brand new SilentReActAgent instance to avoid state pollution between different tasks
    """
    return SilentReActAgent(
        name="Evaluator",
        sys_prompt="""You are a helpful software engineer. I will provide you with information about a class, and I want you to return a description of this class. The description should be concise, accurate, and highlight the responsibilities and role of this class in the software system. Please generate the description based on the following points:
1. Main responsibilities and functions of the class
2. Role and position of the class in the system
3. Relationships between this class and other classes
Please note:
- The description should be concise and clear, avoiding lengthy and complex sentences.
- Use professional terminology, but ensure the description is easy to understand.
- If the class's responsibilities are unclear, you can use words like "general", "auxiliary", etc. to describe it.
""",
        model=DashScopeChatModel(
            model_name="qwen3-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
        ),
        formatter=DashScopeChatFormatter(),
    )


class DescriptionResult(BaseModel):
    description: str = Field(description="Class description")


async def description_generator(node, pbar=None) -> str:
    cohesion_agent = create_new_cohesion_agent()

    try:
        res = await cohesion_agent(
            Msg(
                "user",
                f"Class name: {node["qualifiedName"]}, Class javadoc: {node["javaDoc"]}, Methods in class: {node["methods"]}\nPlease extract and return a description of this class based on the above information,",
                "user"
            ),
            structured_model=DescriptionResult,
        )

        if not hasattr(res, "metadata") or res.metadata is None:
            print(f"\nWarning: Node {node.get('qualifiedName', 'unknown node')} has no valid metadata, returning empty result")
            result = ""
        else:
            try:
                description_result = DescriptionResult.model_validate(res.metadata)
                result = description_result.description
            except ValidationError as e:
                print(f"\nWarning: Node {node.get('qualifiedName', 'unknown node')} structured data conversion failed (invalid model response format): {e}")
                print(f"Raw metadata returned by model: {json.dumps(res.metadata, indent=4, ensure_ascii=False)}")
                result = ""

    except Exception as e:
        print(f"\nError: Node {node.get('qualifiedName', 'unknown node')} Agent analysis encountered an exception: {e}")
        import traceback
        traceback.print_exc()
        result = ""

    finally:
        if pbar:
            pbar.update(1)
            pbar.set_description(f"Processing: {node.get('qualifiedName', 'unknown')[:30]}...")

    return result


async def process_all_nodes_parallel(data: list, max_concurrent: int = 5) -> list:
    """
    Process all nodes in parallel, supporting max concurrency control and progress display
    :param data: Raw node data
    :param max_concurrent: Maximum concurrency (adjust based on DashScope quota, default 5)
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    pbar = tqdm(
        total=len(data),
        desc="Processing nodes",
        unit="nodes",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    async def bounded_description_generator(node):
        async with semaphore:
            return await description_generator(node, pbar)

    try:
        tasks = [bounded_description_generator(node) for node in data]

        descriptions = await asyncio.gather(*tasks, return_exceptions=False)

    finally:
        pbar.close()

    for node, description in zip(data, descriptions):
        node["description"] = description

    return data


def main(config: InputConfig):
    print(f"Reading file: {config.data_path}")
    with open(config.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Successfully loaded {len(data)} nodes, starting processing...")

    processed_data = asyncio.run(process_all_nodes_parallel(data, max_concurrent=4))

    with open(config.data_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"\nAll nodes processed, results saved to {config.data_path}")
    print(f"Successfully processed {len(processed_data)} nodes")
