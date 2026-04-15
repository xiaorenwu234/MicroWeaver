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
    创建一个全新的 SilentReActAgent 实例，避免不同任务之间的状态污染
    """
    return SilentReActAgent(
        name="Evaluator",
        sys_prompt="""你是一个有用的软件工程师，我将要提供给你一个类的信息，我希望你给我返回这个类的描述。这个类的描述应该尽量简洁，准确，突出这个类在软件系统中的职责和作用。请基于以下几点来生成描述：
1. 类的主要职责和功能
2. 类在系统中的作用和位置
3. 类与其他类的关系
请注意：
- 描述应简洁明了，避免冗长和复杂的句子。
- 使用专业的术语，但确保描述易于理解。
- 如果类的职责不明确，可以使用“通用”、“辅助”等词汇来描述。
""",
        model=DashScopeChatModel(
            model_name="qwen3-max",
            api_key=os.environ["DASHSCOPE_API_KEY"],
        ),
        formatter=DashScopeChatFormatter(),
    )


class DescriptionResult(BaseModel):
    description: str = Field(description="类的描述")


async def description_generator(node, pbar=None) -> str:
    cohesion_agent = create_new_cohesion_agent()

    try:
        res = await cohesion_agent(
            Msg(
                "user",
                f"类名称：{node["qualifiedName"]}, 类的 javadoc：{node["javaDoc"]}, 类中包含的方法：{node["methods"]}\n请基于上述信息，提取并返回该类的描述，",
                "user"
            ),
            structured_model=DescriptionResult,
        )

        if not hasattr(res, "metadata") or res.metadata is None:
            print(f"\n警告：节点 {node.get('qualifiedName', '未知节点')} 无有效 metadata 数据，返回空结果")
            result = ""
        else:
            try:
                description_result = DescriptionResult.model_validate(res.metadata)
                result = description_result.description
            except ValidationError as e:
                print(f"\n警告：节点 {node.get('qualifiedName', '未知节点')} 结构化数据转换失败（模型返回格式不合法）：{e}")
                print(f"模型原始返回的 metadata：{json.dumps(res.metadata, indent=4, ensure_ascii=False)}")
                result = ""

    except Exception as e:
        print(f"\n错误：节点 {node.get('qualifiedName', '未知节点')} Agent 分析过程中发生异常：{e}")
        import traceback
        traceback.print_exc()
        result = ""

    finally:
        if pbar:
            pbar.update(1)
            pbar.set_description(f"处理: {node.get('qualifiedName', '未知')[:30]}...")

    return result


async def process_all_nodes_parallel(data: list, max_concurrent: int = 5) -> list:
    """
    并行处理所有节点，支持最大并发数控制和进度显示
    :param data: 原始节点数据
    :param max_concurrent: 最大并发数（建议根据 DashScope 配额调整，默认 5）
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    pbar = tqdm(
        total=len(data),
        desc="处理节点",
        unit="个",
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
    print(f"正在读取文件: {config.data_path}")
    with open(config.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"成功加载 {len(data)} 个节点，开始处理...")

    processed_data = asyncio.run(process_all_nodes_parallel(data, max_concurrent=4))

    with open(config.data_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"\n所有节点处理完成，结果已保存到 {config.data_path}")
    print(f"成功处理 {len(processed_data)} 个节点")
