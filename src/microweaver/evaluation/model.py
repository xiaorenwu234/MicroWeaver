from typing import List

from pydantic import BaseModel, Field


class CompareResult(BaseModel):
    SC: List[int] = Field(description="语义内聚性评分，范围0-100分，分数越高表示内聚度越高")
    SCP: List[int] = Field(description="语义耦合性评分，范围0-100分，分数越低表示耦合度越低")
    SBC: List[int] = Field(description="服务边界清晰度评分，范围0-100分，分数越高表示服务边界越清晰")
    judge_result: List[str] = Field(description="当前微服务划分的建议")


class EvaluateResult:

    def __init__(self, SC: float, SCP: float, SBC: float, judge_result: str,
                 SSB: float, SII: float, ICP: float, Modularity: float):
        self.SC = SC
        self.SCP = SCP
        self.SBC = SBC
        self.judge_result = judge_result
        self.SSB = SSB
        self.SII = SII
        self.ICP = ICP
        self.Modularity = Modularity

    def __str__(self):
        return str({
            "SC": self.SC,
            "SCP": self.SCP,
            "SBC": self.SBC,
            "judge_result": self.judge_result,
            "SSB": self.SSB,
            "SII": self.SII,
            "ICP": self.ICP,
            "Modularity": self.Modularity,
        })

    def to_dict(self):
        return {
            "SC": self.SC,
            "SCP": self.SCP,
            "SBC": self.SBC,
            "judge_result": self.judge_result,
            "SSB": self.SSB,
            "SII": self.SII,
            "ICP": self.ICP,
            "Modularity": self.Modularity,
        }
