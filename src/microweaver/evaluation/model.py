from typing import List

from pydantic import BaseModel, Field


class CompareResult(BaseModel):
    SC: List[int] = Field(description="Semantic cohesion score, range 0-100, higher score indicates higher cohesion")
    SCP: List[int] = Field(description="Semantic coupling score, range 0-100, lower score indicates lower coupling")
    SBC: List[int] = Field(description="Service boundary clarity score, range 0-100, higher score indicates clearer service boundaries")
    judge_result: List[str] = Field(description="Suggestions for current microservice partitioning")


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
