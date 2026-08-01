"""
============================================================
AgroFedVision Explainable AI Engine
Author : Amit Mishra
============================================================
"""

from datetime import datetime


class ExplainableFusion:

    def __init__(self):
        self.modules = []

    def add_module(
        self,
        name,
        score=None,
        confidence=None,
        summary="",
        evidence=None,
        recommendation=None,
    ):

        self.modules.append({
            "module": name,
            "score": score,
            "confidence": confidence,
            "summary": summary,
            "evidence": evidence or {},
            "recommendation": recommendation
        })

    def generate(
        self,
        overall_status,
        risk_level,
        health_score
    ):

        report = {

            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "overall_status": overall_status,

            "risk_level": risk_level,

            "health_score": health_score,

            "modules": self.modules
        }

        return report