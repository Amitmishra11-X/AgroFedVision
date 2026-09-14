from datetime import datetime


class ExplainableFusion:

    def __init__(self):

        self.modules = []

        # SHAP explanation for structured
        # sensor + UAV inputs
        self.shap_explanation = None

    # ======================================================
    # Add Explainability Module
    # ======================================================

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

    # ======================================================
    # Add SHAP Explanation
    # ======================================================

    def add_shap_explanation(
        self,
        shap_result
    ):

        self.shap_explanation = shap_result

    # ======================================================
    # Generate Final Explainability Report
    # ======================================================

    def generate(
        self,
        overall_status,
        risk_level,
        health_score
    ):

        report = {

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "overall_status":
                overall_status,

            "risk_level":
                risk_level,

            "health_score":
                health_score,

            "modules":
                self.modules,

            "shap_explanation":
                self.shap_explanation

        }

        return report