"""
Inference script for credit scorecard
Loads trained model and applies cutoff to score new applications
"""

import json
import logging

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditScorecard:
    """Credit scorecard for application scoring"""

    def __init__(self, model_dir=None):
        """Load trained pipeline and metadata"""
        if model_dir is None:
            from pathlib import Path

            # Look for model_output in parent directory (local/)
            base_dir = Path(__file__).parent.parent.parent
            model_dir = base_dir / "model_output"
        else:
            from pathlib import Path

            model_dir = Path(model_dir)

        self.pipeline = joblib.load(str(model_dir / "scorecard_pipeline.joblib"))

        with open(model_dir / "scorecard_metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.cutoff = self.metadata["cutoff"]
        self.feature_names = self.metadata["feature_names"]

        logger.info("✓ Loaded scorecard model")
        logger.info(f"  Cutoff: {self.cutoff}")
        logger.info(f"  Gini: {self.metadata['gini']:.4f}")
        logger.info(f"  Features: {self.feature_names}")

    def calculate_score(self, X):
        """
        Calculate credit score for applications

        Args:
            X: DataFrame with features

        Returns:
            Array of credit scores
        """
        # Transform to WOE
        X_woe = self.pipeline.named_steps["woe"].transform(X)

        # Get log-odds from logistic regression
        logistic = self.pipeline.named_steps["logistic"]
        log_odds = logistic.decision_function(X_woe)

        # Convert to credit scores (300-850 scale)
        # Using scorecard formula with reversed sign
        factor = self.metadata["factor"]
        base_score = self.metadata["base_score"]
        offset = base_score - factor * (-logistic.intercept_[0])

        return offset - factor * log_odds

    def predict(self, X):
        """
        Score applications and make approve/decline decisions

        Args:
            X: DataFrame with features

        Returns:
            DataFrame with scores and decisions
        """
        scores = self.calculate_score(X)
        decisions = np.where(scores >= self.cutoff, "APPROVED", "DECLINED")

        return pd.DataFrame({"score": scores, "decision": decisions})

    def predict_single(self, application_data):
        """
        Score a single application (from CSV string or dict)

        Args:
            application_data: CSV string or dict with feature values

        Returns:
            dict with score and decision
        """
        if isinstance(application_data, str):
            # Parse CSV string
            values = [float(x) for x in application_data.split(",")]
            application_dict = dict(zip(self.feature_names, values))
        else:
            application_dict = application_data

        # Create DataFrame
        X = pd.DataFrame([application_dict])

        # Score
        score = self.calculate_score(X)[0]
        decision = "APPROVED" if score >= self.cutoff else "DECLINED"

        return {"score": float(score), "decision": decision, "cutoff": self.cutoff}


def test_inference():
    """Test inference on sample applications"""
    logger.info("=" * 80)
    logger.info("Testing Credit Scorecard Inference")
    logger.info("=" * 80)

    # Load scorecard
    scorecard = CreditScorecard()

    # Test applications
    test_apps = [
        {
            "name": "Good applicant",
            "data": {
                "Mortgage": 300000,
                "Balance": 500,
                "Amount Past Due": 0,
                "Delinquency": 0,
                "Inquiry": 0,
                "Open Trade": 0,
                "Utilization": 0.2,
                "Demographic": 1,
            },
        },
        {
            "name": "Risky applicant",
            "data": {
                "Mortgage": 150000,
                "Balance": 2500,
                "Amount Past Due": 500,
                "Delinquency": 2,
                "Inquiry": 1,
                "Open Trade": 1,
                "Utilization": 0.9,
                "Demographic": 0,
            },
        },
        {
            "name": "Borderline applicant",
            "data": {
                "Mortgage": 250000,
                "Balance": 1200,
                "Amount Past Due": 0,
                "Delinquency": 0,
                "Inquiry": 0,
                "Open Trade": 0,
                "Utilization": 0.5,
                "Demographic": 1,
            },
        },
    ]

    logger.info(f"\nTesting {len(test_apps)} applications:")
    logger.info("=" * 80)

    for app in test_apps:
        result = scorecard.predict_single(app["data"])

        emoji = "✅" if result["decision"] == "APPROVED" else "❌"
        logger.info(f"\n{emoji} {app['name']}")
        logger.info(f"   Score: {result['score']:.0f}")
        logger.info(f"   Decision: {result['decision']}")
        logger.info(f"   Cutoff: {result['cutoff']}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Inference test complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    test_inference()
    test_inference()
