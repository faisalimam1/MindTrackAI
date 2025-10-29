"""
Accuracy Testing Framework for MindTrackAI
Tests algorithm accuracy against validated ground truth
"""

import json
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clinical_scoring import ClinicalScorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Single test case result"""
    test_id: str
    category: str

    # Ground truth
    ground_truth_depression: float
    ground_truth_anxiety: float
    ground_truth_confidence: float

    # Predicted values
    predicted_depression: float
    predicted_anxiety: float
    predicted_confidence: float

    # Errors
    depression_error: float
    anxiety_error: float
    confidence_error: float

    # Binary classification (correct/incorrect)
    depression_level_correct: bool
    anxiety_level_correct: bool
    confidence_level_correct: bool

    # Additional info
    ground_truth_level: str
    predicted_level: str

    def to_dict(self):
        return asdict(self)


@dataclass
class AccuracyMetrics:
    """Overall accuracy metrics"""
    total_tests: int

    # Regression metrics (for continuous scores)
    depression_mae: float  # Mean Absolute Error
    depression_rmse: float  # Root Mean Squared Error
    depression_r2: float  # R-squared

    anxiety_mae: float
    anxiety_rmse: float
    anxiety_r2: float

    confidence_mae: float
    confidence_rmse: float
    confidence_r2: float

    # Classification metrics (for severity levels)
    depression_level_accuracy: float  # % correct level
    anxiety_level_accuracy: float
    confidence_level_accuracy: float

    # Overall accuracy
    overall_accuracy: float  # Average of all accuracies

    # Per-category accuracy
    category_accuracies: Dict[str, float]

    def to_dict(self):
        return asdict(self)


class AccuracyTester:
    """
    Test mental health assessment accuracy
    """

    def __init__(self, test_scenarios_path: str = None):
        """
        Initialize accuracy tester

        Args:
            test_scenarios_path: Path to test_scenarios.json
        """
        self.scorer = ClinicalScorer()

        if test_scenarios_path is None:
            # Default path
            test_scenarios_path = os.path.join(
                os.path.dirname(__file__),
                'test_scenarios.json'
            )

        # Load test scenarios
        with open(test_scenarios_path, 'r') as f:
            data = json.load(f)
            self.test_scenarios = data['test_scenarios']

        logger.info(f"Loaded {len(self.test_scenarios)} test scenarios")

        self.test_results: List[TestResult] = []

    def run_single_test(self, scenario: Dict) -> TestResult:
        """
        Run a single test case

        Args:
            scenario: Test scenario dictionary

        Returns:
            TestResult with predictions vs ground truth
        """
        test_id = scenario['id']
        category = scenario['category']
        emotions = scenario['emotions']
        ground_truth = scenario['ground_truth']
        transcription = scenario.get('transcription', None)

        # Run predictions (with text sentiment analysis if available)
        depression_score = self.scorer.calculate_depression_score(
            emotions,
            transcription=transcription
        )
        depression_level = self.scorer.get_depression_level(depression_score)

        anxiety_score = self.scorer.calculate_anxiety_score(emotions)
        anxiety_level = self.scorer.get_anxiety_level(anxiety_score)

        # Calculate engagement and confidence
        engagement = self.scorer.calculate_engagement_level(emotions=emotions)
        confidence_level = self.scorer.get_confidence_level(engagement, anxiety_score)

        # Map confidence level to score (approximate)
        confidence_score_map = {
            'very_high': 0.80,
            'high': 0.65,
            'moderate': 0.45,
            'low': 0.25,
            'very_low': 0.10
        }
        confidence_score = confidence_score_map.get(confidence_level, 0.5)

        # Calculate errors
        depression_error = abs(depression_score - ground_truth['depression_score'])
        anxiety_error = abs(anxiety_score - ground_truth['anxiety_score'])
        confidence_error = abs(confidence_score - ground_truth['confidence_score'])

        # Check level correctness
        depression_level_correct = (depression_level == ground_truth['depression_level'])
        anxiety_level_correct = (anxiety_level == ground_truth['anxiety_level'])
        confidence_level_correct = (confidence_level == ground_truth['confidence_level'])

        result = TestResult(
            test_id=test_id,
            category=category,
            ground_truth_depression=ground_truth['depression_score'],
            ground_truth_anxiety=ground_truth['anxiety_score'],
            ground_truth_confidence=ground_truth['confidence_score'],
            predicted_depression=depression_score,
            predicted_anxiety=anxiety_score,
            predicted_confidence=confidence_score,
            depression_error=depression_error,
            anxiety_error=anxiety_error,
            confidence_error=confidence_error,
            depression_level_correct=depression_level_correct,
            anxiety_level_correct=anxiety_level_correct,
            confidence_level_correct=confidence_level_correct,
            ground_truth_level=ground_truth['depression_level'],
            predicted_level=depression_level
        )

        return result

    def run_all_tests(self) -> List[TestResult]:
        """
        Run all test scenarios

        Returns:
            List of test results
        """
        logger.info("Running all test scenarios...")
        self.test_results = []

        for scenario in self.test_scenarios:
            result = self.run_single_test(scenario)
            self.test_results.append(result)

            logger.info(
                f"Test {result.test_id}: "
                f"Depression {result.predicted_depression:.2f} vs {result.ground_truth_depression:.2f} "
                f"(error: {result.depression_error:.2f}) "
                f"Level: {result.predicted_level} vs {result.ground_truth_level} "
                f"({'✓' if result.depression_level_correct else '✗'})"
            )

        logger.info(f"Completed {len(self.test_results)} tests")
        return self.test_results

    def calculate_metrics(self) -> AccuracyMetrics:
        """
        Calculate overall accuracy metrics

        Returns:
            AccuracyMetrics object
        """
        if not self.test_results:
            raise ValueError("No test results available. Run tests first.")

        n = len(self.test_results)

        # Calculate MAE (Mean Absolute Error)
        depression_mae = sum(r.depression_error for r in self.test_results) / n
        anxiety_mae = sum(r.anxiety_error for r in self.test_results) / n
        confidence_mae = sum(r.confidence_error for r in self.test_results) / n

        # Calculate RMSE (Root Mean Squared Error)
        depression_rmse = (sum(r.depression_error ** 2 for r in self.test_results) / n) ** 0.5
        anxiety_rmse = (sum(r.anxiety_error ** 2 for r in self.test_results) / n) ** 0.5
        confidence_rmse = (sum(r.confidence_error ** 2 for r in self.test_results) / n) ** 0.5

        # Calculate R² (coefficient of determination)
        # R² = 1 - (SS_res / SS_tot)
        def calculate_r2(predictions, actuals):
            mean_actual = sum(actuals) / len(actuals)
            ss_tot = sum((a - mean_actual) ** 2 for a in actuals)
            ss_res = sum((a - p) ** 2 for a, p in zip(actuals, predictions))
            if ss_tot == 0:
                return 0.0
            return 1 - (ss_res / ss_tot)

        depression_predictions = [r.predicted_depression for r in self.test_results]
        depression_actuals = [r.ground_truth_depression for r in self.test_results]
        depression_r2 = calculate_r2(depression_predictions, depression_actuals)

        anxiety_predictions = [r.predicted_anxiety for r in self.test_results]
        anxiety_actuals = [r.ground_truth_anxiety for r in self.test_results]
        anxiety_r2 = calculate_r2(anxiety_predictions, anxiety_actuals)

        confidence_predictions = [r.predicted_confidence for r in self.test_results]
        confidence_actuals = [r.ground_truth_confidence for r in self.test_results]
        confidence_r2 = calculate_r2(confidence_predictions, confidence_actuals)

        # Calculate classification accuracy (% correct levels)
        depression_level_accuracy = sum(1 for r in self.test_results if r.depression_level_correct) / n
        anxiety_level_accuracy = sum(1 for r in self.test_results if r.anxiety_level_correct) / n
        confidence_level_accuracy = sum(1 for r in self.test_results if r.confidence_level_correct) / n

        # Calculate per-category accuracy
        categories = set(r.category for r in self.test_results)
        category_accuracies = {}
        for cat in categories:
            cat_results = [r for r in self.test_results if r.category == cat]
            if cat_results:
                cat_acc = sum(1 for r in cat_results if r.depression_level_correct) / len(cat_results)
                category_accuracies[cat] = cat_acc

        # Calculate overall accuracy
        # Weighted average: regression accuracy (1 - MAE) and classification accuracy
        regression_acc = (
            (1 - depression_mae) * 0.4 +
            (1 - anxiety_mae) * 0.3 +
            (1 - confidence_mae) * 0.3
        )

        classification_acc = (
            depression_level_accuracy * 0.5 +
            anxiety_level_accuracy * 0.3 +
            confidence_level_accuracy * 0.2
        )

        overall_accuracy = (regression_acc * 0.5 + classification_acc * 0.5)

        metrics = AccuracyMetrics(
            total_tests=n,
            depression_mae=depression_mae,
            depression_rmse=depression_rmse,
            depression_r2=depression_r2,
            anxiety_mae=anxiety_mae,
            anxiety_rmse=anxiety_rmse,
            anxiety_r2=anxiety_r2,
            confidence_mae=confidence_mae,
            confidence_rmse=confidence_rmse,
            confidence_r2=confidence_r2,
            depression_level_accuracy=depression_level_accuracy,
            anxiety_level_accuracy=anxiety_level_accuracy,
            confidence_level_accuracy=confidence_level_accuracy,
            overall_accuracy=overall_accuracy,
            category_accuracies=category_accuracies
        )

        return metrics

    def generate_report(self, output_path: str = None) -> str:
        """
        Generate detailed accuracy report

        Args:
            output_path: Path to save report JSON

        Returns:
            Path to saved report
        """
        if not self.test_results:
            raise ValueError("No test results available. Run tests first.")

        metrics = self.calculate_metrics()

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': metrics.total_tests,
                'overall_accuracy': f"{metrics.overall_accuracy * 100:.1f}%",
                'target_accuracy': "95.0%",
                'gap_to_target': f"{(0.95 - metrics.overall_accuracy) * 100:.1f}%"
            },
            'metrics': metrics.to_dict(),
            'test_results': [r.to_dict() for r in self.test_results],
            'recommendations': self._generate_recommendations(metrics)
        }

        if output_path is None:
            output_path = os.path.join(
                os.path.dirname(__file__),
                f'accuracy_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {output_path}")
        return output_path

    def _generate_recommendations(self, metrics: AccuracyMetrics) -> List[str]:
        """Generate recommendations for improving accuracy"""
        recommendations = []

        # Check depression accuracy
        if metrics.depression_level_accuracy < 0.95:
            recommendations.append(
                f"Depression classification accuracy is {metrics.depression_level_accuracy*100:.1f}%. "
                f"Optimize emotion weights in clinical_scoring.py"
            )

        if metrics.depression_mae > 0.10:
            recommendations.append(
                f"Depression MAE is {metrics.depression_mae:.3f}. "
                f"Consider adding contextual analysis from transcribed text"
            )

        # Check anxiety accuracy
        if metrics.anxiety_level_accuracy < 0.93:
            recommendations.append(
                f"Anxiety classification accuracy is {metrics.anxiety_level_accuracy*100:.1f}%. "
                f"Review anxiety calculation weights"
            )

        # Check confidence accuracy
        if metrics.confidence_level_accuracy < 0.90:
            recommendations.append(
                f"Confidence classification accuracy is {metrics.confidence_level_accuracy*100:.1f}%. "
                f"Implement confidence calibration using Platt scaling"
            )

        # Check category-specific issues
        for cat, acc in metrics.category_accuracies.items():
            if acc < 0.85:
                recommendations.append(
                    f"Low accuracy ({acc*100:.1f}%) for category '{cat}'. "
                    f"Review test cases and algorithm behavior for this category"
                )

        # R² analysis
        if metrics.depression_r2 < 0.85:
            recommendations.append(
                f"Depression R² is {metrics.depression_r2:.3f}. "
                f"Predictions may not correlate well with ground truth - check formula"
            )

        if not recommendations:
            recommendations.append("Excellent performance! All metrics above target.")

        return recommendations

    def print_summary(self):
        """Print accuracy summary to console"""
        if not self.test_results:
            raise ValueError("No test results available. Run tests first.")

        metrics = self.calculate_metrics()

        print("\n" + "="*70)
        print("MINDTRACKAI ACCURACY TEST RESULTS")
        print("="*70)
        print(f"\nTotal Tests: {metrics.total_tests}")
        print(f"Overall Accuracy: {metrics.overall_accuracy*100:.1f}%")
        print(f"Target: 95.0%")
        print(f"Gap: {(0.95 - metrics.overall_accuracy)*100:.1f}%")
        print("\n" + "-"*70)

        print("\nDEPRESSION METRICS:")
        print(f"  Classification Accuracy: {metrics.depression_level_accuracy*100:.1f}%")
        print(f"  MAE: {metrics.depression_mae:.3f}")
        print(f"  RMSE: {metrics.depression_rmse:.3f}")
        print(f"  R²: {metrics.depression_r2:.3f}")

        print("\nANXIETY METRICS:")
        print(f"  Classification Accuracy: {metrics.anxiety_level_accuracy*100:.1f}%")
        print(f"  MAE: {metrics.anxiety_mae:.3f}")
        print(f"  RMSE: {metrics.anxiety_rmse:.3f}")
        print(f"  R²: {metrics.anxiety_r2:.3f}")

        print("\nCONFIDENCE METRICS:")
        print(f"  Classification Accuracy: {metrics.confidence_level_accuracy*100:.1f}%")
        print(f"  MAE: {metrics.confidence_mae:.3f}")
        print(f"  RMSE: {metrics.confidence_rmse:.3f}")
        print(f"  R²: {metrics.confidence_r2:.3f}")

        print("\nPER-CATEGORY ACCURACY:")
        for cat, acc in sorted(metrics.category_accuracies.items(), key=lambda x: x[1]):
            status = "✓" if acc >= 0.90 else "✗"
            print(f"  {status} {cat:20s}: {acc*100:.1f}%")

        print("\n" + "="*70)
        print("RECOMMENDATIONS:")
        print("="*70)
        for i, rec in enumerate(self._generate_recommendations(metrics), 1):
            print(f"\n{i}. {rec}")

        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Run accuracy tests
    tester = AccuracyTester()
    tester.run_all_tests()
    tester.print_summary()
    report_path = tester.generate_report()
    print(f"\nDetailed report saved to: {report_path}")
