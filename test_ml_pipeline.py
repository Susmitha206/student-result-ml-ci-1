
import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("student_results.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_dataset_not_empty(self):
        df = pd.read_csv("student_results.csv")
        self.assertGreater(len(df), 0)

    def test_model_can_be_loaded(self):
        model = joblib.load("student_result_model.pkl")
        self.assertIsNotNone(model)

    def test_metrics_have_accuracy(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
