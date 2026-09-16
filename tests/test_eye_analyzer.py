import unittest

from modules.eye_analyzer import EyeAnalyzer


def make_landmarks(ear: float):
    landmarks = [(0.0, 0.0)] * 468
    for indices in (EyeAnalyzer.LEFT_EYE, EyeAnalyzer.RIGHT_EYE):
        outer, upper_one, upper_two, inner, lower_two, lower_one = indices
        landmarks[outer] = (0.0, 0.0)
        landmarks[inner] = (10.0, 0.0)
        height = ear * 5.0
        landmarks[upper_one] = (2.0, height)
        landmarks[lower_one] = (2.0, -height)
        landmarks[upper_two] = (8.0, height)
        landmarks[lower_two] = (8.0, -height)
    return landmarks


class EyeAnalyzerTests(unittest.TestCase):
    def test_open_and_closed_state(self):
        analyzer = EyeAnalyzer()
        self.assertEqual(
            analyzer.process_landmarks(make_landmarks(0.5), timestamp=0.0)["eye_state"],
            "OPEN",
        )
        result = analyzer.process_landmarks(make_landmarks(0.05), timestamp=0.1)
        self.assertEqual(result["eye_state"], "CLOSED")
        self.assertLess(result["average_ear"], analyzer.ear_threshold)

    def test_short_closure_counts_as_blink(self):
        analyzer = EyeAnalyzer(max_blink_duration=0.8)
        analyzer.process_landmarks(make_landmarks(0.5), timestamp=0.0)
        analyzer.process_landmarks(make_landmarks(0.05), timestamp=0.1)
        result = analyzer.process_landmarks(make_landmarks(0.5), timestamp=0.4)
        self.assertEqual(result["blink_count"], 1)
        self.assertFalse(result["is_sleepy"])

    def test_long_closure_is_sleepy_and_not_blink(self):
        analyzer = EyeAnalyzer(sleepy_duration=2.0)
        analyzer.process_landmarks(make_landmarks(0.05), timestamp=0.0)
        result = analyzer.process_landmarks(make_landmarks(0.05), timestamp=2.1)
        self.assertTrue(result["is_sleepy"])
        self.assertEqual(result["blink_count"], 0)


if __name__ == "__main__":
    unittest.main()