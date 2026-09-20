import unittest

from modules.face_tracker import FaceTracker


def face(x, y=100, width=80, height=80):
    return {"x": x, "y": y, "width": width, "height": height}


class FaceTrackerTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            FaceTracker(max_distance=0)
        with self.assertRaises(ValueError):
            FaceTracker(max_missed_frames=-1)

    def test_new_faces_receive_sequential_ids(self):
        tracker = FaceTracker()

        tracked = tracker.update([face(100)])
        self.assertEqual(tracked[0]["id"], 1)
        self.assertTrue(tracked[0]["visible"])

        tracked = tracker.update([face(100), face(400)])
        self.assertEqual([item["id"] for item in tracked], [1, 2])

    def test_ids_stay_with_faces_when_detection_order_changes(self):
        tracker = FaceTracker()
        tracker.update([face(100), face(400)])

        tracked = tracker.update([face(400), face(100)])

        ids_by_x = {item["x"]: item["id"] for item in tracked}
        self.assertEqual(ids_by_x, {100: 1, 400: 2})

    def test_new_face_does_not_reuse_previous_id(self):
        tracker = FaceTracker(max_missed_frames=0)
        tracker.update([face(100)])
        tracker.update([])

        tracked = tracker.update([face(400)])

        self.assertEqual(tracked[0]["id"], 2)

    def test_missed_face_is_not_counted_as_visible(self):
        tracker = FaceTracker(max_missed_frames=1)
        tracker.update([face(100)])

        tracked = tracker.update([])

        self.assertEqual(len(tracked), 1)
        self.assertFalse(tracked[0]["visible"])


if __name__ == "__main__":
    unittest.main()