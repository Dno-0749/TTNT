import math


class FaceTracker:
    """Giữ ID khuôn mặt ổn định giữa các frame camera."""

    def __init__(self, max_distance=140, max_missed_frames=4):
        if max_distance <= 0:
            raise ValueError("max_distance must be positive")
        if max_missed_frames < 0:
            raise ValueError("max_missed_frames cannot be negative")
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.next_id = 1
        self.tracked_faces = {}
        self.missed_frames = {}

    @staticmethod
    def _center(face):
        return (
            face["x"] + face["width"] / 2,
            face["y"] + face["height"] / 2,
        )

    @staticmethod
    def _distance(point_a, point_b):
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])

    def update(self, faces):
        detections = sorted(
            (dict(face) for face in faces),
            key=lambda face: (face["x"], face["y"]),
        )
        available_ids = set(self.tracked_faces)
        matches = []

        # Build all possible pairs first, then consume the closest pairs.
        # This prevents detection order from changing an existing ID.
        for face_index, face in enumerate(detections):
            for face_id in available_ids:
                distance = self._distance(
                    self._center(face),
                    self._center(self.tracked_faces[face_id]),
                )
                if distance <= self.max_distance:
                    matches.append((distance, face_index, face_id))

        matched_faces = set()
        matched_ids = set()
        current_faces = {}
        for _, face_index, face_id in sorted(matches):
            if face_index in matched_faces or face_id in matched_ids:
                continue
            face = detections[face_index]
            face["id"] = face_id
            face["visible"] = True
            current_faces[face_id] = face
            matched_faces.add(face_index)
            matched_ids.add(face_id)

        # Tối ưu cho bài toán 1 học sinh: Nếu toàn bộ khuôn mặt cũ đã bị xóa (mất dấu quá lâu)
        # thì khi có khuôn mặt mới xuất hiện, ta reset ID về 1 thay vì tăng lên 2, 3, 4...
        if len(self.tracked_faces) == 0 and len(matched_faces) == 0:
            self.next_id = 1

        for face_index, face in enumerate(detections):
            if face_index in matched_faces:
                continue
            
            face_id = self.next_id
            self.next_id += 1
            face["id"] = face_id
            face["visible"] = True
            current_faces[face_id] = face

        next_missed_frames = {}
        for face_id, previous_face in self.tracked_faces.items():
            if face_id in current_faces:
                next_missed_frames[face_id] = 0
                continue
            missed = self.missed_frames.get(face_id, 0) + 1
            if missed <= self.max_missed_frames:
                missed_face = dict(previous_face)
                missed_face["visible"] = False
                current_faces[face_id] = missed_face
                next_missed_frames[face_id] = missed

        self.tracked_faces = current_faces
        self.missed_frames = next_missed_frames
        return [current_faces[face_id] for face_id in sorted(current_faces)]

    def reset(self):
        self.next_id = 1
        self.tracked_faces = {}
        self.missed_frames = {}