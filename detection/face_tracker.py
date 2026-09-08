import math


class FaceTracker:

    def __init__(self, max_distance=80):

        self.next_id = 1

        self.tracked_faces = {}

        self.max_distance = max_distance

    def _center(self, face):

        x = face["x"]
        y = face["y"]

        w = face["width"]
        h = face["height"]

        center_x = x + w // 2
        center_y = y + h // 2

        return center_x, center_y

    def _distance(self, point1, point2):

        return math.sqrt(
            (point1[0] - point2[0]) ** 2
            +
            (point1[1] - point2[1]) ** 2
        )

    def update(self, faces):

        current_faces = {}

        used_ids = set()

        for face in faces:

            current_center = self._center(face)

            best_id = None
            best_distance = self.max_distance

            for face_id, old_face in self.tracked_faces.items():

                if face_id in used_ids:
                    continue

                old_center = self._center(old_face)

                distance = self._distance(
                    current_center,
                    old_center
                )

                if distance < best_distance:

                    best_distance = distance
                    best_id = face_id

            if best_id is None:

                best_id = self.next_id

                self.next_id += 1

            face["id"] = best_id

            current_faces[best_id] = face

            used_ids.add(best_id)

        self.tracked_faces = current_faces

        return list(current_faces.values())

    def reset(self):

        self.next_id = 1

        self.tracked_faces = {}