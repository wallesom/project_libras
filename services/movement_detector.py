# services/movement_detector.py
import math


class MovementDetector:
    def __init__(self,
                 move_threshold=0.02,
                 still_frames_required=5):
        """
        move_threshold: sensibilidade ao movimento (distância média entre frames)
        still_frames_required: quantos frames "quase parados" pra considerar que o gesto terminou
        """
        self.move_threshold = move_threshold
        self.still_frames_required = still_frames_required

        self.last_points = None
        self.is_moving = False
        self.still_frames = 0

    def _avg_distance(self, pts1, pts2):
        if pts1 is None or pts2 is None:
            return 0.0
        n = min(len(pts1), len(pts2))
        if n == 0:
            return 0.0
        total = 0.0
        for i in range(n):
            x1, y1, z1 = pts1[i]
            x2, y2, z2 = pts2[i]
            dx = x1 - x2
            dy = y1 - y2
            dz = z1 - z2
            total += math.sqrt(dx * dx + dy * dy + dz * dz)
        return total / n

    def update(self, points):
        """
        points: lista de (x, y, z) combinando mãos e rosto.
        Retorna:
          - "no_points"
          - "moving"
          - "stable"
          - "movement_ended"
        """
        if not points:
            # sumiu tudo da tela (sem mãos/rosto confiáveis)
            self.last_points = None
            self.is_moving = False
            self.still_frames = 0
            return "no_points"

        if self.last_points is None:
            self.last_points = points
            self.is_moving = False
            self.still_frames = 0
            return "stable"

        dist = self._avg_distance(self.last_points, points)
        self.last_points = points

        if dist > self.move_threshold:
            # há movimento significativo em qualquer ponto (mãos/rosto)
            self.is_moving = True
            self.still_frames = 0
            return "moving"
        else:
            if self.is_moving:
                # estava em movimento e começou a ficar parado
                self.still_frames += 1
                if self.still_frames >= self.still_frames_required:
                    self.is_moving = False
                    self.still_frames = 0
                    return "movement_ended"
                else:
                    return "moving"
            else:
                return "stable"
