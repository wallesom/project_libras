# services/feature_extractor.py
import numpy as np


class FeatureExtractor:
    def __init__(self, max_face_points=30):
        self.max_face_points = max_face_points

    def _hand_features(self, hand_landmarks):
        """
        Features de UMA mão (21 pontos):
        - coords x,y,z de cada ponto
        - distâncias polegar -> pontas dos dedos
        - dif. de comprimentos entre ponta e base de cada dedo
        """
        if hand_landmarks is None:
            # 21 pontos => 21*3 + 4 + 5 = 72 features
            return np.zeros(72, dtype=float)

        feats = []

        # coords cruas
        for landmark in hand_landmarks.landmark:
            feats.extend([landmark.x, landmark.y, landmark.z])

        # distâncias polegar -> pontas dos dedos
        thumb_tip = hand_landmarks.landmark[4]
        for finger_tip_id in [8, 12, 16, 20]:
            finger_tip = hand_landmarks.landmark[finger_tip_id]
            distance = np.sqrt(
                (thumb_tip.x - finger_tip.x) ** 2 +
                (thumb_tip.y - finger_tip.y) ** 2 +
                (thumb_tip.z - finger_tip.z) ** 2
            )
            feats.append(distance)

        # diferença de distâncias em cada dedo
        finger_indices = [
            [4, 3, 2],
            [8, 6, 5],
            [12, 10, 9],
            [16, 14, 13],
            [20, 18, 17]
        ]

        for indices in finger_indices:
            tip = hand_landmarks.landmark[indices[0]]
            mid = hand_landmarks.landmark[indices[1]]
            base = hand_landmarks.landmark[indices[2]]

            dist_tip_base = np.sqrt(
                (tip.x - base.x) ** 2 +
                (tip.y - base.y) ** 2 +
                (tip.z - base.z) ** 2
            )
            dist_mid_base = np.sqrt(
                (mid.x - base.x) ** 2 +
                (mid.y - base.y) ** 2 +
                (mid.z - base.z) ** 2
            )
            feats.append(dist_tip_base - dist_mid_base)

        return np.array(feats, dtype=float)

    def _face_features(self, face_landmarks):
        """
        Usa os primeiros max_face_points pontos do rosto.
        Se não houver rosto, preenche com zeros.
        """
        feats = []
        if face_landmarks is not None:
            landmarks = face_landmarks.landmark
            n = min(self.max_face_points, len(landmarks))
            for i in range(n):
                lm = landmarks[i]
                feats.extend([lm.x, lm.y, lm.z])

            # se tiver menos que max_face_points (em teoria não acontece), completa zeros
            if n < self.max_face_points:
                missing = self.max_face_points - n
                feats.extend([0.0, 0.0, 0.0] * missing)
        else:
            feats.extend([0.0, 0.0, 0.0] * self.max_face_points)

        return np.array(feats, dtype=float)

    def extract(self, hand_landmarks_list, face_landmarks=None):
        """
        hand_landmarks_list: lista de 0, 1 ou 2 mãos.
        face_landmarks: landmarks do rosto (ou None).

        Ordem:
          mão[0], mão[1] (se existir), rosto
        """
        # garantir no máximo 2 mãos
        hands = list(hand_landmarks_list[:2])

        # se só tiver 1 mão, adiciona None pra segunda vaga
        if len(hands) == 1:
            hands.append(None)
        elif len(hands) == 0:
            hands = [None, None]

        hand1_feats = self._hand_features(hands[0])
        hand2_feats = self._hand_features(hands[1])

        face_feats = self._face_features(face_landmarks)

        features = np.concatenate([hand1_feats, hand2_feats, face_feats])
        return features
