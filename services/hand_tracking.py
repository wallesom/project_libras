# services/hand_tracking.py
import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame_bgr):
        """
        Retorna:
          - frame (espelhado)
          - lista de mãos [hand_landmarks, ...] (0, 1 ou 2 mãos)
          - face_landmarks (ou None)
        """
        frame = cv2.flip(frame_bgr, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)

        hand_landmarks_list = hand_results.multi_hand_landmarks or []
        face_landmarks = (
            face_results.multi_face_landmarks[0]
            if face_results.multi_face_landmarks
            else None
        )

        return frame, hand_landmarks_list, face_landmarks

    def draw_hand(self, frame, hand_landmarks):
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS
        )

    def draw_face(self, frame, face_landmarks):
        # Desenha só contornos básicos do rosto (pra debug)
        self.mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            self.mp_face.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                thickness=1,
                circle_radius=1
            )
        )
