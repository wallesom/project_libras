# app/detector.py
from collections import deque, Counter
import time
import cv2

from services.hand_tracking import HandTracker
from services.feature_extractor import FeatureExtractor
from services.model_service import LibrasModelService
from services.tts_service import TTSSpeaker
from services.movement_detector import MovementDetector


class LibrasDetector:
    def __init__(self,
                 tracker=None,
                 extractor=None,
                 model_service=None,
                 speaker=None,
                 movement_detector=None):
        self.tracker = tracker or HandTracker()
        self.extractor = extractor or FeatureExtractor()
        self.model_service = model_service or LibrasModelService()
        self.speaker = speaker or TTSSpeaker()
        self.movement_detector = movement_detector or MovementDetector()

        self.pred_history = deque(maxlen=10)
        self.min_stable_frames = 7
        self.min_confidence = 20.0

        self.last_spoken_letter = None
        self.last_spoken_time = 0
        self.min_interval = 1.5

        self.current_word = ""

    def _update_stable_letter(self, letter, conf):
        if letter is None:
            return None

        self.pred_history.append((letter, conf))

        if len(self.pred_history) < self.pred_history.maxlen:
            return None

        letters = [l for l, c in self.pred_history]
        counts = Counter(letters)
        candidate, count = counts.most_common(1)[0]

        confs = [c for l, c in self.pred_history if l == candidate]
        avg_conf = sum(confs) / len(confs)

        if count >= self.min_stable_frames and avg_conf >= self.min_confidence:
            self.pred_history.clear()
            print(f"[DEBUG] Letra/gesto estável: {candidate} (frames={count}, conf={avg_conf:.1f}%)")
            return candidate

        return None

    def _points_for_movement(self, hand_landmarks_list, face_landmarks):
        """
        Concatena pontos das mãos e do rosto para análise de movimento.
        """
        points = []

        for hand in hand_landmarks_list:
            for lm in hand.landmark:
                points.append((lm.x, lm.y, lm.z))

        if face_landmarks is not None:
            # usa os mesmos max_face_points do FeatureExtractor (30 primeiros)
            landmarks = face_landmarks.landmark
            n = min(30, len(landmarks))
            for i in range(n):
                lm = landmarks[i]
                points.append((lm.x, lm.y, lm.z))

        return points

    # ---------------- COLETA ----------------
    def collect_data(self, label, num_samples=100):
        """
        label pode ser:
          - letra estática: "A", "B", "C"
          - letra com movimento: "J", "H"
          - gesto que representa palavra: "OLA", "SIM" etc.
        """
        print(f"\n=== COLETANDO DADOS PARA: {label} ===")
        print(f"Pressione ESPAÇO para começar a capturar {num_samples} amostras")
        print("Pressione 'q' para cancelar")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Não foi possível abrir a câmera.")
            return [], label

        samples = []
        collecting = False
        count = 0

        try:
            while cap.isOpened() and count < num_samples:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame, hand_landmarks_list, face_landmarks = self.tracker.process_frame(frame_bgr)

                # desenha mãos
                for hand_lm in hand_landmarks_list:
                    self.tracker.draw_hand(frame, hand_lm)

                # desenha rosto (opcional, pra debug)
                if face_landmarks is not None:
                    self.tracker.draw_face(frame, face_landmarks)

                if collecting and hand_landmarks_list:
                    features = self.extractor.extract(hand_landmarks_list, face_landmarks)
                    samples.append(features)
                    count += 1

                status = "COLETANDO" if collecting else "AGUARDANDO"
                color = (0, 255, 0) if collecting else (0, 165, 255)

                cv2.putText(frame, f"Classe: {label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Status: {status}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Amostras: {count}/{num_samples}", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "ESPACO=iniciar | Q=sair", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                cv2.imshow('Coleta de Dados - Libras', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    collecting = not collecting
                elif key == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        print(f"✓ Coletadas {len(samples)} amostras para {label}")
        return samples, label

    def train_model(self):
        return self.model_service.train()

    # ---------------- MODO LETRAS / GESTOS ----------------
    def detect_letters_mode(self):
        if not self.model_service.ensure_model_loaded():
            print("✗ Nenhum modelo encontrado! Treine o modelo primeiro.")
            return

        print("\n=== MODO LETRAS / GESTOS (1 OU 2 MÃOS + ROSTO) ===")
        print("Q = sair")

        self.pred_history.clear()
        self.last_spoken_letter = None
        self.last_spoken_time = 0
        self.movement_detector = MovementDetector()

        self.speaker.speak("Modo de aprendizado de letras e gestos com duas mãos e rosto iniciado.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Não foi possível abrir a câmera.")
            return

        try:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame, hand_landmarks_list, face_landmarks = self.tracker.process_frame(frame_bgr)

                for hand_lm in hand_landmarks_list:
                    self.tracker.draw_hand(frame, hand_lm)
                if face_landmarks is not None:
                    self.tracker.draw_face(frame, face_landmarks)

                points = self._points_for_movement(hand_landmarks_list, face_landmarks)
                movement_state = self.movement_detector.update(points)

                predicted_label = None
                confidence = 0.0

                if hand_landmarks_list:
                    features = self.extractor.extract(hand_landmarks_list, face_landmarks)
                    predicted_label, confidence = self.model_service.predict(features)
                else:
                    self.pred_history.clear()

                stable_label = None
                if movement_state in ("stable", "movement_ended") and predicted_label is not None:
                    stable_label = self._update_stable_letter(predicted_label, confidence)
                else:
                    self.pred_history.clear()

                if predicted_label is not None:
                    cv2.rectangle(frame, (10, 10), (750, 110), (0, 0, 0), -1)
                    cv2.putText(frame,
                                f"Detectado: {predicted_label} ({confidence:.1f}%)",
                                (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame,
                                f"Movimento: {movement_state}",
                                (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

                cv2.putText(frame, "Modo LETRAS/GESTOS | Q = sair",
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                cv2.imshow("Libras - Modo Letras/Gestos", frame)
                key = cv2.waitKey(1) & 0xFF

                if stable_label is not None:
                    now = time.time()
                    if (stable_label != self.last_spoken_letter or
                            (now - self.last_spoken_time) > self.min_interval):
                        self.speaker.speak(stable_label)
                        self.last_spoken_letter = stable_label
                        self.last_spoken_time = now

                if key == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # ---------------- MODO PALAVRA (respeitando o fim do movimento) ----------------
    def detect_words_mode(self):
        if not self.model_service.ensure_model_loaded():
            print("✗ Nenhum modelo encontrado! Treine o modelo primeiro.")
            return

        print("\n=== MODO PALAVRA (GESTOS COMPLEXOS, DUAS MÃOS + ROSTO) ===")
        print("Q = sair | ESPAÇO = falar palavra | B = backspace | C = limpar")

        self.pred_history.clear()
        self.current_word = ""
        self.movement_detector = MovementDetector()

        self.speaker.speak("Modo de montagem de palavras com gestos iniciado.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Não foi possível abrir a câmera.")
            return

        try:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame, hand_landmarks_list, face_landmarks = self.tracker.process_frame(frame_bgr)

                for hand_lm in hand_landmarks_list:
                    self.tracker.draw_hand(frame, hand_lm)
                if face_landmarks is not None:
                    self.tracker.draw_face(frame, face_landmarks)

                points = self._points_for_movement(hand_landmarks_list, face_landmarks)
                movement_state = self.movement_detector.update(points)

                predicted_label = None
                confidence = 0.0

                if hand_landmarks_list:
                    features = self.extractor.extract(hand_landmarks_list, face_landmarks)
                    predicted_label, confidence = self.model_service.predict(features)
                else:
                    self.pred_history.clear()

                stable_label = None
                if movement_state in ("stable", "movement_ended") and predicted_label is not None:
                    stable_label = self._update_stable_letter(predicted_label, confidence)
                else:
                    self.pred_history.clear()

                # Regra principal: só adicionar letra/gesto quando o movimento TERMINA
                if stable_label is not None and movement_state == "movement_ended":
                    if not self.current_word or self.current_word[-1] != stable_label:
                        self.current_word += stable_label
                        print(f"[PALAVRA] {self.current_word}")

                cv2.rectangle(frame, (10, 10), (900, 110), (0, 0, 0), -1)
                cv2.putText(frame, f"Palavra: {self.current_word}",
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"Movimento: {movement_state}",
                            (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

                cv2.putText(
                    frame,
                    "Modo PALAVRA | ESPACO=falar | B=backspace | C=limpar | Q=sair",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2
                )

                cv2.imshow("Libras - Modo Palavra", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('b'):
                    if self.current_word:
                        self.current_word = self.current_word[:-1]
                elif key == ord('c'):
                    self.current_word = ""
                elif key == 32:
                    if self.current_word:
                        self.speaker.speak(self.current_word.lower())
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # ---------------- TESTE ÁUDIO ----------------
    def test_audio_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Não foi possível abrir a câmera.")
            return

        self.speaker.speak("Teste de áudio dentro do sistema de detecção de Libras.")

        print("=== TESTE DE ÁUDIO + CÂMERA ===")
        print("A: falar 'A'")
        print("T: falar uma frase de teste")
        print("Q: sair")

        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame_bgr, 1)
                cv2.putText(
                    frame,
                    "A: falar A | T: frase | Q: sair",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Teste audio+camera", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('a'):
                    self.speaker.speak("A")
                elif key == ord('t'):
                    self.speaker.speak("Teste de voz na detecao de Libras.")
                elif key == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    # ---------------- MODO BÁSICO ----------------
    def detect_realtime_basic(self):
        if not self.model_service.ensure_model_loaded():
            print("✗ Nenhum modelo encontrado! Treine o modelo primeiro.")
            return

        print("\n=== DETECÇÃO EM TEMPO REAL (BÁSICO, 2 MÃOS + ROSTO) ===")
        print("Q = sair | A = teste de áudio")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Não foi possível abrir a câmera.")
            return

        last_spoken_label = ""
        teste_audio_feito = False
        self.movement_detector = MovementDetector()

        try:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame, hand_landmarks_list, face_landmarks = self.tracker.process_frame(frame_bgr)

                for hand_lm in hand_landmarks_list:
                    self.tracker.draw_hand(frame, hand_lm)
                if face_landmarks is not None:
                    self.tracker.draw_face(frame, face_landmarks)

                points = self._points_for_movement(hand_landmarks_list, face_landmarks)
                movement_state = self.movement_detector.update(points)

                detected_label = ""
                confidence = 0.0

                if hand_landmarks_list:
                    features = self.extractor.extract(hand_landmarks_list, face_landmarks)
                    detected_label, confidence = self.model_service.predict(features)

                if detected_label:
                    cv2.rectangle(frame, (10, 10), (600, 170), (0, 0, 0), -1)
                    cv2.putText(frame, f"Classe: {detected_label}", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(frame, f"Confianca: {confidence:.1f}%", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"Movimento: {movement_state}", (20, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

                    print(f"[DEBUG] Detectado: {detected_label} ({confidence:.1f}%) [{movement_state}]")

                    if movement_state in ("stable", "movement_ended"):
                        if detected_label != last_spoken_label:
                            print(f"[DEBUG] Vou falar: {detected_label}")
                            self.speaker.speak(detected_label)
                            last_spoken_label = detected_label

                cv2.putText(
                    frame,
                    "Q = sair | A = teste de áudio",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2
                )

                cv2.imshow('Deteccao de Libras', frame)

                key = cv2.waitKey(1) & 0xFF

                if not teste_audio_feito:
                    print("[DEBUG] Teste automático de áudio dentro da detecção...")
                    self.speaker.speak("Teste de áudio dentro da função de detecção de Libras.")
                    teste_audio_feito = True

                if key == ord('a'):
                    print("[DEBUG] Tecla A pressionada. Teste manual de áudio.")
                    self.speaker.speak("Teste de áudio ao pressionar a tecla A.")

                if key == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
