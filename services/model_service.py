# services/model_service.py
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class LibrasModelService:
    def __init__(self,
                 data_file='libras_data.pkl',
                 model_file='libras_model.pkl'):
        base_dir = os.path.dirname(os.path.dirname(__file__))  # ia_hackathon/
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        self.data_file = os.path.join(data_dir, data_file)
        self.model_file = os.path.join(data_dir, model_file)
        self.model = None

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'rb') as f:
                return pickle.load(f)
        return []

    def save_data(self, all_data):
        with open(self.data_file, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"\n✓ Dados salvos em: {self.data_file}")

    def append_samples(self, samples, letter):
        all_data = self.load_data()
        all_data.append((samples, letter))
        self.save_data(all_data)

    def load_model(self):
        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            return True
        return False

    def train(self):
        print("\n=== TREINANDO MODELO ===")

        all_data = self.load_data()
        if not all_data:
            print("✗ Nenhum dado encontrado! Colete dados primeiro.")
            return False

        X, y = [], []
        for samples, letter in all_data:
            X.extend(samples)
            y.extend([letter] * len(samples))

        X = np.array(X)
        y = np.array(y)

        print(f"Total de amostras: {len(X)}")
        print(f"Letras/gestos: {np.unique(y)}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

        print("Treinando...")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n✓ Acurácia no conjunto de teste: {accuracy:.2%}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))

        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✓ Modelo salvo em: {self.model_file}")

        return True

    def ensure_model_loaded(self):
        if self.model is None:
            return self.load_model()
        return True

    def predict(self, feature_vector):
        if self.model is None:
            raise RuntimeError("Modelo não carregado.")

        features = feature_vector.reshape(1, -1)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = float(max(probabilities) * 100)
        return str(prediction), confidence

    def predict(self, feature_vector):
        if self.model is None:
            raise RuntimeError("Modelo não carregado.")

        features = feature_vector.reshape(1, -1)

        expected = getattr(self.model, "n_features_in_", None)
        if expected is not None and features.shape[1] != expected:
            raise ValueError(
                f"Dimensão de features incompatível: modelo espera {expected}, "
                f"mas recebeu {features.shape[1]}. "
                f"Apague os arquivos em 'data/' e treine o modelo novamente."
            )

        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = float(max(probabilities) * 100)
        return str(prediction), confidence
