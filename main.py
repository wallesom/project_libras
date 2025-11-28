# main.py
from app.detector import LibrasDetector


def main():
    detector = LibrasDetector()
    model_service = detector.model_service

    while True:
        print("\n" + "=" * 50)
        print("SISTEMA DE DETECÇÃO DE LIBRAS")
        print("=" * 50)
        print("1. Coletar dados de treino (letras/gestos)")
        print("2. Treinar modelo")
        print("3. Testar detecção em tempo real (básico)")
        print("4. Ver dados coletados")
        print("5. Modo LETRAS/GESTOS (estável + áudio)")
        print("6. Modo PALAVRA")
        print("7. Teste de áudio + câmera")
        print("8. Sair")
        print("=" * 50)

        choice = input("\nEscolha uma opção: ")

        if choice == '1':
            label = input("Digite a classe (ex: A, J, H, OLA, SIM): ").upper()
            num_samples = int(input("Número de amostras (recomendado: 100-200): "))
            samples, label = detector.collect_data(label, num_samples)
            if samples:
                model_service.append_samples(samples, label)

        elif choice == '2':
            detector.train_model()

        elif choice == '3':
            detector.detect_realtime_basic()

        elif choice == '4':
            all_data = model_service.load_data()
            if all_data:
                print("\n=== DADOS COLETADOS ===")
                for samples, label in all_data:
                    print(f"Classe {label}: {len(samples)} amostras")
            else:
                print("\n✗ Nenhum dado coletado ainda.")

        elif choice == '5':
            detector.detect_letters_mode()

        elif choice == '6':
            detector.detect_words_mode()

        elif choice == '7':
            detector.test_audio_loop()

        elif choice == '8':
            print("\nTeste Encerrado!")
            break
        else:
            print("\n✗ Opção inválida!")


if __name__ == "__main__":
    main()
