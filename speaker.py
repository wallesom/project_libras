import sys
import pyttsx3


def main():
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Nenhum texto informado."
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    main()
