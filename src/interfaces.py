class InteractionInterface:
    def output(self, text: str):
        raise NotImplementedError
    def input(self, prompt: str = "") -> str:
        raise NotImplementedError

class TextCLI(InteractionInterface):
    def output(self, text: str):
        # Blue for AI
        print(f"\n\033[1;34m[AI Examiner]: {text}\033[0m")

    def input(self, prompt: str = "") -> str:
        # Green for Student
        return input(f"\n\033[1;32m[Student]: \033[0m").strip()

# Future: class WebInterface(InteractionInterface)...