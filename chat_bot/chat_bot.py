import sys
import os
from typing import List, Optional

# If text_to_speech.py is in a sibling directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'record_speech'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'text_processor'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'speech_to_text'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'text_to_speech'))

from record_speech import Recorder
from speech_to_text import AudioToTextConverter
from text_processor import GPTTextProcessor
from text_to_speech import TextToSpeech

import argparse

def load_llm(openai_key_path: str, document_path: Optional[str]) -> GPTTextProcessor:
    with open(openai_key_path, 'r') as f:
        openai_key = f.read().strip()
    llm = GPTTextProcessor(
        api_key=openai_key,
        model="gpt-3.5-turbo"
    )
    if document_path:
        with open(document_path, 'r') as f:
            content = f.read()
        # llm.add_documents([content])
        if document_path.endswith(".csv"):
            lines = content.split('\n')
            if len(lines) > 1:
                # Assuming the CSV has a header and we want to treat each row as a separate document with context from the header.
                header = lines[0]
                documents = [f"{header}\n{line}" for line in lines[1:] if line]
                print(f"Adding documents: {documents}")
                llm.add_documents(documents)
        else:
            llm.add_documents([content])
    return llm

class ChatUI:
    """Abstract base class for chat user interfaces."""
    def __init__(self, llm: GPTTextProcessor):
        self.llm = llm
        self.context = ""

    def start_greeting(self):
        raise NotImplementedError

    def get_input(self) -> Optional[str]:
        """Gets user input. Returns None to signal quit."""
        raise NotImplementedError

    def post_output(self, text: str):
        raise NotImplementedError

    def on_quit(self):
        """Handles any cleanup or final message on quitting."""
        raise NotImplementedError

class SpeechUI(ChatUI):
    """Handles chat interaction via speech."""
    def __init__(self, llm: GPTTextProcessor, gcp_key_path: str):
        super().__init__(llm)
        print("Setting up speech components...")
        self.recorder = Recorder()
        self.audio_to_text = AudioToTextConverter(gcp_key_path)
        self.tts = TextToSpeech()

    def start_greeting(self):
        self.tts.speak("Hello, how can I help?")

    def get_input(self) -> Optional[str]:
        audio = self.recorder.record()
        text = self.audio_to_text.transcribe_content(audio, ".wav")['full_transcript']
        if not text.strip():
            return ""  # Return empty string for no input, loop continues
        print("Transcribed text:", text)
        if "quit" in text.lower():
            return None  # Signal to quit
        return text

    def post_output(self, text: str):
        print(f"Response: {text}")
        self.tts.speak(text)

    def on_quit(self):
        self.tts.speak("quitting!")

class CmdUI(ChatUI):
    """Handles chat interaction via command line."""
    def start_greeting(self):
        print("Hello, how can I help? (type 'quit' to exit)")

    def get_input(self) -> Optional[str]:
        try:
            text = input("You: ")
            if "quit" in text.lower():
                return None  # Signal to quit
            return text
        except (EOFError, KeyboardInterrupt):
            return None  # Signal to quit

    def post_output(self, text: str):
        print(f"Bot: {text}")

    def on_quit(self):
        print("\nQuitting!")

def get_chat_ui(ui_name: str, args: argparse.Namespace, llm: GPTTextProcessor) -> ChatUI:
    """Factory function to create a chat UI instance."""
    if ui_name == 'speech':
        if not args.gcp_key_path:
            raise ValueError("--gcp_key_path is required for speech mode.")
        return SpeechUI(llm, args.gcp_key_path)
    elif ui_name == 'cmd':
        return CmdUI(llm)
    else:
        raise ValueError(f"Unknown mode: {ui_name}")

def main():
    parser = argparse.ArgumentParser(description="Chatbot with voice interface")
    parser.add_argument("--gcp_key_path", type=str, help="Path to Google Cloud credentials JSON file. Required for 'speech' mode.")
    parser.add_argument("--openai_key_path", type=str, help="Path to file containing OpenAI API key", required=True)
    parser.add_argument("--document_path", type=str, help="Path to documents to add to the knowledge base", required=False)
    parser.add_argument("--mode", type=str, default='speech', choices=['speech', 'cmd'], help="Interaction UI: 'speech' for voice, 'cmd' for command-line text.")
    args = parser.parse_args()

    # LLM
    print("Setting up GPTTextProcessor")
    llm = load_llm(args.openai_key_path, args.document_path)

    try:
        chat_ui = get_chat_ui(args.mode, args, llm)
    except ValueError as e:
        parser.error(str(e))

    chat_ui.start_greeting()

    while True:
        user_input = chat_ui.get_input()
        if user_input is None:
            chat_ui.on_quit()
            break
        if not user_input.strip():
            continue
        response = chat_ui.llm.process_text(user_input, chat_ui.context).response
        chat_ui.post_output(response)
        chat_ui.context += f"\nmy last query: {user_input}\n"
        chat_ui.context += f"\nyour last response: {response}\n"


if __name__ == "__main__":
        main()