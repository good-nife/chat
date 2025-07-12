import sys
import os

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

def main():
    parser = argparse.ArgumentParser(description="Chatbot with voice interface")
    parser.add_argument("--gcp_key_path", type=str, help="Path to Google Cloud credentials JSON file", required=True)
    parser.add_argument("--openai_key_path", type=str, help="Path to file containing OpenAI API key", required=True)
    args = parser.parse_args()

    # Recorder for speech
    recorder = Recorder()

    # Speech to text converter
    print("Setting up AudioToTextConverter")
    with open(args.openai_key_path, 'r') as f:
        openai_key = f.read().strip()
    audio_to_text = AudioToTextConverter(args.gcp_key_path)

    # LLM
    print("Setting up GPTTextProcessor")
    llm = GPTTextProcessor(
        api_key=openai_key,
        model="gpt-3.5-turbo"
    )

    # Text to speech converter
    tts = TextToSpeech()

    # Save context
    context = ""

    tts.speak("Hello, how can I help?")
    text = ""
    while True:
        audio = recorder.record()
        text = audio_to_text.transcribe_content(audio, ".wav")['full_transcript']
        if text.find("quit") != -1 or text.find("Quit") != -1:
            tts.speak("quitting!")
            return
        print("Transcribed text:", text)
        response = llm.process_text(text, context).response
        print(f"Response: {response}")
        tts.speak(response)
        context += f"\nmy last query: {text}\n"
        context += f"\nyour last response: {response}\n"


if __name__ == "__main__":
        main()