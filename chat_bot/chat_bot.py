import sys
import os

# If text_to_speech.py is in a sibling directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'record_speech'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'text_processor'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'speech_to_text'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'text_to_speech'))

from record_speech import record_and_return_with_vad
from speech_to_text import transcribe_audio_content
from speech_to_text import AudioToTextConverter
from text_processor import ask_llm
from text_processor import GPTTextProcessor
from text_to_speech import TextToSpeech

def main():
    audio_to_text = AudioToTextConverter("<GCP JSON key>")
    llm = GPTTextProcessor(
        api_key="<open API key>",  # Replace with your API key
        model="gpt-3.5-turbo"
    )
    tts = TextToSpeech()

    context = ""
    text = ""
    while True:
        audio = record_and_return_with_vad()
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

    # text = transcribe_audio_content(audio, ".wav", "../../gcp/linear-ellipse-465702-e7-e0955d3d7a30.json")
    # print("Transcribed text:", text)
    # response = ask_llm(text)
    # print(f"Response: {response}")
    # tts.speak(response)

if __name__ == "__main__":
    main()