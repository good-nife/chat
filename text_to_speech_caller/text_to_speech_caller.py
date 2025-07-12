#!/usr/bin/env python3
"""
Example of using the text_to_speech module in another application
"""
import sys
import os

# If text_to_speech.py is in a sibling directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'text_to_speech'))

from text_to_speech import speak
from text_to_speech import save_speech
from text_to_speech import TextToSpeech

def main():
    """Main application function"""
    print("Welcome to my application!")
    
    # Method 1: Simple speaking using the convenience function
    speak("Welcome to my application!")
    
    # Method 2: Save speech to file
    save_speech("This message will be saved to a file", "welcome_message.mp3")
    
    # Method 3: Advanced usage with TextToSpeech class
    tts = TextToSpeech(preferred_engine='pyttsx3')  # Use reliable engine
    
    # Speak with custom settings
    tts.speak("This is spoken with custom settings", rate=150, volume=0.9)
    
    # Interactive example
    user_input = input("Enter something to speak: ")
    if user_input.strip():
        speak(user_input)
    
    # Async example
    print("Speaking multiple messages asynchronously...")
    tts.speak_async("First message")
    tts.speak_async("Second message")
    tts.speak_async("Third message")
    
    # Wait for async messages to complete
    import time
    time.sleep(5)
    tts.stop_async()
    
    print("Done!")

def create_voice_assistant():
    """Example of a simple voice assistant"""
    from text_to_speech import TextToSpeech
    
    # Initialize with preferred engine
    tts = TextToSpeech(preferred_engine='pyttsx3')
    
    # Greet user
    tts.speak("Hello! I am your voice assistant. How can I help you today?")
    
    # Simple command processing
    while True:
        try:
            command = input("\nEnter command (or 'quit' to exit): ").strip().lower()
            
            if command in ['quit', 'exit', 'bye']:
                tts.speak("Goodbye!")
                break
            elif command == 'time':
                import datetime
                current_time = datetime.datetime.now().strftime("%I:%M %p")
                tts.speak(f"The current time is {current_time}")
            elif command == 'date':
                import datetime
                current_date = datetime.datetime.now().strftime("%B %d, %Y")
                tts.speak(f"Today's date is {current_date}")
            elif command.startswith('say '):
                message = command[4:]  # Remove 'say ' prefix
                tts.speak(message)
            elif command == 'help':
                help_text = "Available commands: time, date, say followed by message, or quit"
                tts.speak(help_text)
                print(help_text)
            else:
                tts.speak("I don't understand that command. Say help for available commands.")
                
        except KeyboardInterrupt:
            tts.speak("Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def text_reader():
    """Example of a text file reader"""
    from text_to_speech import TextToSpeech
    
    tts = TextToSpeech()
    
    filename = input("Enter the name of a text file to read: ").strip()
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            
        if content.strip():
            tts.speak(f"Reading file: {filename}")
            
            # Split into smaller chunks for better speech
            sentences = content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    tts.speak(sentence + ".")
                    
        else:
            tts.speak("The file is empty")
            
    except FileNotFoundError:
        tts.speak(f"Could not find file: {filename}")
    except Exception as e:
        tts.speak(f"Error reading file: {str(e)}")

if __name__ == "__main__":
    print("TTS Module Usage Examples")
    print("=" * 30)
    
    # Ask user which example to run
    print("Choose an example:")
    print("1. Basic usage")
    print("2. Voice assistant")
    print("3. Text file reader")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        main()
    elif choice == '2':
        create_voice_assistant()
    elif choice == '3':
        text_reader()
    else:
        print("Invalid choice. Running basic example...")
        main()