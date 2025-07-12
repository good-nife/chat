"""
Audio to Text Library using Google Cloud Speech-to-Text API

This library provides a simple interface for converting audio files to text
using Google Cloud Speech-to-Text service.

Requirements:
- pip install google-cloud-speech
- Google Cloud credentials (service account key or application default credentials)
- Audio files in supported formats (FLAC, WAV, MP3, etc.)
"""

import io
import os
from typing import Optional, List, Dict, Any
from google.cloud import speech
from google.oauth2 import service_account
import json


class AudioToTextConverter:
    """
    A class for converting audio files to text using Google Cloud Speech-to-Text.
    """
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize the AudioToTextConverter.
        
        Args:
            credentials_path: Path to Google Cloud service account JSON file.
                            If None, uses application default credentials.
        """
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path
            )
            self.client = speech.SpeechClient(credentials=credentials)
        else:
            # Use application default credentials
            self.client = speech.SpeechClient()
    
    def transcribe_content(
        self,
        content: bytes,
        file_extension: str = ".wav",
        language_code: str = "en-US",
        sample_rate: Optional[int] = None,
        audio_channel_count: Optional[int] = None,
        enable_word_time_offsets: bool = False,
        enable_automatic_punctuation: bool = True,
        model: str = "default",
        use_enhanced: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file to text.
        
        Args:
            content: Audio file content
            language_code: Language code (e.g., "en-US", "es-ES")
            sample_rate: Sample rate of the audio file
            audio_channel_count: Number of audio channels
            enable_word_time_offsets: Include word-level timestamps
            enable_automatic_punctuation: Add punctuation automatically
            model: Speech recognition model to use
            use_enhanced: Use enhanced model (premium feature)
            
        Returns:
            Dictionary containing transcription results
        """

        # Determine audio encoding from file extension
        encoding = self._get_audio_encoding(file_extension)

        # Configure audio settings
        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=sample_rate,
            language_code=language_code,
            audio_channel_count=audio_channel_count,
            enable_word_time_offsets=enable_word_time_offsets,
            enable_automatic_punctuation=enable_automatic_punctuation,
            model=model,
            use_enhanced=use_enhanced,
        )
        
        # Create recognition request
        audio = speech.RecognitionAudio(content=content)
        
        # Perform the transcription
        response = self.client.recognize(config=config, audio=audio)
        
        # Process results
        return self._process_response(response)


    def transcribe_file(
        self,
        audio_file_path: str,
        language_code: str = "en-US",
        sample_rate: Optional[int] = None,
        audio_channel_count: Optional[int] = None,
        enable_word_time_offsets: bool = False,
        enable_automatic_punctuation: bool = True,
        model: str = "default",
        use_enhanced: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_file_path: Path to the audio file
            language_code: Language code (e.g., "en-US", "es-ES")
            sample_rate: Sample rate of the audio file
            audio_channel_count: Number of audio channels
            enable_word_time_offsets: Include word-level timestamps
            enable_automatic_punctuation: Add punctuation automatically
            model: Speech recognition model to use
            use_enhanced: Use enhanced model (premium feature)
            
        Returns:
            Dictionary containing transcription results
        """
        # Read the audio file
        with io.open(audio_file_path, "rb") as audio_file:
            content = audio_file.read()

        # Get file extension
        file_extension = os.path.splitext(audio_file_path)[1].lower()

        return self.transcribe_content(
            content,
            file_extension,
            language_code,
            sample_rate,
            audio_channel_count,
            enable_word_time_offsets,
            enable_automatic_punctuation,
            model,
            use_enhanced)
    
    def transcribe_long_audio(
        self,
        audio_file_path: str,
        language_code: str = "en-US",
        sample_rate: Optional[int] = None,
        audio_channel_count: Optional[int] = None,
        enable_word_time_offsets: bool = False,
        enable_automatic_punctuation: bool = True,
        model: str = "default",
        use_enhanced: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe a long audio file using asynchronous recognition.
        Use this for audio files longer than 1 minute.
        
        Args:
            Same as transcribe_file()
            
        Returns:
            Dictionary containing transcription results
        """
        # Read the audio file
        with io.open(audio_file_path, "rb") as audio_file:
            content = audio_file.read()
        
        # Determine audio encoding from file extension
        file_extension = os.path.splitext(audio_file_path)[1].lower()
        encoding = self._get_audio_encoding(file_extension)
        
        # Configure audio settings
        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=sample_rate,
            language_code=language_code,
            audio_channel_count=audio_channel_count,
            enable_word_time_offsets=enable_word_time_offsets,
            enable_automatic_punctuation=enable_automatic_punctuation,
            model=model,
            use_enhanced=use_enhanced,
        )
        
        # Create recognition request
        audio = speech.RecognitionAudio(content=content)
        
        # Perform long-running recognition
        operation = self.client.long_running_recognize(config=config, audio=audio)
        
        print("Waiting for operation to complete...")
        response = operation.result(timeout=90)
        
        # Process results
        return self._process_response(response)
    
    def transcribe_streaming(
        self,
        audio_generator,
        language_code: str = "en-US",
        sample_rate: int = 16000,
        chunk_size: int = 1024
    ) -> List[str]:
        """
        Transcribe audio from a streaming source.
        
        Args:
            audio_generator: Generator yielding audio chunks
            language_code: Language code
            sample_rate: Sample rate of the audio stream
            chunk_size: Size of each audio chunk
            
        Returns:
            List of transcribed text segments
        """
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=language_code,
        )
        
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
        )
        
        def request_generator():
            yield speech.StreamingRecognizeRequest(
                streaming_config=streaming_config
            )
            for chunk in audio_generator:
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
        
        requests = request_generator()
        responses = self.client.streaming_recognize(requests)
        
        results = []
        for response in responses:
            for result in response.results:
                if result.is_final:
                    results.append(result.alternatives[0].transcript)
        
        return results
    
    def _get_audio_encoding(self, file_extension: str) -> speech.RecognitionConfig.AudioEncoding:
        """
        Determine audio encoding from file extension.
        
        Args:
            file_extension: File extension (e.g., '.wav', '.flac')
            
        Returns:
            Google Cloud Speech audio encoding enum
        """
        encoding_map = {
            '.wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
            '.flac': speech.RecognitionConfig.AudioEncoding.FLAC,
            '.mp3': speech.RecognitionConfig.AudioEncoding.MP3,
            '.ogg': speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            '.webm': speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
        }
        
        return encoding_map.get(
            file_extension, 
            speech.RecognitionConfig.AudioEncoding.LINEAR16
        )
    
    def _process_response(self, response) -> Dict[str, Any]:
        """
        Process the API response and extract useful information.
        
        Args:
            response: Google Cloud Speech API response
            
        Returns:
            Dictionary with processed results
        """
        results = {
            'transcripts': [],
            'confidence_scores': [],
            'word_timestamps': [],
            'full_transcript': ''
        }
        
        transcript_parts = []
        
        for result in response.results:
            alternative = result.alternatives[0]
            
            # Add transcript and confidence
            results['transcripts'].append(alternative.transcript)
            results['confidence_scores'].append(alternative.confidence)
            transcript_parts.append(alternative.transcript)
            
            # Add word timestamps if available
            if hasattr(alternative, 'words'):
                word_info = []
                for word in alternative.words:
                    word_info.append({
                        'word': word.word,
                        'start_time': word.start_time.total_seconds(),
                        'end_time': word.end_time.total_seconds()
                    })
                results['word_timestamps'].append(word_info)
        
        # Combine all transcripts
        results['full_transcript'] = ' '.join(transcript_parts)
        
        return results


class AudioToTextError(Exception):
    """Custom exception for audio-to-text conversion errors."""
    pass


# Convenience functions for simple usage
def transcribe_audio_file(
    audio_file_path: str,
    credentials_path: Optional[str] = None,
    language_code: str = "en-US"
) -> str:
    """
    Simple function to transcribe an audio file to text.
    
    Args:
        audio_file_path: Path to the audio file
        credentials_path: Path to Google Cloud credentials JSON file
        language_code: Language code for transcription
        
    Returns:
        Transcribed text as string
    """
    try:
        converter = AudioToTextConverter(credentials_path)
        result = converter.transcribe_file(audio_file_path, language_code)
        return result['full_transcript']
    except Exception as e:
        raise AudioToTextError(f"Transcription failed: {str(e)}")


# Convenience functions for simple usage
def transcribe_audio_content(
    audio_content: bytes,
    file_extension: str = ".wav",
    credentials_path: Optional[str] = None,
    language_code: str = "en-US"
) -> str:
    """
    Simple function to transcribe an audio file to text.
    
    Args:
        audio_file_path: Path to the audio file
        credentials_path: Path to Google Cloud credentials JSON file
        language_code: Language code for transcription
        
    Returns:
        Transcribed text as string
    """
    try:
        converter = AudioToTextConverter(credentials_path)
        result = converter.transcribe_content(audio_content, file_extension)
        return result['full_transcript']
    except Exception as e:
        raise AudioToTextError(f"Transcription failed: {str(e)}")



def transcribe_long_audio(
    audio_file_path: str,
    credentials_path: Optional[str] = None,
    language_code: str = "en-US"
) -> str:
    """
    Simple function to transcribe a long audio file to text.
    
    Args:
        audio_file_path: Path to the audio file
        credentials_path: Path to Google Cloud credentials JSON file
        language_code: Language code for transcription
        
    Returns:
        Transcribed text as string
    """
    try:
        converter = AudioToTextConverter(credentials_path)
        result = converter.transcribe_long_audio(audio_file_path, language_code)
        return result['full_transcript']
    except Exception as e:
        raise AudioToTextError(f"Long audio transcription failed: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Example 1: Simple transcription
    try:
        text = transcribe_audio_file("welcome_message.mp3", "<GCP JSON key>")
        print("Transcribed text:", text)
    except AudioToTextError as e:
        print(f"Error: {e}")
    
    # Example 2: Advanced transcription with word timestamps
    try:
        converter = AudioToTextConverter("<GCP JSON key>")
        result = converter.transcribe_file(
            "welcome_message.mp3",
            language_code="en-US",
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True
        )
        
        print("Full transcript:", result['full_transcript'])
        print("Confidence scores:", result['confidence_scores'])
        
        # Print word timestamps
        for i, word_list in enumerate(result['word_timestamps']):
            print(f"Segment {i + 1} word timings:")
            for word_info in word_list:
                print(f"  '{word_info['word']}': {word_info['start_time']:.2f}s - {word_info['end_time']:.2f}s")
    
    except AudioToTextError as e:
        print(f"Error: {e}")