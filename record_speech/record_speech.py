import io
import pyaudio
import webrtcvad
import wave
from collections import deque

_FORMAT = pyaudio.paInt16
_CHANNELS = 1
_RATE = 16000  # VAD requires 8000, 16000, 32000, or 48000 Hz
_CHUNK_DURATION_MS = 30  # VAD supports 10, 20, or 30 ms chunks
_CHUNK_SIZE = int(_RATE * _CHUNK_DURATION_MS / 1000)
_PADDING_DURATION_MS = 1000  # Add 1 second of padding before and after speech
_PADDING_DURATION_PERCENT_TO_TRIGGER = 0.2 # If there's sound for > 20% of the padding duration, we start recording.
_PADDING_DURATION_PERCENT_TO_STOP_RECORDING = 0.4 # If there's no sound for > 40% of the padding duration, we stop recording.
_NUM_PADDING_CHUNKS = int(_PADDING_DURATION_MS / _CHUNK_DURATION_MS)
_SILENCE_TO_END_MS = 2000 # Stop recording after 2 seconds of silence
_NUM_SILENCE_CHUNKS_TO_END = int(_SILENCE_TO_END_MS / _CHUNK_DURATION_MS)

class Recorder:
    def __init__(self):
        self._vad = webrtcvad.Vad(3)  # Set aggressiveness mode (0-3, 3 is most aggressive)
        self._pyaudio = pyaudio.PyAudio()
        self._stream = self._pyaudio.open(format=_FORMAT, channels=_CHANNELS, rate=_RATE,
                    input=True, frames_per_buffer=_CHUNK_SIZE)

    def __del__(self):
        self._stream.stop_stream()
        self._stream.close()
        self._pyaudio.terminate()

    def _record_with_vad(self, file_path_or_stream):
        """
        Records audio from the microphone, starting when speech is detected
        and stopping after a period of silence.
        """
        print("Listening...")

        ring_buffer = deque(maxlen=_NUM_PADDING_CHUNKS)
        triggered = False  # Flag to indicate if speech has been detected
        voiced_frames = []
        silent_chunks = 0

        while True:
            frame = self._stream.read(_CHUNK_SIZE)
            is_speech = self._vad.is_speech(frame, _RATE)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > _PADDING_DURATION_PERCENT_TO_TRIGGER * ring_buffer.maxlen:
                    triggered = True
                    print("Speech detected, starting recording.")
                    voiced_frames.extend([f for f, s in ring_buffer])
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > _PADDING_DURATION_PERCENT_TO_STOP_RECORDING * ring_buffer.maxlen:
                    silent_chunks += 1
                    if silent_chunks > _NUM_SILENCE_CHUNKS_TO_END:
                        print("Silence detected, stopping recording.")
                        break
                else:
                    silent_chunks = 0

        print("Finished recording.")

        # Save the recording
        with wave.open(file_path_or_stream, "wb") as wf:
            wf.setnchannels(_CHANNELS)
            wf.setsampwidth(self._pyaudio.get_sample_size(_FORMAT))
            wf.setframerate(_RATE)
            wf.writeframes(b"".join(voiced_frames))

    def record(self) -> bytes: 
        byte_stream = io.BytesIO()
        self._record_with_vad(byte_stream)
        return byte_stream.getvalue()

    def record_and_save(self, file_path):
        return self._record_with_vad(file_path)

def record() -> bytes:
    recorder = Recorder()
    return recorder.record()


def record_and_save():
    recorder = Recorder()
    return recorder.record_and_save("vad_recording.wav")

if __name__ == "__main__":
    record_and_save()