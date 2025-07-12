import io
import pyaudio
import webrtcvad
import wave
from collections import deque

def _record_with_vad(file_path_or_stream):
    """
    Records audio from the microphone, starting when speech is detected
    and stopping after a period of silence.
    """
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # VAD requires 8000, 16000, 32000, or 48000 Hz
    CHUNK_DURATION_MS = 30  # VAD supports 10, 20, or 30 ms chunks
    CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
    PADDING_DURATION_MS = 1000  # Add 1 second of padding before and after speech
    NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
    SILENCE_TO_END_MS = 2000 # Stop recording after 2 seconds of silence
    NUM_SILENCE_CHUNKS_TO_END = int(SILENCE_TO_END_MS / CHUNK_DURATION_MS)

    vad = webrtcvad.Vad(3)  # Set aggressiveness mode (0-3, 3 is most aggressive)

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK_SIZE)

    print("Listening...")

    ring_buffer = deque(maxlen=NUM_PADDING_CHUNKS)
    triggered = False
    voiced_frames = []
    silent_chunks = 0

    while True:
        frame = stream.read(CHUNK_SIZE)
        is_speech = vad.is_speech(frame, RATE)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                print("Speech detected, starting recording.")
                voiced_frames.extend([f for f, s in ring_buffer])
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                silent_chunks += 1
                if silent_chunks > NUM_SILENCE_CHUNKS_TO_END:
                    print("Silence detected, stopping recording.")
                    break
            else:
                silent_chunks = 0

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recording
    with wave.open(file_path_or_stream, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(voiced_frames))

def record_and_return_with_vad() -> bytes:
    byte_stream = io.BytesIO()
    _record_with_vad(byte_stream)
    return byte_stream.getvalue()


def record_and_save_with_vad():
    return _record_with_vad("vad_recording.wav")

if __name__ == "__main__":
    record_and_save_with_vad()