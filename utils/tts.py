from faster_whisper import WhisperModel
import numpy as np
model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")




def stt(pcm: bytes):
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    segments, info = model.transcribe(audio, beam_size=1, language="ru", vad_filter=True)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
