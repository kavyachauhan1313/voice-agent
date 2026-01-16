import os
import subprocess
import sys
import tempfile
import difflib

# ---------------- CONFIG ----------------
# List of sentences to test
TEXTS = [
    "Hi You can track your order in real time from the Orders section in the app If your order is delayed our support team will notify you with updates",
]

TTS_VOICE = "Magpie-Multilingual.EN-US.Aria"

# NVIDIA API Keys
TTS_API_KEY = "nvapi-"
ASR_API_KEY = "nvapi-"

# Function IDs
ASR_FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"

# Server
RIVA_SERVER = "grpc.nvcf.nvidia.com:443"

# ---------------- HELPER FUNCTIONS ----------------
def find_file(base_dir, filename):
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None

def compute_wer(reference, hypothesis):
    """Compute Word Error Rate using simple difflib approach"""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    seq = difflib.SequenceMatcher(None, ref_words, hyp_words)
    return 1 - seq.ratio()

# ---------------- LOCATE SCRIPTS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TALK_PY = find_file(BASE_DIR, "talk.py")
ASR_SCRIPT = find_file(BASE_DIR, "transcribe_file.py")

if TALK_PY is None:
    print("ERROR: talk.py not found")
    sys.exit(1)
if ASR_SCRIPT is None:
    print("ERROR: transcribe_file.py not found")
    sys.exit(1)

print("Found talk.py at:", TALK_PY)
print("Found ASR script at:", ASR_SCRIPT)

# ---------------- RUN BATCH ----------------
for i, text in enumerate(TEXTS, 1):
    print(f"\n--- Processing sentence {i}/{len(TEXTS)} ---")
    print("Text:", text)

    # 1️⃣ Generate TTS
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tts_output_path = tmp_file.name

    tts_command = [
        sys.executable, TALK_PY,
        "--server", RIVA_SERVER,
        "--use-ssl",
        "--metadata", "function-id", "877104f7-e885-42b9-8de8-f6e4c6303969",
        "--metadata", "authorization", f"Bearer {TTS_API_KEY}",
        "--language-code", "en-US",
        "--text", text,
        "--voice", TTS_VOICE,
        "--output", tts_output_path
    ]
    subprocess.run(tts_command, check=True)

    # 2️⃣ Convert to 16kHz mono for ASR
    asr_ready_audio = tts_output_path.replace(".wav", "_asr.wav")
    subprocess.run([
        r"C:\ffmpeg\bin\ffmpeg.exe", "-y", "-i", tts_output_path, "-ar", "16000", "-ac", "1", asr_ready_audio
    ], check=True)

    # 3️⃣ Run ASR
    asr_command = [
        sys.executable, ASR_SCRIPT,
        "--server", RIVA_SERVER,
        "--use-ssl",
        "--metadata", "function-id", ASR_FUNCTION_ID,
        "--metadata", "authorization", f"Bearer {ASR_API_KEY}",
        "--language-code", "en-US",
        "--input-file", asr_ready_audio
    ]

    result = subprocess.run(asr_command, capture_output=True, text=True)
    transcription = result.stdout.strip()

    # 4️⃣ Compute WER
    wer = compute_wer(text, transcription)

    # 5️⃣ Print results
    print("ASR Transcription:", transcription)
    print(f"Word Error Rate (WER): {wer:.2%}")

    # 6️⃣ Clean up
    os.remove(tts_output_path)
    os.remove(asr_ready_audio)
