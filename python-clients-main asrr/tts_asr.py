import os
import subprocess
import sys
import tempfile

# ---------------- CONFIG ----------------
TTS_TEXT = "Indeed! Our contemporaneous promotional vouchers are available in the ‘Offers’ segment. Apply them judiciously at checkout to optimize monetary conservation."  # Change this to your text
TTS_VOICE = "Magpie-Multilingual.EN-US.Aria"

# NVIDIA API Keys
TTS_API_KEY = "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"
ASR_API_KEY = "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"

# Function IDs
ASR_FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"

# Server
RIVA_SERVER = "grpc.nvcf.nvidia.com:443"

# ---------------- STEP 0: Auto-find talk.py ----------------
def find_talk_py(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if "talk.py" in files:
            return os.path.join(root, "talk.py")
    return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TALK_PY = find_talk_py(BASE_DIR)
if TALK_PY is None:
    print("ERROR: talk.py not found anywhere under", BASE_DIR)
    sys.exit(1)
print("Found talk.py at:", TALK_PY)

# ---------------- STEP 1: Generate TTS ----------------
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
    tts_output_path = tmp_file.name

tts_command = [
    sys.executable, TALK_PY,
    "--server", RIVA_SERVER,
    "--use-ssl",
    "--metadata", "function-id", "877104f7-e885-42b9-8de8-f6e4c6303969",
    "--metadata", "authorization", f"Bearer {TTS_API_KEY}",
    "--language-code", "en-US",
    "--text", TTS_TEXT,
    "--voice", TTS_VOICE,
    "--output", tts_output_path
]

print(f"Running TTS -> {tts_output_path}")
subprocess.run(tts_command, check=True)

# ---------------- STEP 2: Auto-find ASR script ----------------
def find_asr_script(base_dir):
    for root, dirs, files in os.walk(base_dir):
        if "transcribe_file.py" in files:
            return os.path.join(root, "transcribe_file.py")
    return None

ASR_SCRIPT = find_asr_script(BASE_DIR)
if ASR_SCRIPT is None:
    print("ERROR: transcribe_file.py not found anywhere under", BASE_DIR)
    sys.exit(1)
print("Found ASR script at:", ASR_SCRIPT)

# ---------------- STEP 3: Run ASR ----------------
asr_command = [
    sys.executable,
    ASR_SCRIPT,
    "--server", RIVA_SERVER,
    "--use-ssl",
    "--metadata", "function-id", ASR_FUNCTION_ID,
    "--metadata", "authorization", f"Bearer {ASR_API_KEY}",
    "--language-code", "en-US",
    "--input-file", tts_output_path
]

print("Running ASR...")
result = subprocess.run(asr_command, capture_output=True, text=True)

# ---------------- STEP 4: Output ----------------
print("\n=== ASR STDOUT ===")
print(result.stdout)
print("\n=== ASR STDERR ===")
print(result.stderr)
print("Return code:", result.returncode)

# ---------------- STEP 5: Clean up ----------------
os.remove(tts_output_path)
Hi! You can track your order in real-time from the ‘Orders’ section in the app. If your order is delayed, our support team will notify you with updates.
Hi You can track your order in real time from the Orders section in the app If your order is delayed our support team will notify you with updates