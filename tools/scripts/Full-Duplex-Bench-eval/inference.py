"""Full-Duplex Voice Agent Inference Client.

This client connects to a WebSocket-based voice agent server to process audio files
and generate responses. It handles audio preprocessing, streaming, and output generation.
"""

import argparse
import asyncio
import os
import time
import urllib.parse

import numpy as np
import resampy
import soundfile as sf
import websockets
from pipecat.frames.protobufs import frames_pb2

# Audio processing constants
SAMPLE_RATE = 16000  # Target sample rate in Hz
CHUNK_MS = 32  # Chunk duration in milliseconds
SILENCE_DUR = (
    2.0  # Silence duration to append after input audio file in seconds( to ensure end of utterance detection by ASR)
)
RECV_TIMEOUT = 5.0  # Timeout for receiving responses after input file ends in seconds


class InferenceClient:
    """Client for communicating with the voice agent WebSocket server."""

    def __init__(self, host: str, port: int):
        """Initialize the inference client.

        Args:
            host: Server hostname or IP address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.base_uri = f"ws://{host}:{port}/ws/benchmark"

    def preprocess_audio(self, audio_path: str) -> tuple[np.ndarray, float]:
        """Preprocess audio file to 16kHz mono linear PCM format.

        Args:
            audio_path: Path to input audio file

        Returns:
            Tuple of (preprocessed_audio, duration_in_seconds)
        """
        # Load audio file
        audio, sample_rate = sf.read(audio_path)

        # Convert to mono if stereo/multi-channel
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample to target sample rate if needed
        if sample_rate != SAMPLE_RATE:
            audio = resampy.resample(audio, sample_rate, SAMPLE_RATE)

        # Convert to linear PCM int16 format
        if audio.dtype != np.int16:
            if audio.dtype in [np.float32, np.float64]:
                # Clip and convert float to int16
                audio = np.clip(audio, -1.0, 1.0)
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)

        # Calculate duration in seconds
        duration = len(audio) / SAMPLE_RATE

        return audio, duration

    async def send_audio_stream(self, websocket, audio: np.ndarray):
        """Stream preprocessed audio to server in chunks with proper timing.

        Args:
            websocket: Active WebSocket connection
            audio: Preprocessed audio as int16 numpy array (16kHz, mono)
        """
        # Calculate chunk parameters
        chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
        chunk_duration = CHUNK_MS / 1000.0

        # Prepare silence buffer
        silence = np.zeros(chunk_samples, dtype=np.int16).tobytes()
        next_send_time = time.time()
        silence_start = None

        # Calculate total number of chunks
        total_samples = len(audio)
        current_idx = 0

        # Stream audio in chunks
        while True:
            # Maintain real-time streaming pace
            await asyncio.sleep(max(0, next_send_time - time.time()))

            # Check if we have more audio data
            if current_idx < total_samples:
                # Extract chunk from audio
                end_idx = min(current_idx + chunk_samples, total_samples)
                chunk = audio[current_idx:end_idx]

                # Pad last chunk if needed
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

                # Send audio chunk
                frame = frames_pb2.Frame(
                    audio=frames_pb2.AudioRawFrame(audio=chunk.tobytes(), sample_rate=SAMPLE_RATE, num_channels=1)
                )
                await websocket.send(frame.SerializeToString())

                current_idx = end_idx
            else:
                # Audio ended, start sending silence
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DUR:
                    break

                # Send silence chunk
                frame = frames_pb2.Frame(
                    audio=frames_pb2.AudioRawFrame(audio=silence, sample_rate=SAMPLE_RATE, num_channels=1)
                )
                await websocket.send(frame.SerializeToString())

            next_send_time += chunk_duration

    async def receive_audio_stream(
        self, websocket, start_time: float, send_task: asyncio.Task
    ) -> tuple[list[np.ndarray], list[float]]:
        """Receive audio chunks from server.

        After input is sent, wait 5 seconds of no output to confirm receiving is done.

        Args:
            websocket: Active WebSocket connection
            start_time: Timestamp when receiving started
            send_task: Task handle for the audio sending coroutine

        Returns:
            Tuple of (output_chunks, chunk_timestamps)
        """
        output_chunks = []
        chunk_times = []

        while True:
            try:
                # Wait for response with timeout
                response = await asyncio.wait_for(websocket.recv(), timeout=RECV_TIMEOUT)

                # Parse protobuf frame
                frame = frames_pb2.Frame.FromString(response)
                if frame.WhichOneof("frame") == "audio":
                    audio_data = frame.audio.audio
                    if not audio_data:
                        continue

                    # Extract audio chunk (skip WAV header if present)
                    if len(audio_data) > 44 and audio_data.startswith(b"RIFF"):
                        chunk = np.frombuffer(audio_data[44:], dtype=np.int16)
                    else:
                        chunk = np.frombuffer(audio_data, dtype=np.int16)

                    current_time = time.time() - start_time
                    output_chunks.append(chunk)

                    # Calculate chunk timestamp with smoothing
                    if not chunk_times:
                        chunk_times.append(current_time)
                    else:
                        # Use audio duration-based timing if close to previous timestamp
                        expected_time = chunk_times[-1] + len(output_chunks[-1]) / SAMPLE_RATE
                        if abs(current_time - chunk_times[-1]) < 0.05:
                            chunk_times.append(expected_time)
                        else:
                            chunk_times.append(current_time)

            except TimeoutError:
                # Timeout occurred - check if sending is complete
                if send_task.done():
                    # Input sending is complete and no output for 5 seconds, we're done
                    break
                else:
                    # Input still sending, continue waiting for output
                    continue
            except websockets.exceptions.ConnectionClosed:
                # Connection closed
                break

        return output_chunks, chunk_times

    def assemble_and_trim_output(
        self, output_chunks: list[np.ndarray], chunk_times: list[float], target_duration: float
    ) -> np.ndarray:
        """Assemble received audio chunks and trim to match input duration.

        Places chunks at their timestamps, inserting silence for gaps.

        Args:
            output_chunks: List of audio chunks received from server
            chunk_times: List of timestamps for each chunk
            target_duration: Target duration in seconds (input file duration)

        Returns:
            Assembled and trimmed audio as numpy array
        """
        if not output_chunks:
            return np.array([], dtype=np.int16)

        # Create output buffer for target duration
        target_samples = int(target_duration * SAMPLE_RATE)
        output = np.zeros(target_samples, dtype=np.int16)

        # Track where we expect the next chunk based on audio continuity
        next_expected_time = None

        for _i, (chunk, timestamp) in enumerate(zip(output_chunks, chunk_times, strict=False)):
            if len(chunk) == 0:
                continue

            chunk_duration = len(chunk) / SAMPLE_RATE
            start_sample = int(timestamp * SAMPLE_RATE)
            end_sample = start_sample + len(chunk)

            # Check if this chunk should be placed at its timestamp or sequentially
            if next_expected_time is not None:
                time_gap = timestamp - next_expected_time

                # If gap is larger than chunk duration, there's real silence
                # Place at timestamp (silence will be preserved)
                if time_gap > chunk_duration * 1.5:
                    # Use timestamp placement - silence gap exists
                    pass
                else:
                    # Chunks are continuous or nearly continuous
                    # Place sequentially to avoid glitches
                    start_sample = int(next_expected_time * SAMPLE_RATE)
                    end_sample = start_sample + len(chunk)

            # Ensure we don't exceed buffer
            if start_sample >= target_samples:
                break

            end_sample = min(end_sample, target_samples)
            chunk_to_write = chunk[: end_sample - start_sample]

            # Write chunk to output
            output[start_sample:end_sample] = chunk_to_write

            # Update expected time for next chunk
            next_expected_time = start_sample / SAMPLE_RATE + len(chunk_to_write) / SAMPLE_RATE

        return output

    async def process_single_file(self, input_path: str, output_path: str):
        """Process a single audio file through the voice agent.

        Args:
            input_path: Path to input audio file
            output_path: Path where output audio will be saved
        """
        # Preprocess input audio
        input_audio, input_duration = self.preprocess_audio(input_path)

        # Create unique WebSocket URI with file path as query parameter
        file_identifier = urllib.parse.quote(input_path, safe="")
        unique_uri = f"{self.base_uri}?file_path={file_identifier}"

        # Connect to server with unique URI
        async with websockets.connect(unique_uri) as websocket:
            # Start streaming input audio in real-time fashion
            send_task = asyncio.create_task(self.send_audio_stream(websocket, input_audio))

            # Receive output audio
            start_time = time.time()
            output_chunks, chunk_times = await self.receive_audio_stream(websocket, start_time, send_task)

            # Ensure send task is complete
            await send_task

        # Assemble received chunks and trim to input duration
        output_audio = self.assemble_and_trim_output(output_chunks, chunk_times, input_duration)

        sf.write(output_path, output_audio, SAMPLE_RATE)

    async def process_directory(self, input_dir: str, retry_samples: list[int] = None):
        """Process all audio files in a directory structure.

        Automatically detects and processes:
        - input.wav -> output.wav (always, if exists)
        - clean_input.wav -> clean_output.wav (if exists)

        Args:
            input_dir: Root directory containing sample subdirectories
            retry_samples: Optional list of specific sample IDs to process
        """
        # Get list of sample IDs to process
        if retry_samples:
            sample_ids = retry_samples
        else:
            sample_ids = sorted(
                [
                    int(name)
                    for name in os.listdir(input_dir)
                    if os.path.isdir(os.path.join(input_dir, name)) and name.isdigit()
                ]
            )

        # Process each sample
        for sample_id in sample_ids:
            sample_dir = os.path.join(input_dir, str(sample_id))

            # Define file pairs to process
            file_pairs = [("input.wav", "output.wav"), ("clean_input.wav", "clean_output.wav")]

            processed_count = 0
            for input_filename, output_filename in file_pairs:
                input_path = os.path.join(sample_dir, input_filename)
                output_path = os.path.join(sample_dir, output_filename)

                # Skip if input file doesn't exist
                if not os.path.exists(input_path):
                    continue

                print(f"Processing sample {sample_id}/{input_filename}...")
                try:
                    await self.process_single_file(input_path, output_path)
                    print(f"Successfully processed sample {sample_id}/{input_filename}")
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing sample {sample_id}/{input_filename}: {e}")

                # Brief pause between files
                await asyncio.sleep(1)

            if processed_count == 0:
                print(f"Warning: Skipped sample {sample_id}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Full-Duplex Voice Agent Inference Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all samples in a directory
  python client.py --input_dir /path/to/samples --host localhost --port 8100

  # Retry specific samples
  python client.py --input_dir /path/to/samples --host localhost --port 8100 --retry_samples 1 5 10

Note:
  The script automatically processes both input.wav and clean_input.wav if they exist
  in each sample directory, generating output.wav and clean_output.wav respectively.
        """,
    )

    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing sample subdirectories with audio files"
    )

    parser.add_argument("--host", type=str, required=True, help="Server hostname or IP address")

    parser.add_argument("--port", type=int, required=True, help="Server port number")

    parser.add_argument("--retry_samples", nargs="+", type=int, help="Specific sample IDs to process (optional)")

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()

    # Create client
    client = InferenceClient(host=args.host, port=args.port)

    print(f"Connecting to voice agent at ws://{args.host}:{args.port}/ws/benchmark")
    print(f"Processing directory: {args.input_dir}")

    # Process directory
    await client.process_directory(input_dir=args.input_dir, retry_samples=args.retry_samples)

    print("Processing complete!")


if __name__ == "__main__":
    asyncio.run(main())
