"""
Script to start multiple chunker worker processes across all available GPUs.
Supports running 2 processes per GPU (suitable for A100).
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import psutil
import torch

# Add src to path so we can import parkiet modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parkiet.audioprep.chunker import ChunkerWorker


def get_available_gpus() -> list[int]:
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        print("CUDA not available, no GPUs detected")
        return []

    gpu_count = torch.cuda.device_count()
    available_gpus = []

    for gpu_id in range(gpu_count):
        try:
            # Test if GPU is accessible
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
            available_gpus.append(gpu_id)
        except Exception as e:
            print(f"GPU {gpu_id} not accessible: {e}")

    return available_gpus


def setup_logging(log_dir: Path, worker_id: int, gpu_id: int) -> str:
    """Setup logging for a worker process."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = f"worker_{worker_id}_gpu_{gpu_id}_{timestamp}.log"
    log_path = log_dir / log_filename

    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - Worker-%d-GPU-%d - %(levelname)s - %(message)s"
        % (worker_id, gpu_id),
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )

    return str(log_path)


def start_worker_process(
    worker_id: int,
    gpu_id: int,
    queue_name: str,
    whisper_checkpoint_path: str,
    temp_dir: str,
    max_workers: int,
    log_dir: Path,
    device: str = "cuda",
) -> subprocess.Popen:
    """Start a single worker process."""

    # Setup logging for this worker
    log_path = setup_logging(log_dir, worker_id, gpu_id)

    # Create worker-specific temp directory
    worker_temp_dir = f"{temp_dir}/worker_{worker_id}_gpu_{gpu_id}"

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "parkiet.audioprep.chunker",
        "worker",
        "--queue-name",
        queue_name,
        "--whisper-checkpoint-path",
        whisper_checkpoint_path,
        "--temp-dir",
        worker_temp_dir,
        "--gpu-id",
        str(gpu_id),
        "--max-workers",
        str(max_workers),
    ]

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["WORKER_ID"] = str(worker_id)
    env["GPU_ID"] = str(gpu_id)

    # Start process
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    print(f"Started worker {worker_id} on GPU {gpu_id} (PID: {process.pid})")
    print(f"Log file: {log_path}")

    return process


def monitor_processes(processes: list[subprocess.Popen], log_dir: Path):
    """Monitor worker processes and restart if they die."""
    print(f"Monitoring {len(processes)} worker processes...")
    print("Press Ctrl+C to stop all workers")

    try:
        while True:
            # Check if any processes have died
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    print(f"Worker process {i} (PID: {process.pid}) has died")
                    # For now, just log the death - you could add restart logic here

            time.sleep(5)  # Check every 5 seconds

    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down workers...")
        stop_all_workers(processes)


def stop_all_workers(processes: list[subprocess.Popen]):
    """Stop all worker processes gracefully."""
    print("Stopping all worker processes...")

    for i, process in enumerate(processes):
        if process.poll() is None:  # Process is still running
            print(f"Stopping worker {i} (PID: {process.pid})...")
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"Force killing worker {i} (PID: {process.pid})...")
                process.kill()
            except Exception as e:
                print(f"Error stopping worker {i}: {e}")

    print("All workers stopped")


def cleanup_zombie_processes():
    """Clean up any zombie processes from previous runs."""
    current_pid = os.getpid()

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info["cmdline"]
            if (
                cmdline
                and "parkiet.audioprep.chunker" in " ".join(cmdline)
                and proc.info["pid"] != current_pid
            ):
                print(
                    f"Found existing chunker process (PID: {proc.info['pid']}), terminating..."
                )
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except psutil.TimeoutExpired:
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Start multiple chunker worker processes across all available GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --processes-per-gpu 2
  %(prog)s --queue-name my_queue --processes-per-gpu 1
  %(prog)s --gpu-ids 0,1 --processes-per-gpu 2
        """,
    )

    parser.add_argument(
        "--queue-name",
        "-q",
        default="audio_processing",
        help="Redis queue name (default: audio_processing)",
    )

    parser.add_argument(
        "--whisper-checkpoint-path",
        "-c",
        default="pevers/whisperd-nl",
        help="Path to Whisper checkpoint (default: pevers/whisperd-nl)",
    )

    parser.add_argument(
        "--temp-dir",
        "-t",
        default="/tmp/parkiet_chunks",
        help="Base temporary directory for processing (default: /tmp/parkiet_chunks)",
    )

    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=8,
        help="Maximum number of parallel workers for audio extraction per process (default: 8)",
    )

    parser.add_argument(
        "--processes-per-gpu",
        "-p",
        type=int,
        default=2,
        help="Number of processes to run per GPU (default: 2, suitable for A100)",
    )

    parser.add_argument(
        "--gpu-ids",
        "-g",
        help="Comma-separated list of GPU IDs to use (default: all available GPUs)",
    )

    parser.add_argument(
        "--log-dir",
        "-l",
        default="logs/workers",
        help="Directory for log files (default: logs/workers)",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up existing chunker processes before starting",
    )

    args = parser.parse_args()

    # Clean up existing processes if requested
    if args.cleanup:
        cleanup_zombie_processes()

    # Determine which GPUs to use
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    else:
        gpu_ids = get_available_gpus()

    if not gpu_ids:
        print("No GPUs available, exiting")
        sys.exit(1)

    print(f"Using GPUs: {gpu_ids}")
    print(f"Starting {args.processes_per_gpu} processes per GPU")

    # Setup log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Start worker processes
    processes = []
    worker_id = 0

    for gpu_id in gpu_ids:
        for process_num in range(args.processes_per_gpu):
            process = start_worker_process(
                worker_id=worker_id,
                gpu_id=gpu_id,
                queue_name=args.queue_name,
                whisper_checkpoint_path=args.whisper_checkpoint_path,
                temp_dir=args.temp_dir,
                max_workers=args.max_workers,
                log_dir=log_dir,
            )
            processes.append(process)
            worker_id += 1

            # Small delay between starting processes to avoid resource conflicts
            time.sleep(2)

    print(f"Started {len(processes)} worker processes total")

    # Monitor processes
    monitor_processes(processes, log_dir)


if __name__ == "__main__":
    main()
