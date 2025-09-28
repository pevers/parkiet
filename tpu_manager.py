"""
Simple TPU Manager - Keep retrying to create v4-16 TPU until capacity is available
"""

import subprocess
import time
import json

TPU_CONFIG = {"type": "v5p-16", "zone": "us-east5-a", "spot": False}


def run_gcloud_command(command: list[str]) -> dict:
    """Run a gcloud command and return parsed JSON output."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            return json.loads(result.stdout)
        return {}
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(command)}")
        print(f"Error: {e.stderr}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output: {e}")
        return {}


def create_tpu(name: str) -> bool:
    """Attempt to create the v4-16 TPU."""
    print(f"Attempting to create TPU: {name}")
    print(f"  Type: {TPU_CONFIG['type']}")
    print(f"  Zone: {TPU_CONFIG['zone']}")
    print(f"  Spot: {TPU_CONFIG['spot']}")

    command = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "create",
        name,
        f"--zone={TPU_CONFIG['zone']}",
        f"--accelerator-type={TPU_CONFIG['type']}",
        "--version=tpu-ubuntu2204-base",
        "--format=json",
    ]

    result = run_gcloud_command(command)

    if result:
        print(f"✓ Successfully created TPU: {name}")
        return True
    else:
        print(f"✗ Failed to create TPU: {name}")
        return False


def wait_for_tpu_ready(name: str, timeout: int = 300) -> bool:
    """Wait for TPU to become ready."""
    print(f"Waiting for TPU {name} to become ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        command = [
            "gcloud",
            "compute",
            "tpus",
            "tpu-vm",
            "describe",
            name,
            f"--zone={TPU_CONFIG['zone']}",
            "--format=json",
        ]

        tpu_info = run_gcloud_command(command)
        if tpu_info.get("state") == "READY":
            print(f"✓ TPU {name} is now ready!")
            return True

        print(".", end="", flush=True)
        time.sleep(10)

    print(f"\n✗ Timeout waiting for TPU {name} to become ready")
    return False


def main():
    """Main function - keep retrying until TPU is created successfully."""
    attempt = 1
    retry_delay = 30  # seconds

    while True:
        print(f"\n{'=' * 60}")
        print(f"TPU Creation Attempt #{attempt}")
        print(f"Target: {TPU_CONFIG['type']} in {TPU_CONFIG['zone']}")
        print(f"{'=' * 60}")

        tpu_name = f"{TPU_CONFIG['type']}-tpu-{int(time.time())}"

        if create_tpu(tpu_name):
            print("\n🎉 TPU creation request successful!")

            if wait_for_tpu_ready(tpu_name):
                print(f"\n🎉 TPU {tpu_name} is ready and available!")
                print(
                    f"SSH command: gcloud compute tpus tpu-vm ssh {tpu_name} --zone={TPU_CONFIG['zone']}"
                )
                break
            else:
                print("\n⚠️ TPU created but not ready within timeout")

        print(
            f"\nAttempt #{attempt} failed. Waiting {retry_delay} seconds before retry..."
        )
        time.sleep(retry_delay)
        attempt += 1

        # Increase delay slightly after every 5 attempts to avoid overwhelming the API
        if attempt % 5 == 0:
            retry_delay = min(retry_delay + 30, 300)  # Cap at 5 minutes
            print(f"Increasing retry delay to {retry_delay} seconds")


if __name__ == "__main__":
    main()
