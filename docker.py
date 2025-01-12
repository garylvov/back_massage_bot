#!/usr/bin/env python3
# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import argparse
import os
import subprocess


def parse_devices(devices: str | None) -> list[str]:
    """Convert device string into docker device arguments"""
    if not devices:
        return []
    return [arg for device in devices.split(",") for arg in ["--device", f"{device}:{device}"]]


def parse_volumes(volumes: str | None) -> list[str]:
    """Convert volume string into docker volume arguments"""
    if not volumes:
        return []
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    return [arg for volume in volumes.split(",") for arg in ["-v", f"{os.path.join(parent_dir, volume)}:/{volume}"]]


def run_docker_command(cmd: list[str], use_entrypoint: bool = False) -> bool:
    """Run docker command and return True if successful"""
    cmd_str: str = " ".join(cmd)
    print(f"🐋 Executing command: {cmd_str}")

    try:
        if use_entrypoint:
            # For entrypoint, stream output in real-time
            process = subprocess.Popen(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,  # Line buffered
            )

            # Stream output in real-time
            output_received = False
            while True:
                output = process.stdout.readline()
                if output == "" and process.poll() is not None:
                    break
                if output:
                    output_received = True
                    print(output.rstrip())

            returncode = process.poll()

            # If no output was received and command failed, print helpful message
            if not output_received and returncode != 0:
                print("\n❌ Docker command failed without output. Possible issues:")
                print("   - Container might not exist")
                print("   - entrypoint.sh might not be executable")
                print("   - Docker daemon might not be running")
                print(f"\nExit code: {returncode}")

        else:
            # For interactive mode, execute directly without output capture
            # This allows proper handling of TTY and interactive sessions
            returncode = subprocess.call(cmd_str, shell=True)

        # Exit codes 130 (interrupt) and 0 (normal exit) are considered successful
        if returncode in [0, 130]:
            return True

        print(f"\n❌ Docker command failed with exit code: {returncode}")
        return False

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Docker command failed with error:\n{e}")
        if hasattr(e, "output") and e.output:
            print(f"\nOutput:\n{e.output}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error running docker command:\n{e}")
        return False


def build_docker_command(
    container: str,
    *,
    cuda: bool = False,
    x11: bool = False,
    interactive: bool = False,
    devices: str | None = None,
    volumes: str | None = None,
    use_entrypoint: bool = False,
    endpoint: str = "bash",
) -> list[str]:
    """Build docker command with specified options"""
    cmd: list[str] = ["docker", "run", "--rm", "--net=host", "--ipc=host", "--user", "$(id -u):$(id -g)"]

    # Interactive mode
    if interactive:
        cmd.append("-it")

    # X11 forwarding
    if x11:
        xauth_path = f"{os.environ['HOME']}/.Xauthority"
        user = os.environ["USER"]
        user_xauth = f"/home/{user}/.Xauthority"

        cmd.extend([
            "-e",
            "DISPLAY",
            "-e",
            "QT_X11_NO_MITSHM=1",
            "-e",
            f"XDG_RUNTIME_DIR=/tmp/runtime-{os.getuid()}",
            "-v",
            f"{xauth_path}:{user_xauth}",
            "-v",
            "/tmp/.X11-unix:/tmp/.X11-unix:rw",
        ])

    # Device mappings
    if devices:
        cmd.extend(parse_devices(devices))

    # Volume mounts
    if volumes:
        cmd.extend(parse_volumes(volumes))

    # CUDA support
    if cuda:
        cmd.extend(["-e", "NVIDIA_DRIVER_CAPABILITIES=all", "--runtime=nvidia", "--gpus", "all"])

    # Container and endpoint
    if use_entrypoint:
        cmd.extend(["--entrypoint", "./entrypoint.sh", container])
    else:
        cmd.extend([container, endpoint])

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run docker container with configurable options")
    parser.add_argument("container", help="Name of the container to run")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run container in interactive mode")
    parser.add_argument("-e", "--entrypoint", action="store_true", help="Use /entrypoint.sh as the entrypoint")
    parser.add_argument("--devices", help="Comma-separated list of devices to mount (e.g., /dev/ttyUSB0)")
    parser.add_argument(
        "-v",
        "--volumes",
        help="Comma-separated list of paths to mount relative to TTECHDIR (e.g., projects/foo,projects/bar)",
    )
    parser.add_argument("--endpoint", default="bash", help="Container endpoint/command (default: bash)")
    args: argparse.Namespace = parser.parse_args()

    # First try with CUDA
    print("🔮 Attempting to cast the docker spell with GPUs enabled via CUDA...")

    cmd = build_docker_command(
        container=args.container,
        cuda=True,
        x11=True,
        interactive=args.interactive,
        devices=args.devices,
        volumes=args.volumes,
        use_entrypoint=args.entrypoint,
        endpoint=args.endpoint,
    )
    if run_docker_command(cmd, args.entrypoint):
        return

    # Try without CUDA
    print("\n🔮 CUDA enchantment failed, trying without it...")
    print("🐋 Casting the basic docker spell (no gpus :( )...")
    cmd = build_docker_command(
        container=args.container,
        cuda=False,
        x11=True,
        interactive=args.interactive,
        devices=args.devices,
        volumes=args.volumes,
        use_entrypoint=args.entrypoint,
        endpoint=args.endpoint,
    )
    if run_docker_command(cmd, args.entrypoint):
        return

    print("\n🔮 Basic spell failed, trying one last time without X11...")
    cmd = build_docker_command(
        container=args.container,
        cuda=False,
        x11=False,
        interactive=args.interactive,
        devices=args.devices,
        volumes=args.volumes,
        use_entrypoint=args.entrypoint,
        endpoint=args.endpoint,
    )
    run_docker_command(cmd, args.entrypoint)


if __name__ == "__main__":
    main()
