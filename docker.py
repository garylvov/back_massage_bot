#!/usr/bin/env python3
# Copyright (c) 2025, Gary Lvov, Vinay Balaji, Tim Bennet, Xandar Ingare, Ben Yoon
# All rights reserved.
#
# SPDX-License-Identifier: MIT

import argparse
import glob
import grp
import json
import os
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto


class DockerError(Exception):
    """Base exception for docker-related errors"""

    pass


class CommandStatus(Enum):
    """Enum for command execution status"""

    SUCCESS = auto()
    INTERRUPTED = auto()
    FAILED = auto()

    @classmethod
    def is_successful(cls, code: int) -> bool:
        """Check if return code indicates success"""
        return code in [0, 130]  # Normal exit or interrupt


@dataclass
class DockerConfig:
    """Configuration for docker container"""

    container: str
    cuda: bool = False
    x11: bool = False
    interactive: bool = False
    devices: str | None = None
    volumes: str | None = None
    use_entrypoint: bool = False
    endpoint: str = "bash"


class DockerManager:
    """Manages docker container operations"""

    def __init__(self):
        self.parent_dir = os.path.dirname(os.path.abspath(__file__))

    def get_running_containers(self) -> list[dict[str, str]]:
        """Get list of running containers with their details"""
        try:
            cmd = ["docker", "ps", "--format", "{{json .}}"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            return [json.loads(line) for line in result.stdout.strip().split("\n") if line]
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to get running containers:\n{e}")
            return []
        except Exception as e:
            print(f"\n❌ Unexpected error getting containers:\n{e}")
            return []

    def select_container(self) -> str | None:
        """Display running containers and let user select one"""
        containers = self.get_running_containers()

        if not containers:
            print("❌ No running containers found! Run build.sh && develop.sh .")
            return None

        print("\n🐋 Running containers:")
        for idx, container in enumerate(containers, 1):
            print(f"{idx}. {container['Names']} ({container['Image']})")
            print(f"   ID: {container['ID']}")
            print(f"   Status: {container['Status']}")
            print()

        while True:
            try:
                choice = input("\nEnter container number to dive into (or 'q' to quit): ").strip()
                if choice.lower() == "q":
                    return None

                idx = int(choice) - 1
                if 0 <= idx < len(containers):
                    return containers[idx]["ID"]
                print("❌ Invalid container number, please try again")
            except (ValueError, IndexError):
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n🔄 Operation cancelled")
                return None

    def dive_into_container(self, container_id: str) -> bool:
        """Execute docker exec to enter running container"""
        cmd = ["docker", "exec", "-it", container_id, "bash"]
        try:
            returncode = subprocess.call(cmd)
            return CommandStatus.is_successful(returncode)
        except KeyboardInterrupt:
            print("\n🔄 Exiting container...")
            return True
        except subprocess.CalledProcessError:
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error entering container:\n{e}")
            return False

    def _parse_devices(self, devices: str | None) -> Sequence[str]:
        """Convert device string into docker device arguments"""
        if not devices:
            return []
        return [arg for device in devices.split(",") for arg in ["--device", f"{device}:{device}"]]

    def _detect_video_devices(self) -> Sequence[str]:
        """Detect video devices and prepare them for docker with proper group permissions"""
        video_devices = []
        found_devices = []  # For debug printing

        try:
            # Get the video group ID
            video_group_id = grp.getgrnam("video").gr_gid

            # List of device patterns to check
            device_patterns = [
                ("/dev/video*", range(100)),  # video0-100
            ]

            # Check each device pattern
            for pattern, range_obj in device_patterns:
                if range_obj is not None:
                    # Numbered devices
                    for i in range_obj:
                        device = pattern.replace("*", str(i))
                        if os.path.exists(device):
                            video_devices.extend(["--device", f"{device}:{device}", "--group-add", str(video_group_id)])
                            found_devices.append(device)
                else:
                    matching_devices = glob.glob(pattern)
                    for device in matching_devices:
                        video_devices.extend(["--device", f"{device}:{device}", "--group-add", str(video_group_id)])
                        found_devices.append(device)

            # Print found devices
            if found_devices:
                video_devices.append("-v")
                video_devices.append("/dev:/dev")
                video_devices.append("--device /dev/bus/usb")
                print("\n🎥 Found video devices:")
                for device in sorted(found_devices):
                    print(f"   - {device}")
            else:
                print("No video devices found")

        except KeyError:
            print("Warning: 'video' group not found on system")
        except Exception as e:
            print(f"Warning: Error detecting video devices: {e}")

        return video_devices

    def _detect_devices(self) -> Sequence[str]:
        """Detect video and other devices and prepare them for docker with proper group permissions"""
        device_args = []
        found_devices = []  # For debug printing

        try:
            # Get the required group IDs
            video_group_id = grp.getgrnam("video").gr_gid
            try:
                dialout_group_id = grp.getgrnam("dialout").gr_gid
            except KeyError:
                print("Warning: 'dialout' group not found on system")
                dialout_group_id = None

            # List of device patterns to check
            device_patterns = [
                ("/dev/video*", range(100)),  # video0-100
                ("/dev/ttyUSB*", range(10)),  # USB serial devices
            ]

            # Check for Kinova device
            try:
                lsusb_output = subprocess.check_output(["lsusb"], text=True)
                if "Kinova" in lsusb_output:  # Look for any Kinova device
                    device_args.extend(["--group-add", str(dialout_group_id)] if dialout_group_id else [])
                    device_args.extend(["--device", "/dev/bus/usb", "-v", "/dev/bus/usb:/dev/bus/usb"])
                    found_devices.append("Kinova Robotic Platform (USB)")
            except subprocess.CalledProcessError:
                print("Warning: Could not check for Kinova device")

            # Check for ESP32 Device
            try:
                lsusb_output = subprocess.check_output(["lsusb"], text=True)
                if "Espressif" in lsusb_output:  # Look for any ESP32 device
                    device_args.extend(["--group-add", str(dialout_group_id)] if dialout_group_id else [])
                    # The ESP32 is a ttyACM device, which we cannot mount directly
                    # Instead, we mount the entire /dev/:dev directory
                    device_args.extend([
                        "--device",
                        "/dev/bus/usb",
                        "-v",
                        "/dev/bus/usb:/dev/bus/usb",
                        "--privileged",
                        "-v",
                        "/dev:/dev",
                    ])
            except subprocess.CalledProcessError:
                print("Warning: Could not check for ESP32 device")

            # Check each device pattern
            for pattern, range_obj in device_patterns:
                if range_obj is not None:
                    # Numbered devices
                    for i in range_obj:
                        device = pattern.replace("*", str(i))
                        if os.path.exists(device):
                            device_args.extend(["--device", f"{device}:{device}", "--group-add", str(video_group_id)])
                            found_devices.append(device)
                else:
                    matching_devices = glob.glob(pattern)
                    for device in matching_devices:
                        device_args.extend(["--device", f"{device}:{device}", "--group-add", str(video_group_id)])
                        found_devices.append(device)

            # Print found devices and add general device access
            if found_devices:
                device_args.extend(["-v", "/dev:/dev"])
                print("\n🔍 Found devices:")
                for device in sorted(found_devices):
                    print(f"   - {device}")
            else:
                print("No devices found")

        except KeyError:
            print("Warning: 'video' group not found on system")
        except Exception as e:
            print(f"Warning: Error detecting devices: {e}")

        return device_args

    def _parse_volumes(self, volumes: str | None) -> Sequence[str]:
        """Convert volume string into docker volume arguments"""
        if not volumes:
            return []
        return [
            arg for volume in volumes.split(",") for arg in ["-v", f"{os.path.join(self.parent_dir, volume)}:/{volume}"]
        ]

    def build_docker_command(self, config: DockerConfig) -> list[str]:
        """Build docker command with specified options"""
        cmd = ["docker", "run", "--rm", "--net=host", "--ipc=host", "--user", "$(id -u):$(id -g)"]
        cmd.extend(self._detect_devices())
        if config.interactive:
            cmd.append("-it")

        if config.x11:
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

        cmd.extend(self._parse_devices(config.devices))
        cmd.extend(self._parse_volumes(config.volumes))

        if config.cuda:
            cmd.extend(["-e", "NVIDIA_DRIVER_CAPABILITIES=all", "--runtime=nvidia", "--gpus", "all"])

        if config.use_entrypoint:
            cmd.extend(["--entrypoint", "./entrypoint.sh", config.container])
        else:
            cmd.extend([config.container, config.endpoint])

        return cmd

    def run_docker_command(self, cmd: list[str], use_entrypoint: bool = False) -> bool:
        """Run docker command and return True if successful"""
        cmd_str = " ".join(cmd)
        print(f"🐋 Executing command: {cmd_str}")

        try:
            if use_entrypoint:
                return self._run_with_entrypoint(cmd_str)
            return self._run_interactive(cmd_str)
        except Exception as e:
            print(f"\n❌ Unexpected error running docker command:\n{e}")
            return False

    def _run_with_entrypoint(self, cmd_str: str) -> bool:
        """Run docker command with entrypoint and stream output"""
        process = subprocess.Popen(
            cmd_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        output_received = False
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                output_received = True
                print(output.rstrip())

        returncode = process.poll()

        if not output_received and returncode != 0:
            self._print_error_help()
            print(f"\nExit code: {returncode}")

        return CommandStatus.is_successful(returncode)

    def _run_interactive(self, cmd_str: str) -> bool:
        """Run docker command in interactive mode"""
        returncode = subprocess.call(cmd_str, shell=True)
        if not CommandStatus.is_successful(returncode):
            print(f"\n❌ Docker command failed with exit code: {returncode}")
            return False
        return True

    @staticmethod
    def _print_error_help() -> None:
        """Print helpful error message when command fails without output"""
        print("\n❌ Docker command failed without output. Possible issues:")
        print("   - Container might not exist")
        print("   - entrypoint.sh might not be executable")
        print("   - Docker daemon might not be running")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run docker container with configurable options")
    parser.add_argument("--container", "-c", default="tt_ubuntu22_humble", help="Name of the container to run")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run container in interactive mode")
    parser.add_argument("-e", "--entrypoint", action="store_true", help="Use /entrypoint.sh as the entrypoint")
    parser.add_argument("--devices", help="Comma-separated list of devices to mount (e.g., /dev/ttyUSB0)")
    parser.add_argument("-v", "--volumes", help="Comma-separated list of paths to mount relative to TTECHDIR")
    parser.add_argument("--endpoint", default="bash", help="Container endpoint/command (default: bash)")
    parser.add_argument("--dive", action="store_true", help="List running containers and dive into selected one")

    args = parser.parse_args()
    docker_manager = DockerManager()

    if args.dive:
        container_id = docker_manager.select_container()
        if container_id:
            docker_manager.dive_into_container(container_id)
        return

    if not args.container:
        parser.error("container name is required when not using --dive")

    # Try different configurations in order of preference
    configs = [
        DockerConfig(  # Try with CUDA first
            container=args.container,
            cuda=True,
            x11=True,
            interactive=args.interactive,
            devices=args.devices,
            volumes=args.volumes,
            use_entrypoint=args.entrypoint,
            endpoint=args.endpoint,
        ),
        DockerConfig(  # Try without CUDA
            container=args.container,
            x11=True,
            interactive=args.interactive,
            devices=args.devices,
            volumes=args.volumes,
            use_entrypoint=args.entrypoint,
            endpoint=args.endpoint,
        ),
        DockerConfig(  # Try without X11
            container=args.container,
            interactive=args.interactive,
            devices=args.devices,
            volumes=args.volumes,
            use_entrypoint=args.entrypoint,
            endpoint=args.endpoint,
        ),
    ]

    messages = [
        "🔮 Attempting to cast the docker spell with GPUs enabled via CUDA...",
        "\n🔮 CUDA enchantment failed, trying without it...\n🐋 Casting the basic docker spell (no gpus :( )...",
        "\n🔮 Basic spell failed, trying one last time without X11...",
    ]

    for config, message in zip(configs, messages):
        print(message)
        cmd = docker_manager.build_docker_command(config)
        if docker_manager.run_docker_command(cmd, args.entrypoint):
            return


if __name__ == "__main__":
    main()
