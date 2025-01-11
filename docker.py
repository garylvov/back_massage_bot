#!/usr/bin/env python3
import argparse
import os
import subprocess

def parse_devices(devices: str | None) -> list[str]:
    """Convert device string into docker device arguments"""
    if not devices:
        return []
    return [arg for device in devices.split(',') 
            for arg in ['--device', f'{device}:{device}']]

def parse_volumes(volumes: str | None) -> list[str]:
    """Convert volume string into docker volume arguments"""
    if not volumes:
        return []
    ttechdir = os.environ.get('TTECHDIR', '')
    return [arg for volume in volumes.split(',')
            for arg in ['-v', f'{os.path.join(ttechdir, volume)}:{volume}']]

def run_docker_command(cmd: list[str]) -> bool:
    """Run docker command and return True if successful"""
    cmd_str: str = " ".join(cmd)
    try:
        subprocess.run(cmd_str, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nDocker command failed with error:\n{e}")
        return False

def build_docker_command(
    container: str,
    *,
    cuda: bool = False,
    x11: bool = False,
    interactive: bool = False,
    devices: str | None = None,
    volumes: str | None = None,
    endpoint: str = "bash",
) -> list[str]:
    """Build docker command with specified options"""
    cmd: list[str] = ["docker", "run", "--rm", "--net=host", "--ipc=host"]
    
    # Interactive mode
    if interactive:
        cmd.append("-it")
    
    # X11 forwarding
    if x11:
        cmd.extend([
            "xhost", "+local", "&&",
            "-e", "DISPLAY",
            "-v", f"{os.environ['HOME']}/.Xauthority:/home/randuser/.Xauthority",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix"
        ])
    
    # Device mappings
    if devices:
        cmd.extend(parse_devices(devices))
    
    # Volume mounts
    if volumes:
        cmd.extend(parse_volumes(volumes))
    
    # CUDA support
    if cuda:
        cmd.extend(["--runtime=nvidia", "--gpus", "all"])
    
    # Container and endpoint
    cmd.extend([container, endpoint])
    
    return cmd

def main() -> None:
    parser = argparse.ArgumentParser(description='Run docker container with configurable options')
    parser.add_argument('container', help='Name of the container to run')
    parser.add_argument('--x11', action='store_true', help='Enable X11 forwarding')
    parser.add_argument('-i', '--interactive', action='store_true', help='Run container in interactive mode')
    parser.add_argument('--devices', help='Comma-separated list of devices to mount (e.g., /dev/ttyUSB0,/dev/ttyUSB1)')
    parser.add_argument('--volumes', help='Comma-separated list of paths to mount relative to TTECHDIR (e.g., projects/foo,projects/bar)')
    parser.add_argument('--endpoint', default='bash', help='Container endpoint/command (default: bash)')

    args: argparse.Namespace = parser.parse_args()

    # First try with CUDA
    print("🚀 Attempting to cast the docker spell with CUDA enchantment...")
    cmd = build_docker_command(
        container=args.container,
        cuda=True,
        x11=args.x11,
        interactive=args.interactive,
        devices=args.devices,
        volumes=args.volumes,
        endpoint=args.endpoint,
    )
    if run_docker_command(cmd):
        return

    print("\n🔮 CUDA enchantment failed, trying without it...")

    # Try without CUDA
    print("🐋 Casting the basic docker spell...")
    cmd = build_docker_command(
        container=args.container,
        cuda=False,
        x11=args.x11,
        interactive=args.interactive,
        devices=args.devices,
        volumes=args.volumes,
        endpoint=args.endpoint,
    )
    run_docker_command(cmd)

if __name__ == "__main__":
    main()
