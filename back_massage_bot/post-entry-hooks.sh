#!/usr/bin/env bash
# If you modify me, your next docker build may take long!

# Activate pixi environment
export PATH="/home/${USER}/.pixi/bin:$PATH"
cd /back_massage_bot
pixi shell

echo "Activated Pixi environment."
