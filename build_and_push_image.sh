#!/usr/bin/env bash

# Default is to use timestamp as tag
TAG=$(date +%Y%m%d_%H%M%S)
IMAGE_NAME="garylvov/back_massage_bot"

# Function to display usage information
show_usage() {
  echo "Usage: $(basename $0) [options]"
  echo
  echo "Build, tag, and push Docker image for back_massage_bot"
  echo
  echo "Options:"
  echo "  -s              Use 'stable' as the tag"
  echo "  -t TAG          Use custom tag"
  echo "  -h              Display this help message"
  echo
  echo "Examples:"
  echo "  $(basename $0)                   # Uses timestamp as tag (e.g. 20250322_142530)"
  echo "  $(basename $0) -s                # Uses 'stable' as tag"
  echo "  $(basename $0) -t v1.2.3         # Uses 'v1.2.3' as tag"
  echo "  $(basename $0) -t experimental   # Uses 'experimental' as tag"
}

# Process command line options
while getopts "st:h" opt; do
  case $opt in
    s)
      # Use "stable" as the tag
      TAG="stable"
      ;;
    t)
      # Use custom tag provided as argument
      TAG=$OPTARG
      ;;
    h)
      show_usage
      exit 0
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      show_usage
      exit 1
      ;;
  esac
done

echo "Building image with tag: $TAG"

# Run the build script
bash build.sh

# Tag and push the image
docker tag bmb_ubuntu22_humble:latest ${IMAGE_NAME}:${TAG}
docker push ${IMAGE_NAME}:${TAG}

echo "Successfully built and pushed ${IMAGE_NAME}:${TAG}"