FROM garylvov/back_massage_bot:stable
# This is built on top of what's deployed from ws/src/main_ros/Dockerfile
# It copies what's in the local dirs into the image ;)
# This so that builds can cache from the existing image to save everyone time
# Everything in this image should eventually be copied into the ws/src/main_ros/Dockerfile
# And then pushed to the Dockerhub.

# This can be done like:
# Copy the things from this Dockerfile into the ws/src/main_ros/Dockerfile
# Then run:
# bash ws/src/main_ros/build.sh
# docker tag bmb_ubuntu22_humble:latest garylvov/back_massage_bot:stable && docker push garylvov/back_massage_bot:stable

SHELL ["conda", "run", "-n", "back_massage_bot", "/bin/bash", "-c"]
ARG USERNAME
USER ${USERNAME}
WORKDIR /ws/src/main_ros

# Install vim as en example of how to install new things on top of the base image.
RUN sudo apt-get install -y vim

# Alternatively, you could install other things here.
# You can even use root if you'd like.
# USER root
# SHELL ["/bin/bash", "-c"]

# You may want to chown files:
# COPY --chown=${USERNAME}:${USERNAME} --chmod=777 back_massage_bot /back_massage_bot
