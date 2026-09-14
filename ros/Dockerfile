FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup \
    ros-jazzy-slam-toolbox \
    python3-requests \
    python3-opencv \
    python3-numpy \
    python3-yaml \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/milo