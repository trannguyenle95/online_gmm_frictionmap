FROM osrf/ros:noetic-desktop

# Add vscode user with same UID and GID as your host system
# (copied from https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user#_creating-a-nonroot-user)
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
# Switch from root to user
USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install Git
RUN sudo apt install -y git

# Install Python + others
RUN sudo apt-get update
RUN sudo apt-get install -y software-properties-common
RUN sudo add-apt-repository -y ppa:deadsnakes/ppa
    ## Install py39 from deadsnakes repository
RUN sudo apt-get install -y python3.9
    ## Install pip from standard ubuntu packages
RUN sudo apt-get install -y python3-pip
RUN pip3 install numpy==1.24.4 scipy==1.10.1 pandas==2.0.3 matplotlib==3.7.5 rosnumpy
# Rosdep update
RUN rosdep update

# Source the ROS setup file
# RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
RUN echo "source /online_friction/devel/setup.bash" >> ~/.bashrc

# Install Micro editor
RUN sudo apt install -y micro

# Install dependencies for online_friction
RUN sudo apt-get install -y ros-noetic-jsk-pcl-ros
RUN sudo apt-get install -y ros-noetic-jsk-visualization