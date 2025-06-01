# Usage Instructions

### Prerequisites

- Docker Installation
  ```bash
  # Install Docker using convenience script
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh ./get-docker.sh

  # Post-install configuration
  sudo groupadd docker
  sudo usermod -aG docker $USER
  sudo systemctl enable docker.service
  sudo systemctl enable containerd.service

  # Verify installation
  sudo systemctl is-enabled docker
  ```

 **Reboot before proceeding further**

**GHCR Authentication** 
  ```bash
  echo "<YOUR_GITHUB_PAT>" | docker login ghcr.io -u <YOUR_GITHUB_USERID> --password-stdin
  ```

- VSCode
- Remote Development Extension by Microsoft (Inside VSCode)
  
### Setup Process
- Create a folder for Husky development
    ```bash 
    mkdir husky_ws && cd husky_ws
    # Clone the repo 
    git clone https://github.com/RoboticsIIITH/husky_ws.git .
    # Open VSCode 
    code .
    ```
- To enter the container
    - Open Command Pallete with `Ctrl+Shift+P`
    - Select **Dev Containers: Reopen in Container**

    - Use `Build WS` button to build workspace
  
  

## Start up the ROS

1. Launch
   ```
   sudo chmod +777 /dev/ttyUSB0
   roslaunch husky_base base.launch 
   ```
   - Use ```sudo apt remove brltty``` if /dev/tty/USB0 port is not visible after connecting to Husky.

   - Optionally, you can plug a joystick and teleop the robot.

## Docker

- To permanently add any ROS APT packages, list them in the rosPkgs.list file, then rebuild the Docker image using:
   ```
   docker build -t ghcr.io/rtarun1/husky_base -f .devcontainer/Dockerfile .devcontainer
   ```
- Always run ```sudo apt update``` inside the container before installing any additional packages.
  
- For Docker-related questions or issues, feel free to open an issue on the [DockerForROS2Development](https://github.com/soham2560/DockerForROS2Development.git)


### Credits

- This Docker setup was adapted from [Soham's repository](https://github.com/soham2560/DockerForROS2Development.git).
- If you use this repository for your project or publication, please consider citing or acknowledging [Tarun](https://github.com/rtarun1), [Soham](https://github.com/soham2560), [Official Husky](https://github.com/husky/husky).
