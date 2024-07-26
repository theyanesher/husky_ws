# Husky Workspace

### Barebones Robot Setup

1. Clone the repo and build
    ```
    git clone https://github.com/RoboticsIIITH/husky_ws.git
    cd husky_ws
    catkin_make
    ```

2. (Optional) For Joystick control
   
   Add yourself to `tty` group and set permission for USB device access. 
   Note: Do this when you are connected to Husky, husky will read host at /dev/ttyUSB0. 
   ```
   sudo usermod -a -G tty <username>
   sudo chmod a+x /dev/ttyUSB0
   ```

3. Launch
   ```
   roslaunch husky_base base.launch port:=/dev/ttyUSB0
   ```





