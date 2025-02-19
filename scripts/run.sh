#!/bin/bash
xinit &
export DISPLAY=:0
xhost + local:docker
docker start coral
docker exec -ti -w /home/user/2023-DAC-FER-DEMO/coraltpu_usb coral python3 main.py