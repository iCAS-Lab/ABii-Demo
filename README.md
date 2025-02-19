# Edge Facial Expression Recognition

docker run --privileged --device /dev/video0 --net=host --volume="$HOME/.Xauthority:/root/.Xauthority:rw" -v /tmp/.X11-unix -e DISPLAY -v /dev/bus/usb/:/dev/bus/usb --name coral --hostname coral -ti coral /bin/bash

# Starting after first time

xinit &
xhost + local:docker
docker start -ai coral
cd /home/user/2023-DAC-FER-DEMO/coraltpu_usb
python3 main.py
