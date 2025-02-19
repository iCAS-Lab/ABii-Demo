# Edge Facial Expression Recognition

docker run --privileged --device /dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY -v /dev/bus/usb:/dev/bus/usb --name coral --hostname coral -ti s7117/ubuntu-coraltpu:arm

# Starting after first time

xhost + local:docker
docker start -ai coral
sudo DISPLAY=:0 python3 main.py
