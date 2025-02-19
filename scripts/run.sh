#!/bin/bash
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
xinit &
docker exec -ti abii_demo DISPLAY=:0 python src/main.py &