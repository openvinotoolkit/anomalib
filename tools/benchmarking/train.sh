#!/bin/bash

# Kill all subprocesses and exit when ctrl-c is pressed
function stop_all() {
        echo "Stopping processes"
        pkill top
        pkill pcm
        tokill=`ps aux | grep tools/train.py | grep -v grep | head -n 1 | awk '{print $2}'`
	kill "${tokill}"
}

trap "echo; echo Killing processes...; stop_all; exit 0" SIGINT SIGTERM

#Install PCM if it is not already
if [ -d "./pcm" ]
then
    echo PCM installed
    modprobe msr
else
    git clone https://github.com/opcm/pcm.git
    pushd pcm
    make
    make install
    apt install sysstat
    modprobe msr
    popd
fi

#Creat output dir
DATETIME=`date -u +"%Y%b%d_%H%M"`
mkdir -p output/$DATETIME
rundir="$PWD/output/$DATETIME"
echo ${rundir}

#Run training
pushd ..
python tools/train.py "$@" 2>&1 | tee -a ${rundir}/train.log &
popd
sleep 10

#Collect system-level metrics
top -b -i -o %CPU > "${rundir}"/top.log &
echo TOP STARTED
./pcm/pcm.x > "${rundir}"/pcm.log &
echo PCM STARTED

echo NVIDIA LOOP STARTING
for i in `seq 1 60`
do
	nvidia-smi --query-gpu=utilization.gpu --format=csv >> "${rundir}"/gpu_utilization.log
	nvidia-smi --query-gpu=utilization.memory --format=csv >> "${rundir}"/gpu_mem.log
	nvidia-smi --query-gpu=temperature.gpu --format=csv >> "${rundir}"/gpu_temp.log
	sleep 1
done

#Cleanup when done
pkill top
pkill pcm
