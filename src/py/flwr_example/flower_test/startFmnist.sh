#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
#python -c "from torchvision.datasets import CIFAR10; CIFAR10('./dataset', download=True)"

#Start the poisoned clients
for i in `seq 0 0`; do
    echo "Starting poison client $i"
    python client.py --poison POISON --clientID $i --use_cuda True --data "fmnist" &
done

#Distrubute the remaining clients accross the 4 GPUs
for i in `seq 1 9`; do
    echo "Starting client $i"
    python client.py --clientID $i --use_cuda True --data "fmnist" &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait