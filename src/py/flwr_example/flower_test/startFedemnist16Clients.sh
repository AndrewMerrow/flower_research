#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
#python -c "from torchvision.datasets import CIFAR10; CIFAR10('./dataset', download=True)"

#Start the poisoned clients
for i in `seq 0 3`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "fedemnist" &
done

for i in `seq 4 7`; do
    echo "Starting client $i"
    python client2.py --clientID $i --data "fedemnist" &
done

for i in `seq 8 11`; do
    echo "Starting client $i"
    python client3.py --clientID $i --data "fedemnist" &
done

for i in `seq 12 15`; do
    echo "Starting client $i"
    python client4.py --clientID $i --data "fedemnist" &
done


# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait