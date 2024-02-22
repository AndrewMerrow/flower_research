#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

#Start the poisoned clients
for i in `seq 0 15`; do
    echo "Starting client $i"
    python client.py --clientID $i --data "fedemnist" &
done

for i in `seq 16 31`; do
    echo "Starting client $i"
    python client2.py --clientID $i --data "fedemnist" &
done

for i in `seq 32 48`; do
    echo "Starting client $i"
    python client3.py --clientID $i --data "fedemnist" &
done

for i in `seq 49 65`; do
    echo "Starting client $i"
    python client4.py --clientID $i --data "fedemnist" &
done


# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait