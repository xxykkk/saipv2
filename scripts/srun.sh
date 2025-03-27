#!/bin/bash
srun -p L40 -N 1 -J saipv2test -n 1 --ntasks-per-node 1 --gres=gpu:l40:1 --cpus-per-task=3 --pty /bin/bash


# srun -p A800 -N 1 -n 1 --gres=gpu:a800:1 --cpus-per-task=6 --pty /bin/bash