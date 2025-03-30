#!/bin/bash
srun -p A800 -N 1 -J saipv2test -n 8 --ntasks-per-node 8 --gres=gpu:a800:8 --cpus-per-task=7 --pty /bin/bash


# srun -p A800 -N 1 -n 1 --gres=gpu:a800:1 --cpus-per-task=6 --pty /bin/bash