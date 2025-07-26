#!/bin/bash
LOCKFILE="/tmp/generate.lock"

flock -n "$LOCKFILE" bash -c 'source ~/git/stable-diffusion-sample/sd_env/bin/activate && python ~/git/stable-diffusion-sample/generate.py'

