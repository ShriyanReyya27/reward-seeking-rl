#!/bin/bash
# Memory-aware queue: start each queued command only when >= 1.6 GB is available, 90 s apart.
cd "$(dirname "$0")"; source ../keys/tk_env.sh; source ../keys/oa_env.sh
while read -r cmd; do
  [ -z "$cmd" ] && continue
  until [ "$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)" -ge 1600 ]; do sleep 60; done
  echo "$(date +%H:%M) start: $cmd"; bash -c "$cmd" &
  sleep 90
done < "$1"
wait
