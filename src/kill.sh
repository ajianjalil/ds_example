#!/bin/bash

# Function to kill processes based on a pattern
kill_processes() {
  local pattern=$1
  pids=$(ps -aux | grep "$pattern" | grep -v grep | awk '{print $2}')
  
  for pid in $pids; do
    kill -9 $pid
    echo "killed $pid"
  done
}

# Kill processes containing "driver"
kill_processes "driver"

# Kill processes containing "spawn"
kill_processes "spawn"

# Kill processes containing "python3 wrapper.py"
kill_processes "python3 wrapper.py"

# Kill processes containing "python3 driver31.py 0"
kill_processes "python3 driver31.py 0"

# Kill processes containing "/usr/bin/python3 -c from multiprocessing.resource_tracker import"
kill_processes "/usr/bin/python3 -c from multiprocessing.resource_tracker import"

# Kill processes containing "/usr/bin/python3 -c from multiprocessing.spawn import spawn_main"
kill_processes "/usr/bin/python3 -c from multiprocessing.spawn import spawn_main"

# Kill processes containing "dbus-launch --autolaunch=57969d545c4b453498457696713b0e11 --binary-syntax --close-stderr"
kill_processes "dbus-launch --autolaunch"

# Kill processes containing "/usr/bin/dbus-daemon --syslog-only --fork --print-pid 5 --print-address 7 --session"
kill_processes "/usr/bin/dbus-daemon --syslog-only "
