#!/bin/sh

# Log the current timestamp
# echo "$(date -u) - Health check triggered" >> /opt/nvidia/deepstream/deepstream-7.0/sources/src/health_check_log.log


# Check if nvidia-smi has a non-zero exit code
# encoder_utilisation=$(nvidia-smi --query-gpu=utilization.encoder --format=csv,noheader,nounits)
encoder_utilisation=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)
nvidia_exit_code=$?

echo "$(date ) - Able to access GPU, and GPU is $encoder_utilisation"  >> /opt/nvidia/deepstream/deepstream-7.0/sources/src/logs/health_check_log.log
if [ $nvidia_exit_code -ne 0 ]; then
    echo "$(date ) - Unhealthy: nvidia-smi returned a non-zero exit code ($nvidia_exit_code)"
    echo "$(date ) - Unhealthy: nvidia-smi returned a non-zero exit code ($nvidia_exit_code)" >> /opt/nvidia/deepstream/deepstream-7.0/sources/src/logs/health_check_log.log
    exit 1
fi

ram_utilization=$(free | awk '/Mem/ {print int($3/$2 * 100)}')
echo "$(date) - RAM Utilization is $ram_utilization%" >> /opt/nvidia/deepstream/deepstream-7.0/sources/src/logs/health_check_log.log

# Check if RAM utilization is above 90%
if [ $ram_utilization -gt 90 ]; then
    echo "$(date) - Unhealthy: RAM utilization is above 90%"
    echo "$(date) - Unhealthy: RAM utilization is above 90%" >> /opt/nvidia/deepstream/deepstream-7.0/sources/src/logs/health_check_log.log
    exit 1
fi

# Perform the actual health check
if [ -f /opt/nvidia/deepstream/deepstream-7.0/sources/src/kill.txt ]; then
    echo "$(date ) - Unhealthy: kill.txt file exists"
    echo "$(date ) - Unhealthy: kill.txt file exists" >> /opt/nvidia/deepstream/deepstream-7.0/sources/src/logs/health_check_log.log
    exit 1
else
    echo "$(date ) - Healthy: kill.txt is not at /opt/nvidia/deepstream/deepstream-7.0/sources/src/"
    exit 0
fi
