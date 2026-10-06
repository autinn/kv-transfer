#!/bin/bash
trap "echo \"EXIT signal \$(date +%T)\" >> /root/logs/run.log" TERM HUP INT
bash /workspace/run.sh >> /root/logs/run.log 2>&1
echo "EXIT $? $(date +%T)" >> /root/logs/run.log
