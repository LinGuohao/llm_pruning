#!/bin/bash
# 强力版本，当stop_evolution.sh无法终止所有进程时使用
# Simple and aggressive script to kill all evolution-related processes
# Usage: ./genetic/kill_evolution.sh

echo "=========================================="
echo "Force Killing All Evolution Processes"
echo "=========================================="

# Kill all processes matching the pattern
echo "Killing run_evolution.sh processes..."
pkill -9 -f "run_evolution.sh"

echo "Killing main_evolution.py processes..."
pkill -9 -f "main_evolution.py"

echo "Killing genetic_pruning related Python processes..."
pkill -9 -f "genetic_pruning"

# Wait a moment
sleep 1

# Check what's left
echo ""
echo "Checking remaining processes..."

REMAINING=$(pgrep -f "evolution" | wc -l)

if [ "$REMAINING" -gt 0 ]; then
    echo "⚠️  $REMAINING evolution-related processes still running:"
    ps aux | grep -E "evolution|genetic" | grep -v grep | grep -v $0
    echo ""
    echo "You may need to manually kill these processes."
else
    echo "✓ All evolution processes killed successfully"
fi

# Show GPU status
echo ""
echo "Current GPU usage:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | \
    while IFS=, read -r pid mem; do
        echo "  PID $pid using $mem"
        ps -p $pid -o comm= 2>/dev/null || echo "    (process name unavailable)"
    done
else
    echo "nvidia-smi not available"
fi

echo ""
echo "=========================================="
echo "Done"
echo "=========================================="
