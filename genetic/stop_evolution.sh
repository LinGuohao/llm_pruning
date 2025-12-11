#!/bin/bash
# 糅合版本，无法结束干净时使用kill_evolution.sh
# Stop evolution script - kills the main process and all its children
# Usage: ./genetic/stop_evolution.sh [PID]
#        If no PID provided, will search for run_evolution.sh process

echo "=========================================="
echo "Stopping Genetic Algorithm Evolution"
echo "=========================================="

# Function to kill process tree
kill_process_tree() {
    local pid=$1
    local children=$(pgrep -P $pid)

    # Kill children first (recursively)
    for child in $children; do
        kill_process_tree $child
    done

    # Kill the process itself
    if ps -p $pid > /dev/null 2>&1; then
        echo "Killing process $pid ($(ps -p $pid -o comm=))"
        kill -9 $pid 2>/dev/null
    fi
}

# Get PID from argument or find it
if [ -n "$1" ]; then
    MAIN_PID=$1
else
    # Find run_evolution.sh process
    MAIN_PID=$(pgrep -f "run_evolution.sh" | head -1)

    if [ -z "$MAIN_PID" ]; then
        echo "No run_evolution.sh process found."
        echo "Searching for main_evolution.py processes..."

        # Try to find Python processes running main_evolution.py
        PYTHON_PIDS=$(pgrep -f "main_evolution.py")

        if [ -z "$PYTHON_PIDS" ]; then
            echo "No evolution processes found."
            echo ""
            echo "Current Python processes:"
            ps aux | grep python | grep -v grep
            exit 1
        else
            echo "Found Python evolution processes: $PYTHON_PIDS"
            for pid in $PYTHON_PIDS; do
                echo "Killing Python process $pid..."
                kill_process_tree $pid
            done
            echo "✓ All Python evolution processes killed."
            exit 0
        fi
    fi
fi

echo "Found main process PID: $MAIN_PID"
echo ""

# Show process tree before killing
echo "Process tree:"
pstree -p $MAIN_PID 2>/dev/null || ps -ef | grep $MAIN_PID

echo ""
echo "Killing process tree..."

# Kill the entire process tree
kill_process_tree $MAIN_PID

echo ""
echo "Waiting for processes to terminate..."
sleep 2

# Verify all processes are killed
echo ""
echo "Checking for remaining processes..."

REMAINING_SHELL=$(pgrep -f "run_evolution.sh")
REMAINING_PYTHON=$(pgrep -f "main_evolution.py")

if [ -n "$REMAINING_SHELL" ] || [ -n "$REMAINING_PYTHON" ]; then
    echo "⚠️  Some processes are still running:"
    [ -n "$REMAINING_SHELL" ] && echo "  Shell: $REMAINING_SHELL"
    [ -n "$REMAINING_PYTHON" ] && echo "  Python: $REMAINING_PYTHON"

    echo ""
    echo "Force killing remaining processes..."
    [ -n "$REMAINING_SHELL" ] && kill -9 $REMAINING_SHELL 2>/dev/null
    [ -n "$REMAINING_PYTHON" ] && kill -9 $REMAINING_PYTHON 2>/dev/null

    sleep 1
fi

# Check GPU usage
echo ""
echo "Checking GPU usage..."
if command -v nvidia-smi &> /dev/null; then
    echo "Processes using GPU:"
    nvidia-smi | grep -E "python|Python" || echo "  No Python processes using GPU"
else
    echo "nvidia-smi not available, skipping GPU check"
fi

echo ""
echo "=========================================="
echo "✓ Evolution processes stopped"
echo "=========================================="
