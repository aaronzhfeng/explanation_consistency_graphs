#!/bin/bash
# Live experiment monitor — run with: bash scripts/monitor.sh
# Refreshes every 10 seconds. Ctrl+C to stop.

TASKS_DIR="/tmp/claude-0/-root-explanation-consistency-graphs/tasks"

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║            ECG EXPERIMENT MONITOR  $(date '+%H:%M:%S')              ║"
    echo "╠══════════════════════════════════════════════════════════════╣"

    for f in "$TASKS_DIR"/*.output; do
        [ -f "$f" ] || continue
        id=$(basename "$f" .output)

        # Detect experiment from early lines
        name=$(grep -m1 "ECG Experiment:" "$f" 2>/dev/null | sed 's/.*ECG Experiment: //' || echo "unknown")

        # Get latest progress line
        progress=$(grep "Progress:" "$f" 2>/dev/null | tail -1 | sed 's/.*Progress: //' || echo "—")

        # Check for completion or error
        if grep -q "Results saved to:" "$f" 2>/dev/null; then
            status="DONE"
            # Count how many seeds finished
            seeds_done=$(grep -c "Results saved to:" "$f" 2>/dev/null)
            progress="$seeds_done/5 seeds complete"
        elif grep -q "Traceback" "$f" 2>/dev/null; then
            status="FAIL"
            progress=$(grep "Error\|KeyError\|Exception" "$f" 2>/dev/null | tail -1 | cut -c1-50)
        else
            status=" RUN"
        fi

        # Get current phase
        phase=$(grep -E "^\s+\[[0-9]" "$f" 2>/dev/null | tail -1 | sed 's/.*\]//' | head -c30 || echo "")
        seed_line=$(grep "^Seed:" "$f" 2>/dev/null | tail -1 || echo "")

        printf "║ [%s] %-45s ║\n" "$status" "$name"
        printf "║        %-52s ║\n" "$seed_line  |  $progress"
    done

    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  Refreshing every 10s. Press Ctrl+C to stop.               ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    sleep 10
done
