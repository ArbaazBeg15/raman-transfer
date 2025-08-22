#!/bin/bash

# Set n (e.g., 4 instances)
n=4  # Change this to your desired number

# Define the full absolute path to your Python script
SCRIPT_PATH="/spectogram-transfer/code/xgb.py"  # Replace with your actual full path (e.g., /home/user/scripts/xgb.py)

# Start a new tmux session
tmux new-session -d -s optuna_session

# Launch the first instance in the main window
tmux send-keys -t optuna_session "python3 $SCRIPT_PATH" C-m

# Create and launch additional panes/windows
for ((i=2; i<=n; i++)); do
    tmux split-window -h -t optuna_session  # Or -v for vertical split
    tmux send-keys -t optuna_session "python3 $SCRIPT_PATH" C-m
done

# Attach to the session to view all panes
tmux attach -t optuna_session
