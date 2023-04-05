#!/bin/bash

# Define the path
path="" 

cd "$path"

# Start a new tmux session
tmux new-session -d -s tez 

# Select window
tmux select-window -t tez:0
tmux rename-window "nvim-1"

# Send keys to selected window
tmux send-keys "nvim" C-m

# Create a new window and set the working directory to the path
tmux new-window -n "nvim-2" 
tmux send-keys "nvim" C-m

tmux new-window -n "run" 

tmux new-window -n "python" 
tmux send-keys "python" C-m

# Start a new tmux session
tmux new-session -d -s deneme 

# Select window
tmux select-window -t deneme:0
tmux rename-window "deneme"
tmux send-keys "nvim" C-m

# Create a new window and set the working directory to the path
tmux new-window -n "terminal" 

# Attach to the tmux session
tmux attach-session -d -t tez

