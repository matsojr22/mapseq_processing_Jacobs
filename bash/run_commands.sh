#!/bin/bash

# Create a log file with a timestamp
LOGFILE="processing_$(date +'%Y%m%d_%H%M%S').log"

# Loop through each line of the command file
while IFS= read -r line || [[ -n "$line" ]]; do
  echo "Running: $line" | tee -a "$LOGFILE"
  eval "$line" >> "$LOGFILE" 2>&1
done < all_commands.txt

echo "Finished all commands at $(date)" | tee -a "$LOGFILE"
