#!/bin/bash

# Configuration variables
HOST="0.0.0.0"  # Default host
PORT=8100             # Default port
NUM_CLIENTS=1         # Default number of parallel clients
BASE_OUTPUT_DIR="./results"
TEST_DURATION=150      # Default test duration in seconds
CLIENT_START_DELAY=1  # Delay between client starts in seconds
THRESHOLD=0.5         # Default threshold for valid average latency

# Generate timestamp for unique directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${BASE_OUTPUT_DIR}_${TIMESTAMP}"
SUMMARY_FILE="$OUTPUT_DIR/summary.json"

# Process command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --clients)
      NUM_CLIENTS="$2"
      shift 2
      ;;
    --output-dir)
      BASE_OUTPUT_DIR="$2"
      OUTPUT_DIR="${BASE_OUTPUT_DIR}_${TIMESTAMP}"
      SUMMARY_FILE="$OUTPUT_DIR/summary.json"
      shift 2
      ;;
    --test-duration)
      TEST_DURATION="$2"
      shift 2
      ;;
    --client-start-delay)
      CLIENT_START_DELAY="$2"
      shift 2
      ;;
    --threshold)
      THRESHOLD="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--host HOST] [--port PORT] [--clients NUM_CLIENTS] [--output-dir DIR] [--test-duration SECONDS] [--client-start-delay SECONDS] [--threshold SECONDS]"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=== Voice Agent Multi-Client Benchmark (Staggered Start) ==="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Number of clients: $NUM_CLIENTS"
echo "Client start delay: $CLIENT_START_DELAY seconds"
echo "Test duration: $TEST_DURATION seconds"
echo "Latency threshold (minimum valid turn): $THRESHOLD seconds"
echo "Output directory: $OUTPUT_DIR"
echo "================================================================"

# Calculate timing
# All clients will start within (NUM_CLIENTS - 1) * CLIENT_START_DELAY seconds
# Metrics collection will start after the last client starts
TOTAL_START_TIME=$(( (NUM_CLIENTS - 1) * CLIENT_START_DELAY ))
METRICS_START_TIME=$(date +%s)
METRICS_START_TIME=$((METRICS_START_TIME + TOTAL_START_TIME))  # Set to when the last client starts

# Run clients with staggered starts
pids=()

for ((i=1; i<=$NUM_CLIENTS; i++)); do
  # Generate a unique stream ID for each client
  STREAM_ID="client_${i}_$(date +%s%N | cut -b1-13)"
  
  # Calculate start delay for this client (0 for first client, increasing for others)
  START_DELAY=$(( (i - 1) * CLIENT_START_DELAY ))
  
  # Run client in background with appropriate delays
  python ./file_input_client.py \
    --stream-id "$STREAM_ID" \
    --host "$HOST" \
    --port "$PORT" \
    --output-dir "$OUTPUT_DIR" \
    --start-delay "$START_DELAY" \
    --metrics-start-time "$METRICS_START_TIME" \
    --test-duration "$TEST_DURATION" \
    --threshold "$THRESHOLD" > "$OUTPUT_DIR/client_${i}.log" 2>&1 &
  
  # Store the process ID
  pids+=($!)
  
  # Small delay to ensure proper process creation
  sleep 0.1
done

echo ""
echo "Timing plan:"
echo "- First client starts immediately"
echo "- Last client starts in $TOTAL_START_TIME seconds"
echo "- Metrics collection starts at $(date -d @$METRICS_START_TIME)"
echo "- Test will run for $TEST_DURATION seconds after metrics collection starts"
echo "- Expected completion time: $(date -d @$((METRICS_START_TIME + TEST_DURATION)))"
echo ""

# Wait for all clients to finish
for pid in "${pids[@]}"; do
  wait "$pid"
done

# Calculate aggregate statistics across all clients
TOTAL_LATENCY=0
TOTAL_TURNS=0
CLIENT_COUNT=0
MIN_LATENCY=9999
MAX_LATENCY=0

# Variables for valid (thresholded) latency statistics
TOTAL_VALID_LATENCY=0
TOTAL_VALID_TURNS=0
CLIENTS_WITH_VALID=0
MIN_VALID_LATENCY=9999
MAX_VALID_LATENCY=0

# Arrays to store client average latencies for p95 calculation
CLIENT_LATENCIES=()
CLIENT_VALID_LATENCIES=()

# Arrays to track glitch detection
CLIENTS_WITH_GLITCHES=()
TOTAL_GLITCH_COUNT=0

# Variables to track reverse barge-in detection
TOTAL_REVERSE_BARGE_INS=0
CLIENT_REVERSE_BARGE_INS=()
CLIENTS_WITH_REVERSE_BARGE_INS=0

# Function to calculate p95 from an array of values
calculate_p95() {
  local values=("$@")
  local count=${#values[@]}
  
  if [ $count -eq 0 ]; then
    echo "0"
    return
  fi
  
  # Sort the array (using a simple bubble sort for bash compatibility)
  for ((i = 0; i < count; i++)); do
    for ((j = i + 1; j < count; j++)); do
      if (( $(echo "${values[i]} > ${values[j]}" | bc -l) )); then
        temp=${values[i]}
        values[i]=${values[j]}
        values[j]=$temp
      fi
    done
  done
  
  # Calculate p95 index 
  local p95_index=$(echo "scale=0; ($count - 1) * 0.95" | bc -l | cut -d'.' -f1)
  
  # Ensure index is within bounds
  if [ $p95_index -ge $count ]; then
    p95_index=$((count - 1))
  fi
  
  echo "${values[$p95_index]}"
}

# Process all result files
for result_file in "$OUTPUT_DIR"/latency_results_*.json; do
  if [ -f "$result_file" ]; then
    # Extract data using jq if available, otherwise use awk as fallback
    if command -v jq &> /dev/null; then
      AVG_LATENCY=$(jq '.average_latency' "$result_file")
      VALID_AVG_LATENCY=$(jq '.valid_average_latency' "$result_file")
      NUM_TURNS=$(jq '.num_turns' "$result_file")
      NUM_VALID_TURNS=$(jq '.num_valid_turns' "$result_file")
      STREAM_ID=$(jq -r '.stream_id' "$result_file")
      GLITCH_DETECTED=$(jq '.glitch_detected' "$result_file")
      REVERSE_BARGE_INS_COUNT=$(jq '.reverse_barge_ins_count' "$result_file")
    else
      # Fallback to grep and basic string processing
      AVG_LATENCY=$(grep -o '"average_latency": [0-9.]*' "$result_file" | cut -d' ' -f2)
      VALID_AVG_LATENCY=$(grep -o '"valid_average_latency": [0-9.]*' "$result_file" | cut -d' ' -f2)
      NUM_TURNS=$(grep -o '"num_turns": [0-9]*' "$result_file" | cut -d' ' -f2)
      NUM_VALID_TURNS=$(grep -o '"num_valid_turns": [0-9]*' "$result_file" | cut -d' ' -f2)
      STREAM_ID=$(grep -o '"stream_id": "[^"]*"' "$result_file" | cut -d'"' -f4)
      GLITCH_DETECTED=$(grep -o '"glitch_detected": [a-z]*' "$result_file" | cut -d' ' -f2)
      REVERSE_BARGE_INS_COUNT=$(grep -o '"reverse_barge_ins_count": [0-9]*' "$result_file" | cut -d' ' -f2)
    fi
    
    echo "Client $STREAM_ID: Average latency = $AVG_LATENCY seconds over $NUM_TURNS turns"
    
    # Display valid (thresholded) latency information
    if [ "$VALID_AVG_LATENCY" != "null" ] && [ -n "$VALID_AVG_LATENCY" ]; then
      echo "  Valid Average latency (>= $THRESHOLD s) = $VALID_AVG_LATENCY seconds over $NUM_VALID_TURNS turns"
      
      # Add to valid latency statistics
      TOTAL_VALID_LATENCY=$(echo "$TOTAL_VALID_LATENCY + $VALID_AVG_LATENCY" | bc -l)
      TOTAL_VALID_TURNS=$((TOTAL_VALID_TURNS + NUM_VALID_TURNS))
      CLIENTS_WITH_VALID=$((CLIENTS_WITH_VALID + 1))
      CLIENT_VALID_LATENCIES+=($VALID_AVG_LATENCY)
      
      # Update min/max valid latency
      if (( $(echo "$VALID_AVG_LATENCY < $MIN_VALID_LATENCY" | bc -l) )); then
        MIN_VALID_LATENCY=$VALID_AVG_LATENCY
      fi
      
      if (( $(echo "$VALID_AVG_LATENCY > $MAX_VALID_LATENCY" | bc -l) )); then
        MAX_VALID_LATENCY=$VALID_AVG_LATENCY
      fi
    else
      echo "  No latencies above $THRESHOLD s threshold"
    fi
    
    # Check for glitch detection
    if [ "$GLITCH_DETECTED" = "true" ]; then
      echo "  ⚠️  Audio glitches detected in client $STREAM_ID"
      CLIENTS_WITH_GLITCHES+=("$STREAM_ID")
      TOTAL_GLITCH_COUNT=$((TOTAL_GLITCH_COUNT + 1))
    fi

    # Display and track reverse barge-in count
    if [ -n "$REVERSE_BARGE_INS_COUNT" ] && [ "$REVERSE_BARGE_INS_COUNT" -gt 0 ]; then
      echo "  Reverse barge-ins detected: $REVERSE_BARGE_INS_COUNT occurrences"
      CLIENT_REVERSE_BARGE_INS+=("$STREAM_ID")
      CLIENTS_WITH_REVERSE_BARGE_INS=$((CLIENTS_WITH_REVERSE_BARGE_INS + 1))
    fi
    
    # Add to total reverse barge-in count
    if [ -n "$REVERSE_BARGE_INS_COUNT" ]; then
      TOTAL_REVERSE_BARGE_INS=$((TOTAL_REVERSE_BARGE_INS + REVERSE_BARGE_INS_COUNT))
    fi
    
    # Add to array for p95 calculation
    CLIENT_LATENCIES+=($AVG_LATENCY)
    
    # Update aggregate statistics
    TOTAL_LATENCY=$(echo "$TOTAL_LATENCY + $AVG_LATENCY" | bc -l)
    TOTAL_TURNS=$((TOTAL_TURNS + NUM_TURNS))
    CLIENT_COUNT=$((CLIENT_COUNT + 1))
    
    # Update min/max latency
    if (( $(echo "$AVG_LATENCY < $MIN_LATENCY" | bc -l) )); then
      MIN_LATENCY=$AVG_LATENCY
    fi
    
    if (( $(echo "$AVG_LATENCY > $MAX_LATENCY" | bc -l) )); then
      MAX_LATENCY=$AVG_LATENCY
    fi
  fi
done

# Calculate overall statistics
if [ $CLIENT_COUNT -gt 0 ]; then
  AGGREGATE_AVG_LATENCY=$(echo "scale=3; $TOTAL_LATENCY / $CLIENT_COUNT" | bc -l)
  P95_CLIENT_LATENCY=$(calculate_p95 "${CLIENT_LATENCIES[@]}")
  
  # Calculate valid statistics
  AGGREGATE_VALID_AVG_LATENCY="null"
  P95_VALID_CLIENT_LATENCY="null"
  
  if [ $CLIENTS_WITH_VALID -gt 0 ]; then
    AGGREGATE_VALID_AVG_LATENCY=$(echo "scale=3; $TOTAL_VALID_LATENCY / $CLIENTS_WITH_VALID" | bc -l)
    P95_VALID_CLIENT_LATENCY=$(calculate_p95 "${CLIENT_VALID_LATENCIES[@]}")
  fi
  
  echo ""
  echo "=============================================="
  echo "BENCHMARK SUMMARY (Staggered Start)"
  echo "=============================================="
  echo "Total clients: $CLIENT_COUNT"
  echo "Latency threshold (minimum valid turn): $THRESHOLD seconds"
  echo ""
  echo "STANDARD LATENCY STATISTICS:"
  echo "Average latency across all clients: $AGGREGATE_AVG_LATENCY seconds"
  echo "P95 latency across client averages: $P95_CLIENT_LATENCY seconds"
  echo "Minimum client average latency: $MIN_LATENCY seconds"
  echo "Maximum client average latency: $MAX_LATENCY seconds"
  echo ""
  # Show valid latency stats only if reverse barge-ins were detected
  if [ $TOTAL_REVERSE_BARGE_INS -gt 0 ] || [ $CLIENTS_WITH_REVERSE_BARGE_INS -gt 0 ]; then
    echo "VALID LATENCY STATISTICS (>= $THRESHOLD s):"
    if [ "$AGGREGATE_VALID_AVG_LATENCY" != "null" ]; then
      echo "Clients with valid data: $CLIENTS_WITH_VALID out of $CLIENT_COUNT"
      echo "Average valid latency: $AGGREGATE_VALID_AVG_LATENCY seconds"
      echo "P95 valid latency: $P95_VALID_CLIENT_LATENCY seconds"
      echo "Minimum client valid latency: $MIN_VALID_LATENCY seconds"
      echo "Maximum client valid latency: $MAX_VALID_LATENCY seconds"
    else
      echo "No latencies >= $THRESHOLD s threshold found across all clients"
    fi
  else
    echo "Valid latency statistics are identical to standard latency (no reverse barge-ins detected)."
  fi
  echo ""
  echo "AUDIO GLITCH DETECTION:"
  if [ $TOTAL_GLITCH_COUNT -gt 0 ]; then
    echo "⚠️  Audio glitches detected in $TOTAL_GLITCH_COUNT out of $CLIENT_COUNT clients"
    echo "Affected clients:"
    for client in "${CLIENTS_WITH_GLITCHES[@]}"; do
      echo "  - $client"
    done
  else
    echo "✅ No audio glitches detected in any client"
  fi

  echo "REVERSE BARGE-IN DETECTION:"
  if [ $TOTAL_REVERSE_BARGE_INS -gt 0 ]; then
    echo "Total reverse barge-ins detected: $TOTAL_REVERSE_BARGE_INS occurrences"
    echo "Clients with reverse barge-ins: $CLIENTS_WITH_REVERSE_BARGE_INS out of $CLIENT_COUNT"
    if [ $CLIENTS_WITH_REVERSE_BARGE_INS -gt 0 ]; then
      echo "Affected clients:"
      for client in "${CLIENT_REVERSE_BARGE_INS[@]}"; do
        echo "  - $client"
      done
    fi
  else
    echo "✅ No reverse barge-ins detected in any client"
  fi

  echo ""
  echo "ERROR DETECTION:"
  # Initialize arrays for error tracking
  declare -A CLIENT_ERROR_COUNTS
  CLIENTS_WITH_ERRORS=0
  TOTAL_ERRORS=0

  # Process logs for each client to find errors
  for ((i=1; i<=$NUM_CLIENTS; i++)); do
    LOG_FILE="$OUTPUT_DIR/client_${i}.log"
    if [ -f "$LOG_FILE" ]; then
      ERROR_COUNT=$(grep -c "^\[ERROR\]" "$LOG_FILE")
      if [ $ERROR_COUNT -gt 0 ]; then
        CLIENTS_WITH_ERRORS=$((CLIENTS_WITH_ERRORS + 1))
        TOTAL_ERRORS=$((TOTAL_ERRORS + ERROR_COUNT))
        CLIENT_ERROR_COUNTS["client_${i}"]=$ERROR_COUNT
        
        echo "⚠️  Client ${i} errors ($ERROR_COUNT):"
        grep "^\[ERROR\]" "$LOG_FILE" | sed 's/^/  /'
      fi
    fi
  done

  if [ $TOTAL_ERRORS -eq 0 ]; then
    echo "✅ No errors detected in any client"
  else
    echo "⚠️  Total errors across all clients: $TOTAL_ERRORS"
    echo "⚠️  Clients with errors: $CLIENTS_WITH_ERRORS out of $CLIENT_COUNT"
  fi
  
  # Create summary JSON
  cat > "$SUMMARY_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "config": {
    "host": "$HOST",
    "port": $PORT,
    "num_clients": $NUM_CLIENTS,
    "client_start_delay": $CLIENT_START_DELAY,
    "test_duration": $TEST_DURATION,
    "threshold": $THRESHOLD,
    "metrics_start_time": $METRICS_START_TIME
  },
  "results": {
    "total_clients": $CLIENT_COUNT,
    "total_turns": $TOTAL_TURNS,
    "aggregate_average_latency": $AGGREGATE_AVG_LATENCY,
    "p95_client_latency": $P95_CLIENT_LATENCY,
    "min_client_latency": $MIN_LATENCY,
    "max_client_latency": $MAX_LATENCY,
    "valid_results": {
      "clients_with_valid_data": $CLIENTS_WITH_VALID,
      "total_valid_turns": $TOTAL_VALID_TURNS,
      "aggregate_valid_average_latency": $AGGREGATE_VALID_AVG_LATENCY,
      "p95_valid_client_latency": $P95_VALID_CLIENT_LATENCY,
      "min_valid_client_latency": $([ "$AGGREGATE_VALID_AVG_LATENCY" != "null" ] && echo "$MIN_VALID_LATENCY" || echo "null"),
      "max_valid_client_latency": $([ "$AGGREGATE_VALID_AVG_LATENCY" != "null" ] && echo "$MAX_VALID_LATENCY" || echo "null")
    },
    "glitch_detection": {
      "clients_with_glitches": $TOTAL_GLITCH_COUNT,
      "total_clients": $CLIENT_COUNT,
      "affected_client_ids": [$(printf '"%s",' "${CLIENTS_WITH_GLITCHES[@]}" | sed 's/,$//')]
    },
    "reverse_barge_in_detection": {
      "total_clients": $CLIENT_COUNT,
      "total_reverse_barge_ins": $TOTAL_REVERSE_BARGE_INS,
      "clients_with_reverse_barge_ins": $CLIENTS_WITH_REVERSE_BARGE_INS,
      "affected_client_ids": [$(printf '"%s",' "${CLIENT_REVERSE_BARGE_INS[@]}" | sed 's/,$//')]
    },
    "error_detection": {
      "total_clients": $CLIENT_COUNT,
      "total_errors": $TOTAL_ERRORS,
      "clients_with_errors": $CLIENTS_WITH_ERRORS,
      "client_error_counts": {
        $(for client in "${!CLIENT_ERROR_COUNTS[@]}"; do
          printf '"%s": %d,\n        ' "$client" "${CLIENT_ERROR_COUNTS[$client]}"
        done | sed 's/,\s*$//')
      }
    }
  }
}
EOF
  
  echo "Summary saved to: $SUMMARY_FILE"
else
  echo "No valid result files found!"
fi

echo "Benchmark complete." 