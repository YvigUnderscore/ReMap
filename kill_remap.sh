#!/bin/bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF_PID="$$"
PARENT_PID="${PPID:-}"

echo "Killing ReMap processes..."

declare -a PATTERNS=(
  "ReMap-GUI.py"
  "desktop_backend.py"
  "remap_server.py"
  "sfm_runner.py"
  "stray_to_colmap.py"
  "remap-desktop"
  "vite"
  "tauri"
)

declare -a EXTERNAL_NAMES=(
  "glomap"
  "colmap"
  "ffmpeg"
)

collect_pids() {
  local pattern="$1"
  pgrep -f "$pattern" 2>/dev/null || true
}

collect_names() {
  local name="$1"
  pgrep -x "$name" 2>/dev/null || true
}

collect_children() {
  local parent="$1"
  local children
  children="$(pgrep -P "$parent" 2>/dev/null || true)"
  if [ -z "$children" ]; then
    return
  fi
  echo "$children"
  while IFS= read -r child; do
    [ -n "$child" ] && collect_children "$child"
  done <<< "$children"
}

build_target_list() {
  {
    for pattern in "${PATTERNS[@]}"; do
      collect_pids "$pattern"
    done
    for name in "${EXTERNAL_NAMES[@]}"; do
      collect_names "$name"
    done
  } |
    awk -v self="$SELF_PID" -v parent="$PARENT_PID" '
      $1 ~ /^[0-9]+$/ && $1 != self && $1 != parent { print $1 }
    ' |
    sort -u
}

TARGETS="$(build_target_list)"
if [ -n "$TARGETS" ]; then
  DESCENDANTS="$(
    while IFS= read -r pid; do
      [ -n "$pid" ] && collect_children "$pid"
    done <<< "$TARGETS"
  )"
  TARGETS="$(printf "%s\n%s\n" "$TARGETS" "$DESCENDANTS" | awk 'NF' | sort -rn | uniq)"
fi

if [ -z "$TARGETS" ]; then
  echo "No ReMap processes were running."
  exit 0
fi

while IFS= read -r pid; do
  [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true
done <<< "$TARGETS"

sleep 0.8

REMAINING="$(build_target_list)"
if [ -n "$REMAINING" ]; then
  while IFS= read -r pid; do
    [ -n "$pid" ] && kill -KILL "$pid" 2>/dev/null || true
  done <<< "$REMAINING"
  sleep 0.4
fi

REMAINING="$(build_target_list)"
if [ -n "$REMAINING" ]; then
  echo "Some ReMap processes are still running:"
  REMAINING_CSV="$(printf "%s\n" "$REMAINING" | paste -sd, -)"
  ps -p "$REMAINING_CSV" -o pid=,ppid=,comm=,args=
  exit 1
fi

echo "All ReMap processes have been terminated."
