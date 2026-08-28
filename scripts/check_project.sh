#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Project dir : $PROJECT_DIR"
echo "OpenPI dir  : $PROJECT_DIR/external/openpi"
echo

missing=0
for path in \
  "$PROJECT_DIR/src/pi05_jax_sft" \
  "$PROJECT_DIR/configs/pi05_company_example.yaml" \
  "$PROJECT_DIR/requirements/openpi_jax.txt" \
  "$PROJECT_DIR/external/openpi/src/openpi" \
  "$PROJECT_DIR/external/openpi/packages/openpi-client/src/openpi_client"; do
  if [[ -e "$path" ]]; then
    echo "OK      $path"
  else
    echo "MISSING $path"
    missing=1
  fi
done

if [[ "$missing" == "1" ]]; then
  echo
  echo "Fix OpenPI by copying, linking, or cloning it into:"
  echo "  $PROJECT_DIR/external/openpi"
  exit 1
fi

echo
echo "Project layout looks OK."
