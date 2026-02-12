#!/bin/bash
# CloudTrail Anomaly Detector — daily analysis wrapper for launchd/cron
# Runs yesterday's analysis, outputs dated JSON report
#
# Usage: Update PROJECT_DIR below or place this script in the project root.
# Schedule via launchd (macOS) or cron for daily automated runs.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
VENV_PYTHON="${PROJECT_DIR}/venv/bin/python"
ANALYZER="${PROJECT_DIR}/cloudtrail_analyzer.py"
REPORTS_DIR="${PROJECT_DIR}/reports"
LOG_FILE="${REPORTS_DIR}/daily-run.log"
REPORT_FILE="${REPORTS_DIR}/$(date +%F).daily-run.json"

mkdir -p "${REPORTS_DIR}"

echo "=== $(date) ===" >> "${LOG_FILE}"
# Auto-detect GeoIP database for impossible travel detection
GEOIP_DB="${PROJECT_DIR}/maxmind/GeoLite2-City.mmdb"
GEOIP_FLAG=""
if [ -f "${GEOIP_DB}" ]; then
    GEOIP_FLAG="--geoip-db ${GEOIP_DB}"
fi

"${VENV_PYTHON}" "${ANALYZER}" \
    --s3-logs \
    --yesterday \
    --max-events 5000 \
    --skip-ai \
    ${GEOIP_FLAG} \
    --output-json "${REPORT_FILE}" \
    >> "${LOG_FILE}" 2>&1
echo "Exit code: $?" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"
