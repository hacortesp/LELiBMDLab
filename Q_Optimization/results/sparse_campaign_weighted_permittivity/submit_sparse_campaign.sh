#!/usr/bin/env bash
set -euo pipefail

# Submit every system listed in launch_sparse_campaign.sh.
#
# This script expects the campaign file to be either:
# 1. the current shell-friendly dry-run format with lines like
#    echo --job-id ... --cation-name Li --anion-name PF6 --salt-conc 1.0 \
#         --solvents '["EC","DMC"]' --solvent-fracs '[0.5,0.5]' \
#         --temperature 298.15 --q-scale 0.85
# 2. a CSV table with equivalent columns using dashed or underscored names.
#
# By default, one directory per row is created under this campaign directory,
# code is copied from the Li-EMDLab root, runlelib.sh is written, and sbatch is
# called from inside each directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_FILE="${SCRIPT_DIR}/launch_sparse_campaign.sh"
PARSER="${SCRIPT_DIR}/parse_sparse_campaign.py"
OUTPUT_ROOT="${SCRIPT_DIR}"
if [[ -d "${SCRIPT_DIR}/LELiBMDLab" && -f "${SCRIPT_DIR}/lelib_mdlab.py" ]]; then
  CODE_ROOT="${SCRIPT_DIR}"
else
  CODE_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_FILE="${SCRIPT_DIR}/submitted_jobs.log"

DRY_RUN=0
PREPARE_ONLY=0
OVERWRITE=0
LIMIT=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --campaign-file PATH  Campaign file to read.
                        Default: ${CAMPAIGN_FILE}
  --output-root PATH    Directory where per-system folders are created.
                        Default: ${OUTPUT_ROOT}
  --code-root PATH      Directory containing LELiBMDLab and lelib_mdlab.py.
                        Default: ${CODE_ROOT}
  --log-file PATH       Submission log path.
                        Default: ${LOG_FILE}
  --dry-run             Print planned actions only. Create nothing; submit nothing.
  --prepare-only        Create folders/scripts, but do not call sbatch.
  --overwrite           Replace an existing per-system folder before preparing it.
  --limit N             Process only the first N campaign rows.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign-file)
      CAMPAIGN_FILE="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --code-root)
      CODE_ROOT="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$CAMPAIGN_FILE" ]]; then
  echo "Campaign file not found: $CAMPAIGN_FILE" >&2
  exit 1
fi

if [[ ! -x "$PARSER" && ! -f "$PARSER" ]]; then
  echo "Parser helper not found: $PARSER" >&2
  exit 1
fi

if [[ ! "$LIMIT" =~ ^[0-9]+$ ]]; then
  echo "--limit must be a non-negative integer, got: $LIMIT" >&2
  exit 2
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  if [[ ! -d "${CODE_ROOT}/LELiBMDLab" ]]; then
    echo "Missing code directory: ${CODE_ROOT}/LELiBMDLab" >&2
    exit 1
  fi
  if [[ ! -f "${CODE_ROOT}/lelib_mdlab.py" ]]; then
    echo "Missing driver script: ${CODE_ROOT}/lelib_mdlab.py" >&2
    exit 1
  fi
  mkdir -p "$OUTPUT_ROOT"
  {
    printf '# submit_sparse_campaign.sh run at %s\n' "$(date -Is)"
    printf '# campaign_file=%s\n' "$CAMPAIGN_FILE"
    printf '# output_root=%s\n' "$OUTPUT_ROOT"
    printf '# code_root=%s\n' "$CODE_ROOT"
    printf 'timestamp\tjob_id\tfolder\tstatus\tsbatch_output\n'
  } >> "$LOG_FILE"
fi

write_run_script() {
  local run_script="$1"
  local job_name="$2"
  local cation_name="$3"
  local anion_name="$4"
  local salt_conc="$5"
  local temperature="$6"
  local q_scale="$7"
  local solvents_text="$8"
  local solvent_fracs_text="$9"

  local solvents=()
  local solvent_fracs=()
  read -r -a solvents <<< "$solvents_text"
  read -r -a solvent_fracs <<< "$solvent_fracs_text"

  {
    cat <<EOF
#!/bin/bash
#SBATCH --account=bcam-exclusive
#SBATCH --partition=bcam-exclusive
#SBATCH --job-name=${job_name}
#SBATCH --ntasks-per-node=48
#SBATCH --mem=50gb
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
##SBATCH --partition=test

module load GROMACS/2019.4-foss-2019b
export PATH="/scratch/hacortes/packmol-21.2.1:\$PATH"

source /scratch/hacortes/anaconda3/etc/profile.d/conda.sh

conda activate md

export PYTHONUNBUFFERED=1

python lelib_mdlab.py \\
  -box 50.0 \\
  -cation_name $(printf '%q' "$cation_name") \\
  -anion_name $(printf '%q' "$anion_name") \\
  -salt-conc $(printf '%q' "$salt_conc") \\
EOF
    printf '  -solvents'
    printf ' %q' "${solvents[@]}"
    printf ' \\\n'
    printf '  -solvent-fracs'
    printf ' %q' "${solvent_fracs[@]}"
    printf ' \\\n'
    printf '  -charge-scale %q \\\n' "$q_scale"
    printf '  -temperature %q \\\n' "$temperature"
    printf '  -parallel True \\\n'
    printf '  > output_LELiB_MDLab.txt\n'
  } > "$run_script"

  chmod +x "$run_script"
}

processed=0
submitted=0
skipped=0
parsed_rows="$(mktemp)"
trap 'rm -f "$parsed_rows"' EXIT

if ! "$PYTHON_BIN" "$PARSER" "$CAMPAIGN_FILE" > "$parsed_rows"; then
  echo "Failed to parse campaign file with ${PYTHON_BIN}: ${CAMPAIGN_FILE}" >&2
  exit 1
fi

while IFS=$'\t' read -r job_id folder_name job_name cation_name anion_name salt_conc temperature q_scale solvents_text solvent_fracs_text; do
  if [[ -z "${job_id:-}" ]]; then
    continue
  fi
  if [[ "$LIMIT" -gt 0 && "$processed" -ge "$LIMIT" ]]; then
    break
  fi

  processed=$((processed + 1))
  folder="${OUTPUT_ROOT}/${folder_name}"
  run_script="${folder}/runlelib.sh"

  echo "--->> ${job_id}: ${cation_name} ${anion_name} ${salt_conc} M, solvents=${solvents_text}, fracs=${solvent_fracs_text}, T=${temperature}, q=${q_scale}"
  echo "      folder: ${folder}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    continue
  fi

  if [[ -e "$folder" ]]; then
    if [[ "$OVERWRITE" -eq 1 ]]; then
      rm -rf "$folder"
    else
      echo "      skipping existing folder; use --overwrite to replace it"
      printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$job_id" "$folder" "SKIPPED_EXISTS" "" >> "$LOG_FILE"
      skipped=$((skipped + 1))
      continue
    fi
  fi

  mkdir -p "$folder"
  cp -r "${CODE_ROOT}/LELiBMDLab" "${CODE_ROOT}/lelib_mdlab.py" "$folder/"
  write_run_script "$run_script" "$job_name" "$cation_name" "$anion_name" "$salt_conc" "$temperature" "$q_scale" "$solvents_text" "$solvent_fracs_text"

  if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$job_id" "$folder" "PREPARED" "" >> "$LOG_FILE"
    continue
  fi

  echo "      submitting with sbatch"
  if sbatch_output="$(cd "$folder" && sbatch runlelib.sh 2>&1)"; then
    echo "      ${sbatch_output}"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$job_id" "$folder" "SUBMITTED" "$sbatch_output" >> "$LOG_FILE"
    submitted=$((submitted + 1))
  else
    echo "      sbatch failed: ${sbatch_output}" >&2
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$job_id" "$folder" "FAILED" "$sbatch_output" >> "$LOG_FILE"
    exit 1
  fi
done < "$parsed_rows"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run complete: ${processed} planned jobs."
elif [[ "$PREPARE_ONLY" -eq 1 ]]; then
  echo "Prepare-only complete: ${processed} jobs prepared, ${skipped} skipped. Log: ${LOG_FILE}"
else
  echo "Submission complete: ${submitted} jobs submitted, ${skipped} skipped. Log: ${LOG_FILE}"
fi
