i#!/usr/bin/env bash

# file: extract_kappa.sh

output_file="set1_k.dat"

folders=(
    "0.25M_ECEMC_T293"
    "0.50M_ECEMC_T293"
    "1.0M_ECEMC_T293"
    "1.5M_ECEMC_T293"
    "2.0M_ECEMC_T293"
    "2.5M_ECEMC_T293"
    "3.0M_ECEMC_T293"
)

> "$output_file"

for dir in "${folders[@]}"; do

    file="$dir/output_LELiB_MDLab.txt"

    if [[ ! -f "$file" ]]; then
        echo "Missing file: $file"
        continue
    fi

    awk '
    /Molality:/ {
        molality = $2
    }

    /Ionic conductivity =/ {
        conductivity = $4
        error = $6

        printf "%.4f\t%.4f\t%.4f\n", molality, conductivity, error
    }
    ' "$file" >> "$output_file"

done

echo "Written: $output_file"
