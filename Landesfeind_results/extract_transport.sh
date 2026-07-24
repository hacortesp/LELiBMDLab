#!/usr/bin/env bash

# file: extract_transport.sh

k_output="set1_k.dat"
tplus_output="set1_tplus.dat"

folders=(
    "0.25M_ECEMC_T293"
    "0.50M_ECEMC_T293"
    "1.0M_ECEMC_T293"
    "1.5M_ECEMC_T293"
    "2.0M_ECEMC_T293"
    "2.5M_ECEMC_T293"
    "3.0M_ECEMC_T293"
)

> "$k_output"
> "$tplus_output"

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
        conductivity_error = $6

        printf "%.4f\t%.4f\t%.4f\n",
               molality,
               conductivity,
               conductivity_error
    }
    ' "$file" >> "$k_output"

    awk '
    /Molality:/ {
        molality = $2
    }

    /t\+ =/ {
        tplus = $3
        tplus_error = $5

        printf "%.4f\t%.4f\t%.4f\n",
               molality,
               tplus,
               tplus_error
    }
    ' "$file" >> "$tplus_output"

done

echo "Written:"
echo "  $k_output"
echo "  $tplus_output"
