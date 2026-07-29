#!/usr/bin/env bash
set -euo pipefail

python_command="${PYTHON:-python3}"
checkpoint="${CHECKPOINT:-checkpoints/mnist_ddpm.pt}"
sample_count=36
seed=2026

run_case() {
    case_name="$1"
    explanation="$2"
    shift 2

    output_dir="samples/${case_name}"
    mkdir -p "$output_dir"

    command=(
        "$python_command" sample.py
        --checkpoint "$checkpoint"
        --num-images "$sample_count"
        --out-dir "$output_dir"
        --seed "$seed"
        --save-animation
        "$@"
    )

    {
        printf '# %s\n\n' "$case_name"
        printf '%s\n\n' "$explanation"
        printf 'This case generates %s images with seed `%s`.\n\n' \
            "$sample_count" "$seed"
        printf '```bash\n'
        printf '%q ' "${command[@]}"
        printf '\n```\n'
    } > "$output_dir/readme.md"

    printf 'Running %s\n' "$case_name"
    "${command[@]}"
}

run_case \
    "unconditional" \
    "Uses the learned null label (-1). No class label is supplied, so guidance scale has no effect."

for label in {0..9}; do
    run_case \
        "class_${label}_guidance_2" \
        "Generates only MNIST class ${label} with classifier-free guidance scale 2." \
        --labels "$label" \
        --guidance-scale 2
done

run_case \
    "all_classes_guidance_2" \
    "Cycles through labels 0-9 in one grid, making it easy to compare every class under the same sampling settings." \
    --labels "0,1,2,3,4,5,6,7,8,9" \
    --guidance-scale 2

for scale in 0 0.5 1 2 4 8; do
    case_scale="${scale//./_}"
    case "$scale" in
        0)
            scale_explanation="Scale 0 removes the class-conditioning contribution and serves as the unconditional baseline."
            ;;
        0.5)
            scale_explanation="Scale 0.5 applies weak conditioning, interpolating between null and class-conditioned predictions."
            ;;
        1)
            scale_explanation="Scale 1 uses the ordinary conditional prediction without classifier-free extrapolation."
            ;;
        2)
            scale_explanation="Scale 2 applies moderate classifier-free guidance and is the project default."
            ;;
        4)
            scale_explanation="Scale 4 applies strong guidance, which may improve class fidelity while reducing variety."
            ;;
        8)
            scale_explanation="Scale 8 is an intentionally aggressive case that can expose over-guidance artifacts and loss of diversity."
            ;;
    esac

    run_case \
        "odd_classes_guidance_${case_scale}" \
        "Cycles through labels 1, 3, 5, and 7. ${scale_explanation} All guidance cases share the same seed for direct comparison." \
        --labels "1,3,5,7" \
        --guidance-scale "$scale"
done

printf 'Finished all sampling cases under ./samples\n'
