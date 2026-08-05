#!/usr/bin/env bash
set -euo pipefail

python_command="${PYTHON:-python3}"
checkpoint="${CHECKPOINT:-checkpoints/mnist_ddpm.pt}"
checkpoint_t50="${CHECKPOINT_T50:-checkpoints/mnist_ddpm_t50.pt}"
experiment_dir="${EXPERIMENT_DIR:-ddim_vs_ddpm}"
num_images="${NUM_IMAGES:-64}"
seed="${SEED:-2026}"

run_sample() {
    output_name="$1"
    sample_checkpoint="$2"
    shift 2
    printf 'Running %s\n' "$output_name"
    "$python_command" sample.py \
        --checkpoint "$sample_checkpoint" \
        --seed "$seed" \
        --out-dir "$experiment_dir/$output_name" \
        "$@"
}

comparison_dir="$experiment_dir/comparisons"

# 1. Compare samplers and training schedules.
run_sample ddpm_10_steps "$checkpoint" --num-images "$num_images" --num-steps 10
run_sample ddim_10_steps_eta_0 "$checkpoint" \
    --num-images "$num_images" --ddim --num-steps 10 --eta 0
run_sample ddpm_t50_10_steps "$checkpoint_t50" \
    --num-images "$num_images" --num-steps 10
run_sample ddim_t50_10_steps_eta_0 "$checkpoint_t50" \
    --num-images "$num_images" --ddim --num-steps 10 --eta 0
run_sample ddpm_t50_50_steps "$checkpoint_t50" \
    --num-images "$num_images" --num-steps 50
run_sample ddim_t50_50_steps_eta_0 "$checkpoint_t50" \
    --num-images "$num_images" --ddim --num-steps 50 --eta 0

"$python_command" combine_results.py --columns 2 \
    --item "T=1000 model: 10-step DDPM=$experiment_dir/ddpm_10_steps/grid.png" \
    --item "T=1000 model: 10-step DDIM=$experiment_dir/ddim_10_steps_eta_0/grid.png" \
    --item "T=50 model: 10-step DDPM=$experiment_dir/ddpm_t50_10_steps/grid.png" \
    --item "T=50 model: 10-step DDIM=$experiment_dir/ddim_t50_10_steps_eta_0/grid.png" \
    --item "T=50 model: 50-step DDPM=$experiment_dir/ddpm_t50_50_steps/grid.png" \
    --item "T=50 model: 50-step DDIM=$experiment_dir/ddim_t50_50_steps_eta_0/grid.png" \
    --output "$comparison_dir/ddpm_vs_ddim_10_steps.png"

# 2. Compare DDIM stochasticity with a five-step sampling schedule.
for eta in 0 0.25 0.5 0.75 1; do
    run_sample "ddim_5_steps_eta_${eta}" "$checkpoint" \
        --num-images "$num_images" --ddim --num-steps 5 --eta "$eta"
done

"$python_command" combine_results.py --columns 3 \
    --item "eta=0=$experiment_dir/ddim_5_steps_eta_0/grid.png" \
    --item "eta=0.25=$experiment_dir/ddim_5_steps_eta_0.25/grid.png" \
    --item "eta=0.5=$experiment_dir/ddim_5_steps_eta_0.5/grid.png" \
    --item "eta=0.75=$experiment_dir/ddim_5_steps_eta_0.75/grid.png" \
    --item "eta=1=$experiment_dir/ddim_5_steps_eta_1/grid.png" \
    --output "$comparison_dir/ddim_eta.png"

# 3. Compare stochastic DDPM and deterministic DDIM across step counts.
step_comparison_args=(
    --item "10-step DDPM=$experiment_dir/ddpm_10_steps/grid.png"
    --item "10-step DDIM=$experiment_dir/ddim_10_steps_eta_0/grid.png"
)
for steps in 50 100 250 500 1000; do
    run_sample "ddpm_${steps}_steps" "$checkpoint" \
        --num-images "$num_images" --num-steps "$steps"
    run_sample "ddim_${steps}_steps_eta_0" "$checkpoint" \
        --num-images "$num_images" --ddim --num-steps "$steps" --eta 0
    step_comparison_args+=(
        --item "$steps-step DDPM=$experiment_dir/ddpm_${steps}_steps/grid.png"
        --item "$steps-step DDIM=$experiment_dir/ddim_${steps}_steps_eta_0/grid.png"
    )
done

"$python_command" combine_results.py --columns 2 \
    "${step_comparison_args[@]}" \
    --output "$comparison_dir/ddpm_vs_ddim_steps.png"

# 4. Compare interpolation from identical latent pairs.
run_sample ddpm_interpolation_10_steps "$checkpoint" \
    --num-steps 10 --interpolation --num-interpolations 12
run_sample ddim_interpolation_10_steps "$checkpoint" \
    --ddim --num-steps 10 --eta 0 --interpolation --num-interpolations 12

"$python_command" combine_results.py --columns 2 \
    --item "10-step DDPM; rows: endpoint 1, endpoint 2, midpoint=$experiment_dir/ddpm_interpolation_10_steps/grid.png" \
    --item "10-step DDIM; rows: endpoint 1, endpoint 2, midpoint=$experiment_dir/ddim_interpolation_10_steps/grid.png" \
    --output "$comparison_dir/ddpm_vs_ddim_interpolation.png"

printf 'Finished experiments and comparisons under %s\n' "$experiment_dir"
