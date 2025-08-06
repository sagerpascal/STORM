envs=("Hero" "Alien" "Frostbite")

for env_name in "${envs[@]}"; do
    sbatch run_training.submit \
        -n "${env_name}-life_done-wm_2L512D8H-100k-seed1" \
        -seed 1 \
        -config_path "config_files/STORM.yaml" \
        -env_name "ALE/${env_name}-v5"
done