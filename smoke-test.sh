set -e

# Locate our project and Shunchang's repository.
cd "$(dirname "$0")"
project_dir="$PWD"
export MEM_DATA="$project_dir/../vlm-em/data"
export MEM_RESULTS="$project_dir/results"

cd "$project_dir/../vlm-em"
python -m evaluation.risky_actions.driver \
    --model qwen3vl-4b \
    --task base \
    --limit 1 \
    --rollouts 1 \
    --batch 1
