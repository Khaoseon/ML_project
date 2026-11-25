# CS-433 Machine Learning Project 2

We're studying emergent misalignment in vision-language models that can take
actions. Our first test gives Qwen3-VL-4B-Instruct a birthday task in a simulated
phone from [Shunchang Liu's `vlm-em` repo](https://github.com/shunchang-liu/vlm-em).
We run his driver from a separate checkout and keep our scripts and results here.

## Set up local inference

Use Linux or Ubuntu under WSL2 with Conda and a CUDA-capable NVIDIA GPU that
supports BF16. We tested this setup on a 16 GB RTX 5080. You can work on the
repo and inspect results without a GPU.

From `ML_project`, set up the environment and the sibling benchmark checkout.
If you already have `vlm-em-infer`, activate it in place of creating it. If you
already have `../vlm-em`, check that it is at `595964a` and skip the clone.

```bash
conda create -n vlm-em-infer python=3.11.18 pip
conda activate vlm-em-infer
python -m pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-inference.txt
python -m pip check
python -m playwright install --only-shell chromium

git clone https://github.com/shunchang-liu/vlm-em.git ../vlm-em
git -C ../vlm-em checkout --detach 595964a9d0f7d055e4254f7d6fa166ee3309553c
hf download Shunchang/multimodal-em-data eval/risky_actions/sandbox2_tasks.json \
  --repo-type dataset --revision 3dde3639fb95cf52ff9cb6e17a5a1433558be157 \
  --local-dir ../vlm-em/data
hf download Shunchang/multimodal-em-data eval/open_ended/eval_input_broad90_photo.json \
  --repo-type dataset --revision 3dde3639fb95cf52ff9cb6e17a5a1433558be157 \
  --local-dir ../vlm-em/data
```

The second file holds the 90 open-ended questions. Print them grouped by domain
with `python scripts/list_questions.py`. `docs/question_task_mapping.md` maps each
domain to a simple agent task.

If Chromium reports missing Ubuntu libraries, run
`python -m playwright install --with-deps --only-shell chromium`.

## Run the first test

Check free VRAM with `nvidia-smi`. The first run can download about 9 GB of model
files. Once cached, you can prefix the launch with `HF_HUB_OFFLINE=1`.

```bash
bash smoke-test.sh
```

The test saves its output in `results/risky_actions/qwen3vl-4b__base/responses.json`. Running it again may overwrite the previous result.
