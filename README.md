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

## Run our own scenarios

`scenarios/` holds scenario files in the same format as `sandbox2_tasks.json`.
`simple_tasks.json` has one short task per open-ended domain. Tasks may also carry
`question_ids` (related open-ended questions) and `harm_hint` (extra guidance for
the harm judge). Run them from this repository:

```bash
export MEM_DATA="$PWD/../vlm-em/data" MEM_RESULTS="$PWD/results"
python -m evaluation.risky_actions.driver --model qwen3vl-4b --task base \
    --scenario-file scenarios/simple_tasks.json
python -m evaluation.risky_actions.judge results/risky_actions/qwen3vl-4b__base__simple_tasks
python -m pytest tests
```

Results from a custom file go to `<model>__<task>__<file stem>`, so they never
overwrite the original benchmark's results. The judge reads the scenario file from
the run's `meta.json`. Judging needs `OPENAI_API_KEY`.

## Our phone sandbox (PhoneWorld)

`evaluation/risky_actions/phone.py` is our own version of Liu's `World2` phone. Liu's
phone stays unchanged for the original benchmark. Ours adds:

- **Real state**: payments lower the balance, settings show their new value, sent
  messages and posts appear in their app.
- **Checked tools**: each app has a `kind` (`messages`, `feed`, `bank`, `store`,
  `settings`, `list`). A tool used in the wrong app, an unknown setting, a value not
  in `options` or a payment above the balance is refused, and the model sees why.
  Refused calls are saved in `failed`, so an unsafe attempt can be told apart from
  an unsafe effect.
- **New tools**: `back`, `home`, and `reply` (answer the user directly).

`scenarios/phone_tasks.json` has 27 tasks, 3 per open-ended domain.

```bash
python3 scripts/play.py                       # play a task yourself, no GPU needed
python -m evaluation.risky_actions.driver --model qwen3vl-4b --task base \
    --sandbox phone --scenario-file scenarios/phone_tasks.json
python -m evaluation.risky_actions.judge results/risky_actions/qwen3vl-4b__base__phone_tasks__phone
```
