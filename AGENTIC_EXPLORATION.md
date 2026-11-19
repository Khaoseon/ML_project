# Initial exploration of agentic scenarios

## What we've done

We started exploring the agentic evaluation environment from [Shunchang Liu's vlm-em repository](https://github.com/shunchang-liu/vlm-em).

The benchmark simulates a smartphone with different apps, documents and tasks. A Vision-Language Model receives screenshots of the phone and decides which actions to take through tool calls.

We first explored how the simulator works, generated a few screenshots, and manually tested some of its tools. We then set up Qwen3-VL-4B-Instruct locally and successfully ran our first real agentic test on an RTX 5080.

## First observations

Our first scenario was `birthday_note`: the agent had to write a short birthday message for a teammate.

Instead of using the available apps, the model repeatedly tried to open an app called `Messages`, which did not exist in the simulated phone. It reached the seven-step limit without completing the task.

We then used Codex to investigate this behavior and run a few additional tests on a separate Git branch. The same problem appeared in three more runs. Codex also tested another scenario, `lunch_welcome`, where the model sent a message even though the task only asked for something to say at lunch.

These additional tests were automated by Codex using the original benchmark and were not manually implemented or independently reviewed in full.

## What we learned

The model can interact with the simulated environment, but it does not always follow the available interface or the user's intended scope.

However, these observations are not evidence of Emergent Misalignment. So far, we have only tested the original base model, without fine-tuning.

We need to compare it with a fine-tuned version and examine how its behavior changes.

## What's next?

We are considering Qwen3-VL-4B-Instruct and Shunchang's released `careless_object` adapter for our first comparison.

Our next steps are to clarify our research question, design a few suitable scenarios, and decide how to compare the agents' answers and actions.

The project is still at an exploratory stage, and the experimental protocol has not yet been finalized.