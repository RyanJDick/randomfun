# Kernel Bot Submissions

The Popcorn CLI is intended to submit kernels to leaderboards on GPU Mode's Kernel Bot.

## Objective 

Agents must write CUDA C and C++ kernels and integrate them into the single-file `submission.py` workflow.

Do not submit pure PyTorch based optimization. The objective of the task is to use any kernel DSL like CUDA or Triton to improved *beyond* the performance of native PyTorch.

## Skills
A skill is a local instruction bundle stored in `SKILL.md`.

### Available skills
- popcorn-submission-workflow: Helps with Popcorn CLI registration, submission setup, submission modes, and file directives. (file: /Users/ryan/src/randomfun/gpu_mode/pmpp_v2/vectorsum_py/.popcorn/skills/popcorn-submission-workflow/SKILL.md)
- load-inline-native-code: Helps write CUDA and HIP kernels using torch.utils.cpp_extension.load_inline(). Use when writing native GPU code inside a Python submission. (file: /Users/ryan/src/randomfun/gpu_mode/pmpp_v2/vectorsum_py/.popcorn/skills/load-inline-native-code/SKILL.md)

### How to use skills
- Load the skill by reading its `SKILL.md` file when user requests match the description.
- Follow progressive disclosure: read only relevant referenced files/scripts as needed.
- Keep the workspace setup aligned with `popcorn setup`.
