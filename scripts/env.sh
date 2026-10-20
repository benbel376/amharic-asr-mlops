# Source this in every new Git Bash window:   source scripts/env.sh
if [ -f .venv/Scripts/activate ]; then source .venv/Scripts/activate; else source .venv/bin/activate; fi
export PYTHONPATH=src                 # lets "python -m asr.<module>" find our code
export PYTHONIOENCODING=utf-8         # prints Amharic correctly in Git Bash
export MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
echo "environment ready: $(python --version)"
