#!/usr/bin/env python
"""
Thin wrapper so you can run the pipeline without `pip install -e .`:

    python scripts/run_pipeline.py --model_key llama3_1_8b_instruct --lang_code hi \
        --device cuda --output_dir outputs/llama3_1_8b_instruct-hi \
        --prompts_file scripts/prompts_example.txt --gammas 0,1,2

If the package *is* installed, `foxp2-run ...` (same arguments) works too.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_foxp2.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
