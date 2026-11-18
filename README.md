# Swedish repetition-loop weight edit vs controls
```
pip install -r requirements.txt
python tests/test_metrics.py
python run_experiment.py --smoke --out results_smoke          # sanity check, minutes
python run_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct --revision <commit> --out results   # seed 0
python build_pdf.py                                           # RESEARCH_NOTE.md -> RESEARCH_NOTE.pdf
```
Outputs: `results/results.json` (config, dev grid, chosen layer/alpha, test metrics + bootstrap CIs), `examples.json`, `prompts.json`. See `RESEARCH_NOTE.md`, `AI_USAGE.md`.
