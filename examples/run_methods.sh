#!/usr/bin/env bash
set -euo pipefail
# Usage: bash examples/run_methods.sh qwen25 /path/to/model data/medqa5.jsonl outputs
profile=${1:?model profile required}
checkpoint=${2:?checkpoint required}
dataset=${3:?dataset required}
output_root=${4:?output root required}
case "$dataset" in
  *.csv) data_type=medbullets_csv ;;
  *.jsonl) data_type=medqa_jsonl ;;
  *) echo "Use a CSV or JSONL dataset" >&2; exit 2 ;;
esac
# Additional method-specific settings should be identical across compared models.
for method in sp cot sc self_refine debate css; do
  common=(--model "$profile" --method "$method" --data_path "$dataset")
  output="$output_root/$profile/$method"
  quant=()
  if [[ "$profile" == ministral ]]; then quant=(--dequantize_fp8_to_bf16); fi
  case "$method" in
    sp|cot) python run.py "${common[@]}" --model_path "$checkpoint" --data_type "$data_type" --out_dir "$output" "${quant[@]}" ;;
    sc) python run.py "${common[@]}" --model_path "$checkpoint" --data_type "$data_type" --save_dir "$output" --prompt_type cot --num_samples 40 "${quant[@]}" ;;
    css) python run.py "${common[@]}" --agent_model_path "$checkpoint" --save_dir "$output" "${quant[@]}" ;;
    *) python run.py "${common[@]}" --model_path "$checkpoint" --data_type "$data_type" --save_dir "$output" --save_traces "${quant[@]}" ;;
  esac
done
