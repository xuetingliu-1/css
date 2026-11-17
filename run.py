#!/usr/bin/env python3

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from cssbench.models import MODELS, select_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--method', choices=['sp', 'cot', 'sc', 'self_refine', 'debate', 'css'], required=True)
    if len(sys.argv) == 1 or sys.argv[1:] == ['--help']:
        parser.print_help()
        print('\nUse --model KEY --method METHOD --help to see method-specific options.')
        return
    args, rest = parser.parse_known_args()
    select_model(args.model)
    engine = 'sp_cot' if args.method in ('sp', 'cot') else args.method
    if args.method in ('sp', 'cot'):
        if '--mode' in rest or any(x.startswith('--mode=') for x in rest):
            parser.error('SP/CoT mode is selected by --method; omit --mode')
        rest += ['--mode', args.method]
    for path_flag in ('--model_path', '--agent_model_path'):
        value = rest[rest.index(path_flag)+1] if path_flag in rest else next((x.split('=',1)[1] for x in rest if x.startswith(path_flag+'=')), None)
        if value:
            known = [k for k,(repo,_) in MODELS.items() if value.rstrip('/').split('/')[-1].lower()==repo.split('/')[-1].lower()]
            if known and known[0] != args.model:
                parser.error('--model profile does not match the supplied checkpoint path')
    if args.method == 'sc' and '--help' not in rest:
        sc_parser=argparse.ArgumentParser(add_help=False)
        sc_parser.add_argument('--prompt_type',default='sp')
        sc_parser.add_argument('--num_samples',type=int,default=1)
        sc_args,_=sc_parser.parse_known_args(rest)
        if sc_args.prompt_type != 'cot' or sc_args.num_samples <= 1:
            print('NOTE: SC sampling requires --prompt_type cot --num_samples K (K > 1); current baseline settings run single-pass greedy.',file=sys.stderr)
    root = Path(__file__).parent
  
    for flag in ['--save_dir', '--out_dir']:
        value = None
        if flag in rest:
            value = rest[rest.index(flag) + 1]
        else:
            value = next((x.split('=', 1)[1] for x in rest if x.startswith(flag + '=')), None)
        if value and '--help' not in rest:
            d = Path(value); d.mkdir(parents=True, exist_ok=True)
            manifest = {'profile': args.model, 'hf_model_id': MODELS[args.model][0], 'method': args.method,
                        'argv': sys.argv[1:], 'qwen3_enable_thinking': False,
                        'prompt_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'cssbench/prompts').glob('*.py'))}}
            (d/'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    sys.argv = [sys.argv[0]] + rest
    importlib.import_module('cssbench.methods.' + engine).main()

if __name__ == '__main__':
    main()
