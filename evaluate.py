#!/usr/bin/env python3
"""Accuracy and fixed-class macro/weighted F1 on saved final labels (no raw-text reparse)."""
import argparse
import csv
import json
from pathlib import Path


def metrics(gold, predictions, options):
    labels = list('ABCDE'[:options])
    if not gold or len(gold) != len(predictions):
        raise ValueError('Gold/prediction counts must be equal and nonzero')
    if any(g not in labels for g in gold):
        raise ValueError('Gold labels must be valid A-D/A-E letters')
    if any(p not in labels + ['Z'] for p in predictions):
        raise ValueError('Predictions must be one valid option letter or Z; do not pass raw responses')
    per_class={}
    for label in labels:
        tp=sum(g==label and p==label for g,p in zip(gold,predictions))
        fp=sum(g!=label and p==label for g,p in zip(gold,predictions))
        fn=sum(g==label and p!=label for g,p in zip(gold,predictions))
        denom=2*tp+fp+fn
        per_class[label]={'f1':2*tp/denom if denom else 0.0,'support':gold.count(label),'tp':tp,'fp':fp,'fn':fn}
    n=len(gold)
    return {'n':n,'correct':sum(g==p for g,p in zip(gold,predictions)),
            'accuracy':sum(g==p for g,p in zip(gold,predictions))/n,
            'macro_f1':sum(v['f1'] for v in per_class.values())/options,
            'weighted_f1':sum(v['f1']*v['support'] for v in per_class.values())/n,
            'invalid_predictions':predictions.count('Z'),'labels':labels,'per_class':per_class}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--options',type=int,choices=[4,5],required=True)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.data.suffix=='.csv':
        with args.data.open(newline='') as f: data=list(csv.DictReader(f))
    else:data=[json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    gold=[str(row['answer_idx']).strip().upper() for row in data]
    lines=args.predictions.read_text().splitlines()
    if args.predictions.suffix=='.jsonl':
        rows=[json.loads(line) for line in lines]
        ids=[row.get('idx') for row in rows]
        if ids != list(range(len(data))):raise ValueError('JSONL idx must cover the dataset exactly once, in order')
        predictions=[str(row['pred']).strip().upper() for row in rows]
    else:predictions=[line.strip().upper() or 'Z' for line in lines]
    result=metrics(gold,predictions,args.options)
    result['metric_scale']='0_to_1';result['prediction_source']='saved_final_labels'
    text=json.dumps(result,indent=2)+'\n';print(text,end='')
    if args.output:args.output.write_text(text)

if __name__=='__main__':main()
