"""Construct size- and peptide-distribution-matched strict/uniform split controls."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from PhongLu.data.preprocessing.build_splits import downsample_for_strict, _write_with_negatives
from PhongLu.data.preprocessing.split import assert_strict_no_leakage, make_folds

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('data/processed/vdjdb_v1_positive.csv'));p.add_argument('--output-dir',type=Path,default=Path('data/splits'));p.add_argument('--seed',type=int,default=13);a=p.parse_args()
    source=pd.read_csv(a.input); strict_ready, downsampling=downsample_for_strict(source,5,a.seed)
    counts=strict_ready.peptide.value_counts(); common=strict_ready.loc[strict_ready.peptide.isin(counts[counts>=5].index)].copy().reset_index(drop=True)
    strict=make_folds(common,'strict',seed=a.seed); assert_strict_no_leakage(strict); uniform=make_folds(common,'uniform',seed=a.seed)
    _write_with_negatives(strict,'matched_strict',a.output_dir,a.seed,source); _write_with_negatives(uniform,'matched_uniform',a.output_dir,a.seed,source)
    audit={'seed':a.seed,'same_positive_pool_rows':len(common),'same_unique_peptides':int(common.peptide.nunique()),'removed_peptides_fewer_than_5':int((counts<5).sum()),'strict_downsampling':downsampling,'negative_strategy':'fold-local rearrangement; exclude full positive pool and prior-fold negatives'}
    (a.output_dir/'matched_split_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__': main()
