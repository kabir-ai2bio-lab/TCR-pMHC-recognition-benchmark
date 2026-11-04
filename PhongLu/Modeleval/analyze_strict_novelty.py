"""Peptide-novelty and error analysis for completed strict-split predictions."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def identity(a: str, b: str) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--splits',type=Path,default=Path('data/splits/strict_five_fold.csv'));p.add_argument('--predictions-dir',type=Path,default=Path('results/nettcr2_vdjdb_strict'));p.add_argument('--outdir',type=Path,default=Path('results/nettcr2_vdjdb_strict/analysis'));a=p.parse_args(); a.outdir.mkdir(parents=True,exist_ok=True)
    splits=pd.read_csv(a.splits); analyses=[]
    for fold in range(5):
        pred=pd.read_csv(a.predictions_dir/f'fold_{fold}_predictions.csv'); train_peptides=splits.loc[splits.fold.ne(fold),'peptide'].unique()
        similarities={pep:max(identity(pep,train) for train in train_peptides) for pep in pred.peptide.unique()}
        pred['max_train_peptide_identity']=pred.peptide.map(similarities); pred['min_train_hamming_distance']=(9*(1-pred.max_train_peptide_identity)).round().astype(int); pred['fold']=fold
        pred['error_type']=np.select([(pred.binder.eq(1)&pred.score.lt(.5)),(pred.binder.eq(0)&pred.score.ge(.5))],['false_negative','false_positive'],'correct')
        analyses.append(pred)
    data=pd.concat(analyses,ignore_index=True); data.to_csv(a.outdir/'strict_predictions_with_novelty.csv',index=False)
    rows=[]
    for peptide, group in data.groupby('peptide'):
        row={'peptide':peptide,'n':len(group),'positives':int(group.binder.sum()),'negatives':int((1-group.binder).sum()),'max_train_peptide_identity':group.max_train_peptide_identity.iloc[0],'min_train_hamming_distance':int(group.min_train_hamming_distance.iloc[0]),'mean_score':float(group.score.mean())}
        row['auroc']=float(roc_auc_score(group.binder,group.score)) if group.binder.nunique()==2 else np.nan; rows.append(row)
    per_peptide=pd.DataFrame(rows); per_peptide.to_csv(a.outdir/'per_peptide_novelty.csv',index=False)
    error=data.groupby(['error_type','min_train_hamming_distance']).size().rename('n').reset_index(); error.to_csv(a.outdir/'error_counts_by_novelty.csv',index=False)
    summary={'n_predictions':len(data),'false_negative_rate':float(((data.binder.eq(1))&(data.score.lt(.5))).mean()),'false_positive_rate':float(((data.binder.eq(0))&(data.score.ge(.5))).mean()),'mean_max_identity':float(data.max_train_peptide_identity.mean()),'mean_min_hamming_distance':float(data.min_train_hamming_distance.mean()),'peptides_with_both_classes_for_auroc':int(per_peptide.auroc.notna().sum())}
    (a.outdir/'novelty_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
