"""Five-fold NetTCR-2.0 training/evaluation for a prepared split dataset."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from PhongLu.Modeleval.models.nettcr2 import build_model, encode

def inputs(data): return [encode(data.cdr3_beta, 30), encode(data.peptide, 9)]
class BatchSequence(tf.keras.utils.Sequence):
    """Encode only one batch at a time, avoiding full-dataset RAM allocation."""
    def __init__(self, data, batch_size=256, shuffle=False):
        super().__init__()
        self.data=data.reset_index(drop=True); self.batch_size=batch_size; self.shuffle=shuffle; self.order=np.arange(len(data)); self.on_epoch_end()
    def __len__(self): return int(np.ceil(len(self.data)/self.batch_size))
    def __getitem__(self, index):
        rows=self.order[index*self.batch_size:(index+1)*self.batch_size]; batch=self.data.iloc[rows]
        return tuple(inputs(batch)), batch.binder.to_numpy(dtype=np.float32)
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.order)
def metrics(y, scores):
    labels=(scores >= .5).astype(int)
    return {"auroc":float(roc_auc_score(y,scores)),"auprc":float(average_precision_score(y,scores)),"f1_at_0_5":float(f1_score(y,labels)),"precision_at_0_5":float(precision_score(y,labels,zero_division=0)),"recall_at_0_5":float(recall_score(y,labels,zero_division=0))}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--splits',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);p.add_argument('--seed',type=int,default=13);p.add_argument('--epochs',type=int,default=30);a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); tf.keras.utils.set_random_seed(a.seed); data=pd.read_csv(a.splits); a.outdir.mkdir(parents=True,exist_ok=True); all_metrics=[]
    for test_fold in range(5):
        validation_fold=(test_fold+1)%5; train=data[~data.fold.isin([test_fold,validation_fold])]; valid=data[data.fold.eq(validation_fold)]; test=data[data.fold.eq(test_fold)]
        model=build_model(); callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
        model.fit(BatchSequence(train, shuffle=True),validation_data=BatchSequence(valid),epochs=a.epochs,verbose=0,callbacks=[callback])
        scores=model.predict(BatchSequence(test),verbose=0).ravel(); row={"fold":test_fold,**metrics(test.binder.to_numpy(),scores),"train_rows":len(train),"validation_rows":len(valid),"test_rows":len(test)};all_metrics.append(row); pd.DataFrame({"cdr3_beta":test.cdr3_beta,"peptide":test.peptide,"binder":test.binder,"score":scores}).to_csv(a.outdir/f'fold_{test_fold}_predictions.csv',index=False)
    pd.DataFrame(all_metrics).to_csv(a.outdir/'fold_metrics.csv',index=False); summary=pd.DataFrame(all_metrics).drop(columns=['fold','train_rows','validation_rows','test_rows']).agg(['mean','std']).to_dict(); (a.outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(pd.DataFrame(all_metrics).to_string(index=False))
if __name__=='__main__': main()
