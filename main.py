"""Text classification experiment. Run: python main.py --data-dir PATH --output-dir PATH"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from threadpoolctl import threadpool_limits

SEED = 42

def vectorizer():
    return TfidfVectorizer(lowercase=True, stop_words='english', max_features=10000,
                           min_df=2, max_df=0.95, sublinear_tf=True, ngram_range=(1, 2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = a.data_dir / 'train_data.csv'; test_path = a.data_dir / 'test_data_unlabeled.csv'
    train = pd.read_csv(train_path); test = pd.read_csv(test_path)
    if not {'text', 'target'}.issubset(train.columns) or 'text' not in test:
        raise ValueError('Required columns: train=text,target; test=text')
    if train[['text','target']].isna().any().any() or test.text.isna().any():
        raise ValueError('Missing text or target values')
    if train.text.duplicated().any():
        raise ValueError('Duplicate training texts: use grouped splitting before evaluation')
    texts = train.text.astype(str); y = train.target.to_numpy()
    ti, vi = train_test_split(np.arange(len(train)), test_size=.2, stratify=y, random_state=SEED)
    v = vectorizer(); xt = v.fit_transform(texts.iloc[ti]); xv = v.transform(texts.iloc[vi])
    yt, yv = y[ti], y[vi]; classes = np.unique(y)
    rows = []; models = {}; histories = {}; epochs = {}
    def record(name, model, **params):
        pred = model.predict(xv)
        rows.append(dict(model=name, accuracy=float(accuracy_score(yv,pred)),
                         macro_f1=float(f1_score(yv,pred,average='macro')), parameters=params))
        models[name] = model
        print(name, rows[-1]['accuracy'], rows[-1]['macro_f1'], flush=True)
    for c in [.1, 1., 10.]:
        name = f'SVM_C{c:g}'; m = SVC(kernel='linear', C=c, random_state=SEED)
        m.fit(xt, yt); record(name,m,C=c,kernel='linear')
        name = f'LR_C{c:g}'; m = LogisticRegression(C=c,max_iter=2000,random_state=SEED)
        m.fit(xt,yt); record(name,m,C=c,max_iter=2000)
    for alpha in [.0001,.01]:
        name = f'MLP_alpha{alpha:g}'
        m = MLPClassifier(hidden_layer_sizes=(50,),alpha=alpha,activation='relu',
                          learning_rate_init=.001,batch_size=128,random_state=SEED)
        history = []; best_loss = float('inf'); best_model = None; stale = 0
        for epoch in range(1,61):
            m.partial_fit(xt,yt,classes=classes)
            tl = log_loss(yt,m.predict_proba(xt),labels=classes)
            vl = log_loss(yv,m.predict_proba(xv),labels=classes)
            history.append(dict(epoch=epoch,train_loss=tl,val_loss=vl,
                                val_accuracy=float(accuracy_score(yv,m.predict(xv)))))
            if vl < best_loss - .0001:
                best_loss = vl; best_model = copy.deepcopy(m); epochs[name]=epoch; stale=0
            else: stale += 1
            if stale >= 8: break
        histories[name] = history
        record(name,best_model,hidden_layer_sizes=[50],alpha=alpha,best_epoch=epochs[name],epochs_run=len(history))
    best = max(rows,key=lambda r:(r['macro_f1'],r['accuracy']))
    chosen = best['model']; pred_val=models[chosen].predict(xv)
    pd.DataFrame(confusion_matrix(yv,pred_val,labels=classes),index=classes,columns=classes).to_csv(a.output_dir/'confusion_matrix.csv')
    pd.DataFrame(rows).to_csv(a.output_dir/'validation_results.csv',index=False)
    pd.DataFrame([dict(model=n,**h) for n,hs in histories.items() for h in hs]).to_csv(a.output_dir/'loss_history.csv',index=False)
    fig,axs = plt.subplots(1,2,figsize=(10,3.2))
    for ax,(name,hs) in zip(axs,histories.items()):
        ax.plot([h['epoch'] for h in hs],[h['train_loss'] for h in hs],label='Train cross-entropy')
        ax.plot([h['epoch'] for h in hs],[h['val_loss'] for h in hs],label='Validation cross-entropy')
        ax.axvline(epochs[name],color='gray',linestyle=':',label='Selected epoch')
        ax.set(title=name,xlabel='Epoch',ylabel='Cross-entropy');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(a.output_dir/'loss_curve.png',dpi=180);plt.close(fig)
    vf=vectorizer();xf=vf.fit_transform(texts);xu=vf.transform(test.text.astype(str))
    final=copy.deepcopy(models[chosen]); final.set_params(random_state=SEED)
    if chosen.startswith('MLP'):
        final=MLPClassifier(hidden_layer_sizes=(50,),alpha=best['parameters']['alpha'],
                           learning_rate_init=.001,batch_size=128,random_state=SEED)
        for _ in range(epochs[chosen]):final.partial_fit(xf,y,classes=classes)
    else:final.fit(xf,y)
    predictions=final.predict(xu)
    path=a.output_dir/'prediction.csv';pd.DataFrame(predictions).to_csv(path,index=False,header=False)
    check=pd.read_csv(path,header=None)
    assert check.shape==(len(test),1) and set(check[0]).issubset(set(classes))
    result=dict(seed=SEED,train_rows=len(train),test_rows=len(test),fit_rows=len(ti),validation_rows=len(vi),
                class_counts={str(k):int(v) for k,v in train.target.value_counts().sort_index().items()},
                vocabulary_size=len(v.vocabulary_),selected=best,results=rows,histories=histories,
                sklearn_version=sklearn.__version__,test_accuracy=None,
                hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [train_path,test_path]})
    (a.output_dir/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Selected:',chosen,'; saved',path,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
