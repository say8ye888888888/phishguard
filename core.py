from pathlib import Path
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, precision_recall_curve

ROOT = Path(__file__).resolve().parent
ART = ROOT / 'artifacts'
ART.mkdir(exist_ok=True)

def normalize(text):
    text = re.sub(r'https?://\S+', ' URL ', str(text).lower())
    text = re.sub(r'\b\d+\b', ' NUMBER ', text)
    return re.sub(r'\s+', ' ', text).strip()

def metrics(y, pred, scores):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    p, r, _ = precision_recall_curve(y, scores)
    return dict(accuracy=float(accuracy_score(y, pred)), precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred, zero_division=0)), f1=float(f1_score(y, pred, zero_division=0)), ap=float(average_precision_score(y, scores)), fpr=float(fp / max(tn+fp, 1)), matrix=[[int(tn),int(fp)],[int(fn),int(tp)]], pr_precision=p[::max(1,len(p)//300)].tolist(), pr_recall=r[::max(1,len(r)//300)].tolist())
