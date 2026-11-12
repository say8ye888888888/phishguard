"""Verify saved split integrity, metrics and all three UI flows."""
import json
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from streamlit.testing.v1 import AppTest
from core import ART, ROOT

def main():
    split=pd.read_csv(ART/'split.csv')
    assert split.groupby('group').split.nunique().max()==1,'Duplicate group leakage'
    assert split.id.nunique()==len(split),'Duplicate sample IDs'
    assert set(split.split)=={'train','validation','test'}
    predictions=pd.read_csv(ART/'predictions.csv')
    assert set(predictions.id)==set(split.loc[split.split=='test','id'])
    results=json.loads((ART/'results.json').read_text())
    pairs=[('Logistic Regression','lr'),('Linear SVM','svm')]
    if 'DistilBERT' in results:
        import hashlib
        assert results['DistilBERT']['split_sha256']==hashlib.sha256((ART/'split.csv').read_bytes()).hexdigest()
        pairs.append(('DistilBERT','distilbert'))
    for name,key in pairs:
        assert abs(f1_score(predictions.label,predictions[key+'_pred'])-results[name]['f1'])<1e-12
        assert abs(accuracy_score(predictions.label,predictions[key+'_pred'])-results[name]['accuracy'])<1e-12
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
    assert not app.exception,app.exception
    app.button[0].click().run()
    app.button[2].click().run(timeout=30)
    assert not app.exception,app.exception
    assert any('疑似钓鱼邮件' in m.value for m in app.markdown),'Demo prediction missing'
    if 'DistilBERT' in results:
        app.selectbox[0].select('DistilBERT').run()
        app.button[2].click().run(timeout=60)
        assert not app.exception,app.exception
        assert not app.error,'Transformer UI inference failed'
        assert any('模型：DistilBERT' in m.value for m in app.markdown),'Transformer result missing'
    for page in ['实验看板','错误案例']:
        app.radio[0].set_value(page).run(timeout=30)
        assert not app.exception,app.exception
        if 'DistilBERT' in results:
            app.selectbox[0].select('DistilBERT' if page=='实验看板' else 'distilbert').run(timeout=30)
            assert not app.exception,app.exception
            if page=='实验看板':
                assert app.metric[0].value==f"{results['DistilBERT']['accuracy']:.2%}"
            else:
                assert len(app.expander)>0,'DistilBERT error cases not displayed'
    print('PASS: group isolation, sample IDs, saved metrics, prediction, dashboard, error cases')

if __name__=='__main__': main()
