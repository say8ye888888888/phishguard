"""Optional GPU training; fixed split from train.py. Ablation stays on validation."""
import argparse,json,time,random,platform,os,hashlib
from pathlib import Path
os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parent/'.cache'/'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding,set_seed,TrainerCallback
from core import ART,metrics

class Emails(torch.utils.data.Dataset):
    def __init__(self,frame,tokenizer,length):
        self.enc=tokenizer(frame.text.tolist(),truncation=True,max_length=length)
        self.labels=frame.label.tolist()
    def __len__(self): return len(self.labels)
    def __getitem__(self,i): return {**{k:v[i] for k,v in self.enc.items()},'labels':self.labels[i]}

def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)

class Progress(TrainerCallback):
    def on_log(self,args,state,control,logs=None,**kwargs):
        value={'step':state.global_step,'total_steps':state.max_steps,'epoch':state.epoch,**(logs or {})}
        write_json(ART/'transformer-run'/'progress.json',value)
        print(json.dumps(value),flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--lengths',nargs='+',type=int,default=[256]); p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=16); p.add_argument('--threads',type=int,default=6)
    p.add_argument('--bf16',action='store_true',help='CPU mixed precision on supported hardware')
    p.add_argument('--model',default='distilbert/distilbert-base-uncased'); args=p.parse_args()
    if any(n not in [128,256,512] for n in args.lengths): p.error('lengths must be 128,256,512')
    if min(args.batch_size,args.threads,args.epochs)<1: p.error('batch-size, threads, epochs must be positive')
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    run=ART/'transformer-run'; run.mkdir(exist_ok=True)
    split_hash=hashlib.sha256((ART/'split.csv').read_bytes()).hexdigest()
    df=pd.read_csv(ART/'split.csv'); train=df[df.split=='train']; val=df[df.split=='validation']; test=df[df.split=='test']
    assert df.groupby('group').split.nunique().max()==1 and df.id.is_unique,'Invalid data split'
    run_config=dict(**vars(args),seed=42,split_sha256=split_hash,torch=torch.__version__,hardware=platform.processor(),train_samples=len(train),validation_samples=len(val),test_samples=len(test))
    write_json(run/'run_config.json',run_config)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    summaries=[]; best=-1; selected=None
    for length in args.lengths:
        set_seed(42)
        model=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=2,id2label={0:'Legitimate',1:'Phishing'},label2id={'Legitimate':0,'Phishing':1},attn_implementation='sdpa')
        cfg=TrainingArguments(output_dir=str(run/f'checkpoints-{length}'),learning_rate=2e-5,per_device_train_batch_size=args.batch_size,per_device_eval_batch_size=32,num_train_epochs=args.epochs,eval_strategy='epoch',save_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='f1',save_total_limit=1,save_only_model=True,report_to='none',seed=42,fp16=torch.cuda.is_available() and not args.bf16,bf16=args.bf16,logging_steps=25,disable_tqdm=True,dataloader_num_workers=0,dataloader_pin_memory=torch.cuda.is_available(),optim='adamw_torch',weight_decay=0.01)
        def compute(out):
            logits,y=out; scores=torch.softmax(torch.tensor(logits),dim=-1).numpy()[:,1]
            return {'f1':metrics(y,logits.argmax(-1),scores)['f1']}
        trainer=Trainer(model=model,args=cfg,train_dataset=Emails(train,tokenizer,length),eval_dataset=Emails(val,tokenizer,length),data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=compute,callbacks=[Progress()])
        started=time.perf_counter(); trainer.train(); elapsed=time.perf_counter()-started
        v=trainer.predict(Emails(val,tokenizer,length)); score=compute((v.predictions,v.label_ids))['f1']
        lengths=[len(tokenizer(t,truncation=False,verbose=False)['input_ids']) for t in val.text]
        summaries.append(dict(max_length=length,validation_f1=score,training_seconds=elapsed,validation_truncated_fraction=float(np.mean(np.array(lengths)>length))))
        write_json(run/'validation_results.json',summaries)
        write_json(run/f'history-{length}.json',trainer.state.log_history)
        if score>best:
            best=score; selected=length
            trainer.save_model(str(run/'best-model')); tokenizer.save_pretrained(run/'best-model')
        del trainer,model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    assert hashlib.sha256((ART/'split.csv').read_bytes()).hexdigest()==split_hash,'Split changed during training'
    model=AutoModelForSequenceClassification.from_pretrained(run/'best-model',attn_implementation='sdpa')
    trainer=Trainer(model=model,args=TrainingArguments(output_dir=str(run/'inference'),per_device_eval_batch_size=32,report_to='none',disable_tqdm=True,dataloader_pin_memory=torch.cuda.is_available()),data_collator=DataCollatorWithPadding(tokenizer))
    dataset=Emails(test,tokenizer,selected)
    trainer.predict(torch.utils.data.Subset(dataset,list(range(min(5,len(dataset))))))
    started=time.perf_counter(); out=trainer.predict(dataset); elapsed=time.perf_counter()-started
    pred=out.predictions.argmax(-1); scores=torch.softmax(torch.tensor(out.predictions),dim=-1).numpy()[:,1]
    r=metrics(test.label,pred,scores); r.update(validation_f1=best,max_length=selected,ms_per_email=elapsed/len(test)*1000,test_samples=len(test),hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),timing='batch 32, warmup, tokenization excluded',threads=args.threads,epochs=args.epochs,split_sha256=split_hash,training_seconds=next(s['training_seconds'] for s in summaries if s['max_length']==selected))
    results=json.loads((ART/'results.json').read_text()) if (ART/'results.json').exists() else {}
    results['DistilBERT']=r
    existing=pd.read_csv(ART/'predictions.csv') if (ART/'predictions.csv').exists() else test[['id','text','label']].copy()
    assert set(existing.id)==set(test.id),'Existing predictions belong to another split'
    # Merge by stable ID rather than assuming CSV row order.
    existing=existing.drop(columns=['distilbert_pred','distilbert_score'],errors='ignore').merge(pd.DataFrame({'id':test.id,'distilbert_pred':pred,'distilbert_score':scores}),on='id',validate='one_to_one')
    if (ART/'distilbert').exists(): (ART/'distilbert').rename(ART/f'distilbert-backup-{int(time.time())}')
    (run/'best-model').rename(ART/'distilbert')
    write_json(ART/'transformer_config.json',dict(**run_config,max_length=selected))
    existing.to_csv(ART/'predictions.csv',index=False)
    write_json(ART/'results.json',results)
    write_json(ART/'ablation.json',summaries)
    name=max(results,key=lambda k: results[k]['validation_f1'])
    (ART/'selected.json').write_text(json.dumps({'name':name,'key':{'Logistic Regression':'lr','Linear SVM':'svm','DistilBERT':'distilbert'}[name]}))
    write_json(run/'progress.json',{'stage':'complete','selected_model':name,'test_metrics':r})
    print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__': main()
