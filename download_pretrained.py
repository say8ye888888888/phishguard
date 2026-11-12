"""Fetch official base weights at a pinned revision, checking file sizes and SHA256."""
import hashlib
import json
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
MODEL = 'distilbert/distilbert-base-uncased'
REVISION = '12040accade4e8a0f71eabdb258fecc2e7e948be'
DEST = ROOT / '.cache' / 'distilbert-base-uncased'

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    response=requests.get(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}',params={'blobs':'true'},timeout=60)
    response.raise_for_status(); metadata=response.json()
    required={'config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt'}
    found={s['rfilename']:s for s in metadata['siblings'] if s['rfilename'] in required}
    if set(found)!=required: raise ValueError('Incomplete upstream model')
    for name,entry in found.items():
        dest=DEST/name
        expected=entry.get('lfs',{}).get('sha256')
        if dest.exists() and dest.stat().st_size==entry['size'] and (not expected or hashlib.sha256(dest.read_bytes()).hexdigest()==expected):
            print('Already verified:',name,flush=True); continue
        for attempt in range(3):
            try:
                with requests.get(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?download=true',stream=True,timeout=(30,120)) as r:
                    r.raise_for_status()
                    digest=hashlib.sha256(); size=0
                    temporary=dest.with_suffix(dest.suffix+'.part')
                    with temporary.open('wb') as stream:
                        for chunk in r.iter_content(1024*1024):
                            stream.write(chunk); digest.update(chunk); size+=len(chunk)
                if size!=entry['size'] or (expected and digest.hexdigest()!=expected): raise ValueError('Size/hash mismatch: '+name)
                temporary.replace(dest)
                print('Verified:',name,size,flush=True); break
            except requests.RequestException:
                if attempt==2: raise
                time.sleep(2)
    (DEST/'source.json').write_text(json.dumps({'model':MODEL,'revision':REVISION,'license':'apache-2.0','files':found},indent=2))
    print('Ready:',DEST,flush=True)

if __name__=='__main__': main()
