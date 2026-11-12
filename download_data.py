"""Download the public dataset via its official parquet conversion."""
import io
import requests
import pandas as pd
from core import ROOT

def main():
    endpoint='https://datasets-server.huggingface.co/parquet'
    r=requests.get(endpoint,params={'dataset':'zefang-liu/phishing-email-dataset'},timeout=60)
    r.raise_for_status()
    files=r.json()['parquet_files']
    frames=[]
    for item in files:
        response=requests.get(item['url']+'?download=true',timeout=180)
        response.raise_for_status()
        frames.append(pd.read_parquet(io.BytesIO(response.content)))
    path=ROOT/'data/emails.csv'; path.parent.mkdir(exist_ok=True)
    data=pd.concat(frames,ignore_index=True); data.to_csv(path,index=False)
    print(f'Saved {len(data)} emails to {path}')

if __name__=='__main__': main()
