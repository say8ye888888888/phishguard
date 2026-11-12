import json, re, time, os
from pathlib import Path
os.environ.setdefault('HF_HOME',str(Path(__file__).resolve().parent/'.cache'/'huggingface'))
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import ART

st.set_page_config(page_title='PhishGuard | Email Intelligence',page_icon='🛡️',layout='wide')
st.markdown('''<style>
.stApp{background:#f5f7fb;color:#18243b} .block-container{max-width:1240px;padding-top:2rem}
h1,h2,h3{letter-spacing:-.025em} [data-testid="stMetric"]{background:white;border:1px solid #e5eaf2;border-radius:16px;padding:20px}
[data-testid="stSidebar"]{background:#10223c} [data-testid="stSidebar"] *{color:#e7efff}
.hero{background:linear-gradient(115deg,#10223c,#214f83);border-radius:22px;padding:32px 38px;color:white;margin-bottom:24px}
.hero h1{color:white;margin:8px 0}.hero p{color:#c3d5ec;margin-bottom:0}.eyebrow{font-size:12px;letter-spacing:3px;color:#8fc8ff}
.verdict{border-radius:18px;padding:25px;background:#fff;border:1px solid #e4eaf2;margin-bottom:16px}
.stButton>button[kind="primary"]{background:#2364dc;border:0;border-radius:10px}
</style>''',unsafe_allow_html=True)

@st.cache_resource
def load_model(key): return joblib.load(ART/f'{key}.joblib')

def read_json(name):
    p=ART/name
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

@st.cache_resource
def load_transformer(model_version):
    import torch
    torch.set_num_threads(4)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    return AutoTokenizer.from_pretrained(ART/'distilbert',local_files_only=True),AutoModelForSequenceClassification.from_pretrained(ART/'distilbert',local_files_only=True).eval()

def predict_one(key,text):
    t=time.perf_counter()
    if key=='distilbert':
        import torch
        tokenizer,model=load_transformer((ART/'distilbert/model.safetensors').stat().st_mtime_ns)
        config=read_json('transformer_config.json')
        inputs=tokenizer(text,return_tensors='pt',truncation=True,max_length=config.get('max_length',256))
        with torch.inference_mode(): pred=int(model(**inputs).logits.argmax(-1).item())
    else: pred=int(load_model(key).predict([text])[0])
    return pred,(time.perf_counter()-t)*1000

with st.sidebar:
    st.markdown('## 🛡️ PhishGuard')
    st.caption('NLP · EMAIL INTELLIGENCE')
    page=st.radio('工作空间',['邮件检测','实验看板','错误案例'],label_visibility='collapsed')
    st.divider()
    st.caption('英文邮件 · 二分类\n\n预测基于邮件文本，不能验证发件人身份或链接安全性。')
    progress=read_json('transformer-run/progress.json')
    if progress and progress.get('total_steps') and not (ART/'distilbert/config.json').exists():
        st.divider()
        st.caption('DistilBERT 正在训练')
        st.progress(min(progress.get('step',0)/progress['total_steps'],1.0))
        st.caption(f"{progress.get('step',0)} / {progress['total_steps']} 步；切换页面可刷新。")

st.markdown('<div class="hero"><div class="eyebrow">PHISHGUARD / NLP LAB</div><h1>让可疑邮件，更容易被识别。</h1><p>传统机器学习与 Transformer 的邮件文本检测实验平台</p></div>',unsafe_allow_html=True)
results=read_json('results.json')
available={n:k for n,k in [('Logistic Regression','lr'),('Linear SVM','svm')] if (ART/f'{k}.joblib').exists()}
if (ART/'distilbert/config.json').exists() and 'DistilBERT' in results: available['DistilBERT']='distilbert'

if page=='邮件检测':
    left,right=st.columns([1.35,1],gap='large')
    with left:
        st.subheader('分析一封邮件')
        st.caption('粘贴英文邮件正文；示例仅供交互演示，不计入测试指标。')
        a,b=st.columns(2)
        if a.button('填入可疑示例',use_container_width=True):
            st.session_state.email='Subject: Urgent account verification\nYour account will be suspended within 24 hours. Verify your password immediately at https://account-check.example/login to avoid losing access.'
        if b.button('填入正常示例',use_container_width=True):
            st.session_state.email='Subject: Project meeting\nHi team, our project meeting is scheduled for Thursday at 2 PM in room 301. Please bring your notes. Thanks, Alex.'
        text=st.text_area('邮件内容',height=270,key='email',placeholder='Subject: ...\nPaste the email body here...')
        if available:
            selected=read_json('selected.json').get('name')
            names=list(available)
            model_name=st.selectbox('检测模型',names,index=names.index(selected) if selected in names else 0)
            run=st.button('开始分析 →',type='primary',use_container_width=True,disabled=not text.strip())
        else:
            st.info('模型尚未训练。请先在项目目录运行 python train.py。')
            run=False
    with right:
        st.subheader('检测结果')
        if run:
            if len(text)>200000: st.error('邮件过长，请限制在 200,000 字符以内。')
            else:
                try:
                    with st.spinner('正在分析邮件…'): pred,ms=predict_one(available[model_name],text)
                    color='#c24136' if pred else '#128267'
                    label='疑似钓鱼邮件' if pred else '预测为正常邮件'
                    st.markdown(f'<div class="verdict"><span style="color:{color};font-size:28px;font-weight:700">{label}</span><p>模型：{model_name}</p></div>',unsafe_allow_html=True)
                    st.metric('本次响应耗时',f'{ms:.0f} ms')
                    st.caption('包含首次模型加载；不与实验看板的批处理耗时直接比较。')
                    urls=re.findall(r'https?://[^\s<>]+',text)
                    words=[w for w in ['urgent','password','verify','suspended','immediately','payment'] if re.search(r'\b'+w+r'\b',text,re.I)]
                    st.markdown('**文本辅助线索**')
                    st.write(f'链接数量：{len(urls)}')
                    st.write('提示词：'+(', '.join(words) if words else '未发现预设提示词'))
                    st.caption('规则提示仅辅助阅读，不是模型判断原因。链接不会被访问。')
                    if not re.search('[a-zA-Z]',text): st.warning('当前模型针对英文训练，此输入可能超出适用范围。')
                except Exception as exc:
                    st.error(f'模型加载或推理失败：{exc}')
                    if 'DLL' in str(exc) or 'torch' in str(exc):
                        st.info('请关闭旧的服务窗口，双击项目中的「启动界面.bat」后重新打开页面。该脚本使用已修复的项目独立环境。')
        else:
            st.markdown('<div class="verdict"><h3>等待分析</h3><p>输入邮件后，查看预测结果和辅助线索。</p></div>',unsafe_allow_html=True)
            st.caption('不会把分类分数伪装为经过校准的概率。')

elif page=='实验看板':
    st.subheader('模型表现，一目了然')
    if not results: st.info('等待真实实验结果。运行 python train.py 后自动显示。')
    else:
        rows=pd.DataFrame([dict(model=n,**{k:v for k,v in r.items() if k in ['accuracy','precision','recall','f1','ap','fpr','ms_per_email','validation_f1']}) for n,r in results.items()])
        selection=read_json('selected.json').get('name',rows.model.iloc[0])
        st.caption(f'部署模型按验证集 F1 选择：{selection}。以下为固定测试集结果，每个模型使用 {next(iter(results.values()))["test_samples"]:,} 条测试邮件。')
        names=list(results)
        view_name=st.selectbox('查看模型指标',names,index=names.index(selection) if selection in names else 0)
        chosen=results[view_name]
        for col,(title,key) in zip(st.columns(4),[('Accuracy','accuracy'),('Phishing F1','f1'),('Precision','precision'),('Recall','recall')]): col.metric(title,f'{chosen[key]:.2%}')
        st.dataframe(rows.set_index('model').round(4),use_container_width=True)
        a,b=st.columns(2)
        with a:
            fig=px.bar(rows.melt(id_vars='model',value_vars=['precision','recall','f1']),x='model',y='value',color='variable',barmode='group',color_discrete_sequence=['#2364dc','#28a88b','#9a78d0'],title='Precision / Recall / F1')
            fig.update_yaxes(range=[0,1]); st.plotly_chart(fig,use_container_width=True)
        with b:
            name=st.selectbox('混淆矩阵模型',names,index=names.index(view_name))
            fig=px.imshow(results[name]['matrix'],x=['预测正常','预测钓鱼'],y=['实际正常','实际钓鱼'],text_auto=True,color_continuous_scale='Blues',title='Confusion matrix')
            st.plotly_chart(fig,use_container_width=True)
        a,b=st.columns(2)
        with a:
            fig=go.Figure()
            for name,r in results.items(): fig.add_trace(go.Scatter(x=r['pr_recall'],y=r['pr_precision'],name=name,mode='lines'))
            fig.update_layout(title='Precision–Recall',xaxis_title='Recall',yaxis_title='Precision')
            st.plotly_chart(fig,use_container_width=True)
        with b:
            st.plotly_chart(px.bar(rows,x='model',y='ms_per_email',title='推理耗时 / ms per email',color_discrete_sequence=['#2364dc']),use_container_width=True)
            st.caption('传统模型为 CPU 批处理、含 TF-IDF；不同硬件的结果不能直接解释为模型本身的速度差异。')
        info=read_json('data_info.json')
        with st.expander('数据划分与复现信息'): st.json(info)
        ablation=read_json('ablation.json')
        if ablation:
            st.subheader('输入长度消融（验证集）' if len(ablation)>1 else 'DistilBERT 训练配置与验证结果')
            st.dataframe(pd.DataFrame(ablation),use_container_width=True)
            histories=[]
            for row in ablation:
                history=read_json(f'transformer-run/history-{row["max_length"]}.json')
                if history:
                    histories.extend(dict(entry,max_length=str(row['max_length'])) for entry in history)
            if histories:
                history=pd.DataFrame(histories)
                a,b=st.columns(2)
                if 'loss' in history:
                    with a: st.plotly_chart(px.line(history.dropna(subset=['loss']),x='epoch',y='loss',color='max_length',title='训练 Loss'),use_container_width=True)
                if 'eval_f1' in history:
                    with b: st.plotly_chart(px.line(history.dropna(subset=['eval_f1']),x='epoch',y='eval_f1',color='max_length',markers=True,title='验证集 Phishing F1'),use_container_width=True)
            with st.expander('DistilBERT 复现信息'): st.json(read_json('transformer_config.json'))
        st.download_button('下载指标 JSON',json.dumps(results,ensure_ascii=False,indent=2),file_name='metrics.json',mime='application/json')
else:
    st.subheader('从错误中理解模型')
    p=ART/'predictions.csv'
    if not p.exists(): st.info('训练完成后可查看实际误分类样本。')
    else:
        df=pd.read_csv(p)
        keys=[c[:-5] for c in df.columns if c.endswith('_pred')]
        key=st.selectbox('模型',keys)
        kind=st.radio('错误类型',['误报：正常 → 钓鱼','漏报：钓鱼 → 正常'],horizontal=True)
        true=0 if kind.startswith('误报') else 1
        errors=df[(df.label==true)&(df[key+'_pred']!=df.label)]
        st.caption(f'共 {len(errors)} 条；邮件以纯文本显示，不渲染邮件 HTML。')
        for _,row in errors.head(30).iterrows():
            with st.expander(f"样本 {row['id']} · {len(str(row.text)):,} 字符"):
                st.text(str(row.text)[:10000])
