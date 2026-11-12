# PhishGuard 运行说明

## 1. 打开项目文件夹

打开 `phishguard` 文件夹，确认能看到 `app.py` 和 `启动界面.bat`。

本机已配置好环境，可直接进行第 2 步。如果是首次在其他电脑运行，请先安装 Python 3.12，在项目文件夹中打开终端，执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-transformer.txt
```

## 2. 启动界面

双击 `启动界面.bat`。

也可以在项目文件夹的终端中执行以下命令，两种方式选一种即可：

```powershell
.\.venv\Scripts\python.exe -I -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

## 3. 打开页面

等终端显示服务地址后，在浏览器打开：

[http://127.0.0.1:8501](http://127.0.0.1:8501)

使用期间保持终端窗口运行；关闭窗口后，本地页面将无法继续访问。模型已训练好，无需重新训练。
