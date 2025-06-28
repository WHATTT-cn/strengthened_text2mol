## **关于配置项目所需要的环境：**
CUDA版本：12.5（或其他兼容版本)\
python版本：3.9\
需要安装的库和依赖项:根据根目录下requirements_mine.yml文件安装\
或手动安装:\
`conda install pandas numpy scipy scikit-learn matplotlib requests urllib3 openssl`\
`pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 
 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 
 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 
 pyg-lib==0.2.0 --extra-index-url https://data.pyg.org/whl/torch-2.0.1+cu118.html
 torch-scatter==2.1.1+pt20cu118 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
 torch-sparse==0.6.17+pt20cu118 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
 torch-cluster==1.6.1+pt20cu118 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
 torch-geometric==2.4.0 --extra-index-url https://data.pyg.org/whl/torch-2.0.1+cu118.html
 click
 transformers
 huggingface-hub
 flask
 pymsql=1.1.1`


