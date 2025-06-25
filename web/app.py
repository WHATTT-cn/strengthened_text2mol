from flask import Flask, request, jsonify, render_template
import os
import sys
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 导入必要的模块
from text2mol_code.models import MLPModel
from text2mol_code.dataloaders import GenerateData

app = Flask(__name__)

# 全局变量
model = None
data_generator = None
all_mol_embeddings = None
all_cids = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_model():
    global model, data_generator, all_mol_embeddings, all_cids

    # 数据路径
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    emb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_output', 'embeddings')

    # 初始化数据生成器
    data_generator = GenerateData(
        text_trunc_length=256,
        path_train=os.path.join(data_path, "training.txt"),
        path_val=os.path.join(data_path, "val.txt"),
        path_test=os.path.join(data_path, "test.txt"),
        path_molecules=os.path.join(data_path, "ChEBI_defintions_substructure_corpus.cp"),
        path_token_embs=os.path.join(data_path, "token_embedding_dict.npy")
    )

    # 初始化模型
    from text2mol_code.models import MLPModel
    model = MLPModel(ninp=768, nhid=600, nout=300)

    # 修改这里：权重文件路径为test_output目录下
    weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_output',
                                'final_weights.40.pt')
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(weights_path))
    else:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

    model.to(device)
    model.eval()

    # 加载嵌入向量
    cids_train = np.load(os.path.join(emb_dir, "cids_train.npy"), allow_pickle=True)
    cids_val = np.load(os.path.join(emb_dir, "cids_val.npy"), allow_pickle=True)
    cids_test = np.load(os.path.join(emb_dir, "cids_test.npy"), allow_pickle=True)

    text_embeddings_train = np.load(os.path.join(emb_dir, "text_embeddings_train.npy"))
    text_embeddings_val = np.load(os.path.join(emb_dir, "text_embeddings_val.npy"))
    text_embeddings_test = np.load(os.path.join(emb_dir, "text_embeddings_test.npy"))

    chem_embeddings_train = np.load(os.path.join(emb_dir, "chem_embeddings_train.npy"))
    chem_embeddings_val = np.load(os.path.join(emb_dir, "chem_embeddings_val.npy"))
    chem_embeddings_test = np.load(os.path.join(emb_dir, "chem_embeddings_test.npy"))

    all_mol_embeddings = np.concatenate((chem_embeddings_train, chem_embeddings_val, chem_embeddings_test), axis=0)
    all_cids = np.concatenate((cids_train, cids_val, cids_test), axis=0)


def name_to_input(name):
    """将文本描述转换为模型输入格式"""
    text_input = data_generator.text_tokenizer(
        name,
        truncation=True,
        padding='max_length',
        max_length=256,
        return_tensors='pt'
    )

    return {
        'cid': '',
        'input': {
            'text': {
                'input_ids': text_input['input_ids'],
                'attention_mask': text_input['attention_mask'],
            },
            'molecule': {
                'mol2vec': torch.zeros((1, 300)),
                'cid': ''
            }
        },
    }


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': '请提供文本描述'}), 400

        text = data['text']

        # 确保模型已初始化
        if model is None:
            initialize_model()

        # 准备输入
        inputs = name_to_input(text)['input']
        text_mask = inputs['text']['attention_mask'].bool()

        text = inputs['text']['input_ids'].to(device)
        text_mask = text_mask.to(device)
        molecule = inputs['molecule']['mol2vec'].float().to(device)

        # 模型推理
        with torch.no_grad():
            text_out, chem_out = model(text, molecule, text_mask)

        # 计算相似度
        name_emb = text_out.cpu().numpy()
        sims = cosine_similarity(name_emb, all_mol_embeddings)

        # 获取最相似的分子
        cid_locs = np.argsort(sims).squeeze()[::-1]
        ranks = np.argsort(cid_locs)
        sorted = np.argsort(ranks)

        # 准备结果
        results = []
        for i in range(min(20, len(sorted))):
            cid = all_cids[sorted[i]]
            results.append({
                'id': cid,
                'description': data_generator.descriptions[cid],
                'score': float(sims[0][sorted[i]])
            })

        return jsonify({
            'status': 'success',
            'results': results
        })

    except Exception as e:
        print(e)  # 新增：打印异常信息到命令行
        return jsonify({'error': str(e)}), 500


@app.route('/molquery', methods=['POST'])
def molquery():
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': '请提供分子ID或标准命名'}), 400

        input_str = data['input']

        # 确保模型已初始化
        if model is None:
            initialize_model()

        results = []
        # 先尝试按ID查询
        try:
            cid = int(input_str)
            if str(cid) in data_generator.descriptions:
                results.append({
                    'id': cid,
                    'description': data_generator.descriptions[str(cid)]
                })
        except ValueError:
            # 若不是数字，尝试按描述模糊查询
            for cid, desc in data_generator.descriptions.items():
                if input_str.lower() in desc.lower():
                    results.append({
                        'id': cid,
                        'description': desc
                    })

        if not results:
            return jsonify({'error': '未找到匹配的分子'}), 404

        return jsonify({
            'status': 'success',
            'results': results
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


@app.route('/moldesigner', methods=['POST'])
def moldesigner():
    try:
        data = request.get_json()
        if not data or 'properties' not in data:
            return jsonify({'error': '请输入药物分子的性质'}), 400

        properties = data['properties']

        # 这里只是简单示例，实际需要更复杂的算法或模型来生成设计建议
        suggestions = [
            {
                'suggestion': f'根据输入的性质 "{properties}"，建议考虑使用含有特定官能团的分子结构。'
            }
        ]

        return jsonify({
            'status': 'success',
            'results': suggestions
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)