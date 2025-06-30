from flask import Flask, request, jsonify, render_template
import os
import sys
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
import requests

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 导入必要的模块
from code.models_nointernet import MLPModel
from code.dataloaders_nointernet import GenerateData
from code.smiles_processor import SMILESProcessor

# 导入配置文件
try:
    from config import OLLAMA_API_URL, OLLAMA_MODEL_NAME, MAX_HISTORY_LENGTH, MAX_TOKENS, TEMPERATURE, SYSTEM_PROMPT
except ImportError:
    # 如果配置文件不存在，使用默认值
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL_NAME = "deepseek-r1:1.5b"
    MAX_HISTORY_LENGTH = 10
    MAX_TOKENS = 2000
    TEMPERATURE = 0.7
    SYSTEM_PROMPT = "你是一个专业的药物设计AI助手..."

app = Flask(__name__)

# 全局变量
model = None
data_generator = None
all_mol_embeddings = None
all_cids = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 添加SMILES处理器
smiles_processor = None

# 初始化标志
models_initialized = False


def initialize_model():
    """初始化所有模型和处理器"""
    global model, data_generator, all_mol_embeddings, all_cids, smiles_processor, models_initialized
    
    try:
        print("正在初始化模型...")
        
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
        print("✓ 数据生成器初始化成功")

        # 初始化检索模型
        model = MLPModel(ninp=768, nhid=600, nout=300)

        # 权重文件路径
        weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_output',
                                    'final_weights.40.pt')
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path))
        else:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

        model.to(device)
        model.eval()
        print("✓ 检索模型初始化成功")

        # 初始化SMILES处理器
        smiles_processor = SMILESProcessor(feature_dim=300)
        print("✓ SMILES处理器初始化成功")

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
        print("✓ 嵌入向量加载成功")
        
        models_initialized = True
        print("✓ 所有模型初始化完成")
        
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        models_initialized = False


def ensure_models_initialized():
    """确保模型已初始化"""
    global models_initialized
    if not models_initialized:
        initialize_model()
    return models_initialized


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
def index():
    return render_template('naturallanguage2mol.html', active_page='naturallanguage2mol')

@app.route('/molquery_page')
def molquery_page():
    return render_template('molquery.html', active_page='molquery')

@app.route('/moldesigner')
def moldesigner():
    return render_template('moldesigner.html', active_page='moldesigner')

@app.route('/moldesigner_page')
def moldesigner_page():
    return render_template('moldesigner.html', active_page='moldesigner')

@app.route('/naturallanguage2mol')
def naturallanguage2mol():
    return render_template('naturallanguage2mol.html', active_page='naturallanguage2mol')

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': '请提供文本描述'}), 400

        text = data['text']

        # 确保模型已初始化
        if not ensure_models_initialized():
            return jsonify({'error': '模型初始化失败'}), 500

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
        print(f"搜索错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'搜索失败: {str(e)}'}), 500


@app.route('/molquery', methods=['POST'])
def molquery():
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': '请提供分子ID或标准命名'}), 400

        input_str = data['input']

        # 确保模型已初始化
        if not ensure_models_initialized():
            return jsonify({'error': '模型初始化失败'}), 500

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
        print(f"查询错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'查询失败: {str(e)}'}), 500


@app.route('/molecule_query', methods=['POST'])
def molecule_query():
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': '请提供输入'}), 400

        user_input = data['input'].strip()

        # 确保模型已初始化
        if not ensure_models_initialized():
            return jsonify({'error': '模型初始化失败'}), 500

        # 判断输入类型
        if user_input.isdigit():
            # CID输入
            cid = user_input
            if cid in data_generator.descriptions:
                description = data_generator.descriptions[cid]
                similar_molecules = find_similar_molecules_by_smiles("", data_generator, all_mol_embeddings, all_cids, top_k=20)
                return jsonify({
                    'status': 'success',
                    'type': 'cid',
                    'cid': cid,
                    'description': description,
                    'similar_molecules': similar_molecules
                })
            else:
                return jsonify({'error': f'未找到CID {cid} 的描述'}), 404
        else:
            # SMILES输入
            if not smiles_processor:
                return jsonify({'error': 'SMILES处理器未初始化'}), 500

            # 验证SMILES
            validation_result = smiles_processor.validate_smiles(user_input)
            if not validation_result['is_valid']:
                return jsonify({
                    'status': 'error',
                    'type': 'smiles',
                    'error': validation_result['error_message']
                }), 400

            # 获取分子信息
            mol_info = smiles_processor.extract_molecular_info(user_input)
            
            # 查找相似分子
            similar_molecules = find_similar_molecules_by_smiles(user_input, data_generator, all_mol_embeddings, all_cids, top_k=20)

            return jsonify({
                'status': 'success',
                'type': 'smiles',
                'smiles': user_input,
                'molecular_info': mol_info,
                'similar_molecules': similar_molecules
            })

    except Exception as e:
        print(f"分子查询错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'查询失败: {str(e)}'}), 500


@app.route('/smiles_validation', methods=['POST'])
def smiles_validation():
    try:
        data = request.get_json()
        if not data or 'smiles' not in data:
            return jsonify({'error': '请提供SMILES字符串'}), 400

        smiles = data['smiles'].strip()

        if not smiles_processor:
            return jsonify({'error': 'SMILES处理器未初始化'}), 500

        validation_result = smiles_processor.validate_smiles(smiles)
        return jsonify(validation_result)

    except Exception as e:
        print(f"SMILES验证错误: {e}")
        return jsonify({'error': f'验证失败: {str(e)}'}), 500


def find_similar_molecules_by_smiles(smiles, data_generator, all_mol_embeddings, all_cids, top_k=20):
    """根据SMILES查找相似分子"""
    try:
        if not smiles or not smiles_processor:
            return "SMILES输入无效或处理器未初始化"

        # 获取输入分子的特征
        input_features = smiles_processor.smiles_to_features(smiles)
        if input_features is None:
            return "无法提取分子特征"

        # 获取分子信息
        mol_info = smiles_processor.extract_molecular_info(smiles)
        input_mw = mol_info.get('molecular_weight', 0)
        input_formula = mol_info.get('molecular_formula', '')

        # 查找相似分子
        similar_molecules = []
        
        for cid, desc in data_generator.descriptions.items():
            if cid in data_generator.mols:
                try:
                    # 获取数据库中的mol2vec特征
                    mol2vec_str = data_generator.mols[cid]
                    mol2vec = np.fromstring(mol2vec_str, sep=" ")
                    
                    # 计算综合相似度分数
                    similarity_score = 0.0
                    
                    # 1. 分子量相似度 (权重: 0.3)
                    mw_similarity = 0.5  # 默认值
                    if "molecular weight" in desc.lower() or "mw" in desc.lower():
                        import re
                        mw_match = re.search(r'(\d+\.?\d*)', desc)
                        if mw_match:
                            try:
                                db_mw = float(mw_match.group(1))
                                mw_diff = abs(input_mw - db_mw)
                                mw_similarity = max(0, 1 - mw_diff / 100)
                            except:
                                pass
                    
                    # 2. 分子式相似度 (权重: 0.2)
                    formula_similarity = 0.5
                    if input_formula in desc or any(element in desc for element in ['C', 'H', 'O', 'N']):
                        formula_similarity = 0.8
                    
                    # 3. 化学性质相似度 (权重: 0.3)
                    organic_keywords = ['organic', 'compound', 'molecule', 'acid', 'ester', 'alcohol', 'ketone', 'aldehyde', 'amine', 'amide', 'ether', 'phenyl', 'benzene', 'aromatic', 'alkyl', 'alkenyl', 'alkynyl']
                    inorganic_keywords = ['ion', 'cation', 'anion', 'salt', 'metal', 'inorganic', 'element', 'atom', 'isotope']
                    
                    organic_score = sum(1 for keyword in organic_keywords if keyword in desc.lower())
                    inorganic_score = sum(1 for keyword in inorganic_keywords if keyword in desc.lower())
                    
                    if organic_score > inorganic_score:
                        chem_similarity = 0.9
                    elif inorganic_score > organic_score:
                        chem_similarity = 0.1
                    else:
                        chem_similarity = 0.5
                    
                    # 4. 结构相似度 (权重: 0.2)
                    structure_keywords = ['phenyl', 'benzene', 'aromatic', 'ester', 'acetate', 'salicylate']
                    structure_score = sum(1 for keyword in structure_keywords if keyword in desc.lower())
                    structure_similarity = min(1.0, structure_score / 3)
                    
                    # 计算综合相似度
                    similarity_score = (mw_similarity * 0.3 + 
                                      formula_similarity * 0.2 + 
                                      chem_similarity * 0.3 + 
                                      structure_similarity * 0.2)
                    
                    similar_molecules.append({
                        'cid': cid,
                        'description': desc,
                        'similarity': similarity_score
                    })
                    
                except Exception as e:
                    print(f"处理CID {cid}时出错: {e}")
                    continue
        
        # 按相似度排序
        similar_molecules.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 返回前top_k个
        result = []
        for i, mol in enumerate(similar_molecules[:top_k]):
            result.append(f"[CID: {mol['cid']}, 总相似度: {mol['similarity']:.3f}] {mol['description']}")
        
        return "\n\n".join(result) if result else "未找到相似分子"
        
    except Exception as e:
        print(f"相似度计算错误: {e}")
        import traceback
        traceback.print_exc()
        return f"相似度计算失败: {str(e)}"


@app.route('/chat_with_ai', methods=['POST'])
def chat_with_ai():
    """与Ollama AI助手聊天"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        history = data.get('history', [])
        
        if not message:
            return jsonify({'error': '请输入消息'}), 400
        
        # 构建完整的对话内容
        full_conversation = SYSTEM_PROMPT + "\n\n"
        
        # 添加历史对话（限制长度）
        for msg in history[-MAX_HISTORY_LENGTH:]:
            if msg['role'] == 'user':
                full_conversation += f"用户: {msg['content']}\n"
            else:
                full_conversation += f"助手: {msg['content']}\n"
        
        # 添加当前用户消息
        full_conversation += f"用户: {message}\n助手: "
        
        # 调用Ollama API
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": OLLAMA_MODEL_NAME,
            "prompt": full_conversation,
            "stream": False,
            "options": {
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS,
                "top_p": 0.8
            }
        }
        
        response = requests.post(
            OLLAMA_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            # Ollama的响应格式
            ai_response = result.get('response', '抱歉，我无法理解您的问题。')
            return jsonify({'response': ai_response})
        else:
            error_msg = f"Ollama API调用失败: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f" - {error_detail.get('error', '')}"
            except:
                error_msg += f" - {response.text}"
            return jsonify({'error': error_msg}), 500
            
    except requests.exceptions.Timeout:
        return jsonify({'error': '请求超时，请稍后重试'}), 408
    except requests.exceptions.ConnectionError:
        return jsonify({'error': '无法连接到Ollama服务，请确保Ollama已启动'}), 503
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'网络请求失败: {str(e)}'}), 500
    except Exception as e:
        print(f"聊天错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500


if __name__ == '__main__':
    print("正在启动Web应用...")
    print("正在初始化模型...")
    initialize_model()
    print("Web应用启动完成，开始运行...")
    app.run(debug=False, host='0.0.0.0', port=5000)