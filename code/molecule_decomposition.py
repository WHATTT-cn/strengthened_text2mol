#mol2vec处理原始分子
#生成子结构序列
#生成分子图结构
#训练token向量映射
#生成训练数据

import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Recap, BRICS
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec
from gensim.models import word2vec
from rdkit.Chem import AllChem

class MoleculeDecomposer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.model = None
        
    def load_molecules(self, file_path):
        """加载分子数据"""
        if file_path.endswith('.sdf'):
            return Chem.SDMolSupplier(file_path)
        elif file_path.endswith('.smi'):
            with open(file_path, 'r') as f:
                return [Chem.MolFromSmiles(line.strip()) for line in f]
        else:
            raise ValueError("不支持的文件格式，请使用.sdf或.smi文件")

    def decompose_with_recap(self, mol):
        """使用Recap算法分解分子"""
        if mol is None:
            return []
        
        # 使用Recap分解分子
        recap_tree = Recap.RecapDecompose(mol)
        leaves = recap_tree.GetLeaves()
        
        # 获取所有子结构
        fragments = []
        for leaf in leaves:
            fragment = leaf.mol
            if fragment is not None:
                fragments.append(fragment)
        
        return fragments

    def decompose_with_brics(self, mol):
        """使用BRICS算法分解分子"""
        if mol is None:
            return []
        
        # 使用BRICS分解分子
        fragments = BRICS.BRICSDecompose(mol)
        return [Chem.MolFromSmiles(frag) for frag in fragments]

    def train_mol2vec(self, molecules, output_path='mol2vec_model.pkl'):
        """训练mol2vec模型"""
        # 生成分子句子
        sentences = []
        for mol in molecules:
            if mol is not None:
                sentence = mol2alt_sentence(mol, 1)
                sentences.append(sentence)
        
        # 训练word2vec模型
        model = word2vec.Word2Vec(sentences, vector_size=100, window=5, min_count=1)
        self.model = model
        model.save(output_path)
        return model

    def get_mol2vec_vectors(self, molecules):
        """获取分子的mol2vec向量表示"""
        if self.model is None:
            raise ValueError("请先训练mol2vec模型")
        
        vectors = []
        for mol in molecules:
            if mol is not None:
                sentence = mol2alt_sentence(mol, 1)
                # 计算分子句子的平均向量
                mol_vec = np.mean([self.model.wv[word] for word in sentence], axis=0)
                vectors.append(mol_vec)
        
        return np.array(vectors)

    def process_molecules(self, input_file, output_dir):
        """处理分子数据并保存结果"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载分子
        molecules = self.load_molecules(os.path.join(self.data_dir, input_file))
        
        # 存储结果
        results = {
            'recap_fragments': [],
            'brics_fragments': [],
            'recap_vectors': [],
            'brics_vectors': []
        }
        
        # 处理每个分子
        for mol in molecules:
            if mol is None:
                continue
                
            # Recap分解
            recap_frags = self.decompose_with_recap(mol)
            results['recap_fragments'].extend(recap_frags)
            
            # BRICS分解
            brics_frags = self.decompose_with_brics(mol)
            results['brics_fragments'].extend(brics_frags)
        
        # 训练mol2vec模型
        all_fragments = results['recap_fragments'] + results['brics_fragments']
        self.train_mol2vec(all_fragments, 
                          output_path=os.path.join(output_dir, 'mol2vec_model.pkl'))
        
        # 获取向量表示
        results['recap_vectors'] = self.get_mol2vec_vectors(results['recap_fragments'])
        results['brics_vectors'] = self.get_mol2vec_vectors(results['brics_fragments'])
        
        # 保存结果
        np.save(os.path.join(output_dir, 'recap_vectors.npy'), results['recap_vectors'])
        np.save(os.path.join(output_dir, 'brics_vectors.npy'), results['brics_vectors'])
        
        # 保存片段信息
        with open(os.path.join(output_dir, 'fragments_info.txt'), 'w') as f:
            f.write("Recap Fragments:\n")
            for frag in results['recap_fragments']:
                f.write(f"{Chem.MolToSmiles(frag)}\n")
            f.write("\nBRICS Fragments:\n")
            for frag in results['brics_fragments']:
                f.write(f"{Chem.MolToSmiles(frag)}\n")
        
        return results

def main():
    # 设置路径
    data_dir = "data"
    output_dir = "text2mol_code/output"
    
    # 创建分解器实例
    decomposer = MoleculeDecomposer(data_dir)
    
    # 处理分子数据
    # 注意：需要根据实际的文件名修改
    results = decomposer.process_molecules("your_molecule_file.sdf", output_dir)
    
    print("处理完成！结果已保存到", output_dir)

if __name__ == "__main__":
    main()