import os
import numpy as np
from rdkit import Chem
from mol2vec.features import mol2alt_sentence
from gensim.models import word2vec
import zipfile
from tqdm import tqdm

class MoleculeProcessor:
    def __init__(self, output_dir):
        """
        初始化分子处理器
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_molecule(self, mol):
        """
        使用mol2vec处理单个分子
        阶段：mol2vec处理原始分子
        Args:
            mol: RDKit分子对象
        Returns:
            sentence: 分子子结构序列
        """
        if mol is None:
            return []
        # 使用mol2vec的Morgan指纹方法生成子结构序列
        sentence = mol2alt_sentence(mol, 1)
        return sentence

    def create_substructure_corpus(self, molecules):
        """
        生成子结构序列文件
        阶段：生成ChEBI_defintions_substructure_corpus.cp
        Args:
            molecules: 分子列表
        """
        print("生成子结构序列文件...")
        with open(os.path.join(self.output_dir, "ChEBI_defintions_substructure_corpus.cp"), 'w') as f:
            for i, mol in enumerate(tqdm(molecules)):
                sentence = self.process_molecule(mol)
                if sentence:  # 确保分子处理成功
                    f.write(f"{i}: {' '.join(sentence)}\n")

    def create_mol_graph(self, mol, cid):
        """
        为单个分子生成图结构
        阶段：生成mol_graphs.zip中的单个.graph文件
        Args:
            mol: RDKit分子对象
            cid: 分子ID
        Returns:
            content: .graph文件内容
        """
        if mol is None:
            return None

        # 1. 获取分子图结构
        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        
        # 2. 生成边列表
        edges = []
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                if adj_matrix[i][j] > 0:
                    edges.append(f"{i} {j}")
                    edges.append(f"{j} {i}")  # 无向图需要双向边
        
        # 3. 获取子结构标识符
        identifiers = {}
        for i in range(num_atoms):
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, 1, i)
            if env is not None:
                amap = {}
                submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                if submol is not None:
                    sentence = mol2alt_sentence(submol, 1)
                    if sentence:  # 确保子结构处理成功
                        identifiers[i] = sentence[0]
        
        # 4. 生成.graph文件内容
        content = "edgelist:\n"
        content += "\n".join(edges)
        content += "\n\nidx to identifier:\n"
        for idx, identifier in identifiers.items():
            content += f"{idx} {identifier}\n"
        
        return content

    def create_mol_graphs(self, molecules):
        """
        生成所有分子的图结构文件
        阶段：生成mol_graphs.zip
        Args:
            molecules: 分子列表
        """
        print("生成分子图结构文件...")
        # 创建临时目录存储.graph文件
        temp_dir = os.path.join(self.output_dir, "temp_graphs")
        os.makedirs(temp_dir, exist_ok=True)
        
        # 生成每个分子的.graph文件
        for i, mol in enumerate(tqdm(molecules)):
            content = self.create_mol_graph(mol, i)
            if content:
                with open(os.path.join(temp_dir, f"{i}.graph"), 'w') as f:
                    f.write(content)
        
        # 将.graph文件打包成zip
        with zipfile.ZipFile(os.path.join(self.output_dir, "mol_graphs.zip"), 'w') as zipf:
            for file in os.listdir(temp_dir):
                zipf.write(os.path.join(temp_dir, file), file)
        
        # 清理临时目录
        import shutil
        shutil.rmtree(temp_dir)

    def train_token_embeddings(self, molecules):
        """
        训练token向量映射
        阶段：生成token_embedding_dict.npy
        Args:
            molecules: 分子列表
        """
        print("训练token向量映射...")
        # 生成语料库
        corpus = []
        for mol in tqdm(molecules):
            sentence = self.process_molecule(mol)
            if sentence:
                corpus.append(sentence)
        
        # 训练Word2Vec模型
        model = word2vec.Word2Vec(corpus, 
                                vector_size=100,
                                window=5,
                                min_count=1,
                                workers=4)
        
        # 保存token向量映射
        token_embeddings = {}
        for token in model.wv.index_to_key:
            token_embeddings[token] = model.wv[token]
        np.save(os.path.join(self.output_dir, "token_embedding_dict.npy"), token_embeddings)

    def create_training_data(self, molecules, descriptions):
        """
        生成训练数据
        阶段：生成training.txt、val.txt、test.txt
        Args:
            molecules: 分子列表
            descriptions: 分子描述字典
        """
        print("生成训练数据...")
        # 加载token向量映射
        token_embeddings = np.load(os.path.join(self.output_dir, "token_embedding_dict.npy"), 
                                 allow_pickle=True)[()]
        
        # 生成分子向量
        mol_vectors = {}
        for i, mol in enumerate(tqdm(molecules)):
            sentence = self.process_molecule(mol)
            if sentence:
                vector = np.mean([token_embeddings[token] for token in sentence], axis=0)
                mol_vectors[i] = vector
        
        # 分割数据集
        train_data = []
        val_data = []
        test_data = []
        
        for i, (mol_id, vector) in enumerate(mol_vectors.items()):
            if mol_id in descriptions:
                line = f"{mol_id}\t{vector}\t{descriptions[mol_id]}\n"
                if i < len(mol_vectors) * 0.8:
                    train_data.append(line)
                elif i < len(mol_vectors) * 0.9:
                    val_data.append(line)
                else:
                    test_data.append(line)
        
        # 保存数据集
        with open(os.path.join(self.output_dir, "training.txt"), 'w') as f:
            f.writelines(train_data)
        with open(os.path.join(self.output_dir, "val.txt"), 'w') as f:
            f.writelines(val_data)
        with open(os.path.join(self.output_dir, "test.txt"), 'w') as f:
            f.writelines(test_data)

def main():
    """
    主函数：执行完整的数据处理流程
    """
    # 设置输出目录
    output_dir = "processed_data"
    
    # 创建处理器实例
    processor = MoleculeProcessor(output_dir)
    
    # 示例：加载分子数据
    molecules = [
        Chem.MolFromSmiles("CC(=O)O"),  # 乙酸
        Chem.MolFromSmiles("c1ccccc1"),  # 苯
        # 添加更多分子...
    ]
    
    # 示例：分子描述
    descriptions = {
        0: "Acetic acid",
        1: "Benzene",
        # 添加更多描述...
    }
    
    # 执行完整的数据处理流程
    processor.create_substructure_corpus(molecules)
    processor.create_mol_graphs(molecules)
    processor.train_token_embeddings(molecules)
    processor.create_training_data(molecules, descriptions)

if __name__ == "__main__":
    main()
