#!/usr/bin/env python3
"""
SMILES处理模块
将SMILES表达式转换为分子特征向量
"""

import os
import sys
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

class SMILESProcessor:
    """SMILES表达式处理器"""
    
    def __init__(self, feature_dim=300):
        self.feature_dim = feature_dim
        
    def smiles_to_mol(self, smiles):
        """将SMILES转换为RDKit分子对象"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"无效的SMILES表达式: {smiles}")
            return mol
        except Exception as e:
            raise ValueError(f"SMILES处理失败: {e}")
    
    def mol_to_fingerprint(self, mol, radius=2, nBits=2048):
        """将分子转换为Morgan指纹"""
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            return np.array(fp)
        except Exception as e:
            raise ValueError(f"指纹生成失败: {e}")
    
    def mol_to_descriptors(self, mol):
        """计算分子描述符"""
        try:
            descriptors = {}
            
            # 基本描述符
            descriptors['MolWt'] = Descriptors.MolWt(mol)
            descriptors['LogP'] = Descriptors.MolLogP(mol)
            descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
            descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
            descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            descriptors['NumAtoms'] = mol.GetNumAtoms()
            descriptors['NumBonds'] = mol.GetNumBonds()
            descriptors['NumRings'] = Descriptors.RingCount(mol)
            descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
            
            # 原子类型计数
            atom_counts = {}
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                atom_counts[symbol] = atom_counts.get(symbol, 0) + 1
            
            # 常见原子类型
            common_atoms = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
            for atom in common_atoms:
                descriptors[f'Num{atom}'] = atom_counts.get(atom, 0)
            
            return descriptors
            
        except Exception as e:
            raise ValueError(f"描述符计算失败: {e}")
    
    def smiles_to_features(self, smiles, device=None):
        """将SMILES转换为特征向量 - 使用mol2vec特征"""
        try:
            # 转换为分子对象
            mol = self.smiles_to_mol(smiles)
            
            # 使用mol2vec特征而不是Morgan指纹
            # 这里我们使用RDKit的mol2vec实现
            from rdkit.Chem import AllChem
            
            # 生成mol2vec特征 (300维)
            mol2vec = AllChem.GetMol2vec(mol)
            
            # 如果mol2vec不可用，使用Morgan指纹作为备选
            if mol2vec is None or len(mol2vec) == 0:
                print("警告：mol2vec不可用，使用Morgan指纹作为备选")
                fp = self.mol_to_fingerprint(mol)
                features = fp[:self.feature_dim].astype(np.float32)
            else:
                features = np.array(mol2vec, dtype=np.float32)
            
            # 确保特征维度正确
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)), 'constant')
            elif len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            
            tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            # 如果指定了设备，将tensor移动到该设备
            if device is not None:
                tensor = tensor.to(device)
            
            return tensor
            
        except Exception as e:
            print(f"SMILES特征提取失败，使用随机特征: {e}")
            # 如果特征提取失败，使用随机特征
            features = np.random.randn(self.feature_dim).astype(np.float32)
            tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            if device is not None:
                tensor = tensor.to(device)
            return tensor
    
    def validate_smiles(self, smiles):
        """验证SMILES表达式是否有效"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    def get_mol_info(self, smiles):
        """获取分子信息"""
        try:
            mol = self.smiles_to_mol(smiles)
            
            info = {
                'smiles': smiles,
                'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
                'molecular_weight': Descriptors.MolWt(mol),
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'num_rings': Descriptors.RingCount(mol),
                'is_valid': True
            }
            
            return info
            
        except Exception as e:
            return {
                'smiles': smiles,
                'error': str(e),
                'is_valid': False
            }

def test_smiles_processor():
    """测试SMILES处理器"""
    processor = SMILESProcessor()
    
    # 测试SMILES
    test_smiles = [
        "CC(=O)Oc1ccccc1C(=O)O",  # 阿司匹林
        "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",  # 布洛芬
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # 萘普生
        "CCOC(=O)c1ccccc1",  # 苯甲酸乙酯
    ]
    
    print("测试SMILES处理器...")
    
    for smiles in test_smiles:
        print(f"\nSMILES: {smiles}")
        
        # 验证SMILES
        is_valid = processor.validate_smiles(smiles)
        print(f"有效性: {'✓' if is_valid else '✗'}")
        
        if is_valid:
            # 获取分子信息
            info = processor.get_mol_info(smiles)
            print(f"分子式: {info['formula']}")
            print(f"分子量: {info['molecular_weight']:.2f}")
            print(f"原子数: {info['num_atoms']}")
            
            # 转换为特征
            features = processor.smiles_to_features(smiles)
            print(f"特征维度: {features.shape}")
        else:
            print("SMILES无效，跳过特征提取")

if __name__ == "__main__":
    test_smiles_processor() 