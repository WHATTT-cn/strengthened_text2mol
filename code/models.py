from transformers import BertTokenizerFast, BertModel

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


from torch.nn import TransformerDecoder, TransformerDecoderLayer
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MLPModel(nn.Module):
    def __init__(self, ninp, nout, nhid):
        super(MLPModel, self).__init__()

        self.text_hidden1 = nn.Linear(ninp, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout

        self.mol_hidden1 = nn.Linear(nout, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nout)
        

        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter( 'temp' , self.temp )

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        
        self.other_params = list(self.parameters()) #get all but bert params
        
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_models', 'scibert')
        self.text_transformer_model = BertModel.from_pretrained(model_path, local_files_only=True)
        self.text_transformer_model.train()

    def forward(self, text, molecule, text_mask = None):
      
        text_encoder_output = self.text_transformer_model(text, attention_mask = text_mask)

        text_x = text_encoder_output['pooler_output']
        text_x = self.text_hidden1(text_x)

        x = self.relu(self.mol_hidden1(molecule))
        x = self.relu(self.mol_hidden2(x))
        x = self.mol_hidden3(x)


        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        return text_x, x

    def generate_molecule_description(self, molecule, data_generator, all_mol_embeddings, all_cids, top_k=20):
        """使用跨模态检索模型生成分子描述"""
        batch_size = molecule.size(0)
        
        # 通过模型处理分子特征
        with torch.no_grad():
            # 使用分子编码器部分
            x = self.relu(self.mol_hidden1(molecule))
            x = self.relu(self.mol_hidden2(x))
            x = self.mol_hidden3(x)
            x = self.ln1(x)
            x = x * torch.exp(self.temp)
            
            # 将分子特征转换为numpy数组用于相似度计算
            mol_embedding = x.cpu().numpy()
            
            # 计算与数据库中所有分子的相似度
            sims = cosine_similarity(mol_embedding, all_mol_embeddings)
            
            # 获取最相似的top_k个分子
            top_indices = np.argsort(sims[0])[::-1][:top_k]
            
            # 收集最相似分子的描述
            descriptions = []
            for idx in top_indices:
                cid = all_cids[idx]
                similarity = float(sims[0][idx])
                if cid in data_generator.descriptions:
                    desc = data_generator.descriptions[cid]
                    descriptions.append(f"[CID: {cid}, 相似度: {similarity:.3f}] {desc}")
            
            if descriptions:
                # 返回所有最相似分子的描述
                return "\n\n".join(descriptions)
            else:
                return "无法生成分子性质描述"



class GCNModel(nn.Module):
    def __init__(self, num_node_features, ninp, nout, nhid, graph_hidden_channels):
        super(GCNModel, self).__init__()
        

        self.text_hidden1 = nn.Linear(ninp, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout

        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter( 'temp' , self.temp )

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        
        #For GCN:
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nout)


        self.other_params = list(self.parameters()) #get all but bert params
        
        self.text_transformer_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_transformer_model.train()

    def forward(self, text, graph_batch, text_mask = None, molecule_mask = None):
      
        text_encoder_output = self.text_transformer_model(text, attention_mask = text_mask)

        text_x = text_encoder_output['pooler_output']
        text_x = self.text_hidden1(text_x)


        #Obtain node embeddings 
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, graph_hidden_channels]

        
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x).relu()
        x = self.mol_hidden3(x)


        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        return text_x, x





class AttentionModel(nn.Module):

    def __init__(self, num_node_features, ninp, nout, nhid, nhead, nlayers, graph_hidden_channels, mol_trunc_length, temp, dropout=0.5):
        super(AttentionModel, self).__init__()
        
        self.text_hidden1 = nn.Linear(ninp, nhid)
        self.text_hidden2 = nn.Linear(nhid, nout)

        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.num_node_features = num_node_features
        self.graph_hidden_channels = graph_hidden_channels
        self.mol_trunc_length = mol_trunc_length

        self.drop = nn.Dropout(p=dropout)

        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.text_transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        

        self.temp = nn.Parameter(torch.Tensor([temp]))
        self.register_parameter( 'temp' , self.temp )

        self.ln1 = nn.LayerNorm((nout))
        self.ln2 = nn.LayerNorm((nout))

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        
        #For GCN:
        self.conv1 = GCNConv(self.num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)


        self.other_params = list(self.parameters()) #get all but bert params
        
        self.text_transformer_model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
        self.text_transformer_model.train()

        self.device = 'cpu' 

    def set_device(self, dev):
        self.to(dev)
        self.device = dev

    def forward(self, text, graph_batch, text_mask = None, molecule_mask = None):
      
        text_encoder_output = self.text_transformer_model(text, attention_mask = text_mask)

        #Obtain node embeddings 
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        mol_x = self.conv3(x, edge_index)

        #turn pytorch geometric output into the correct format for transformer
        #requires recovering the nodes from each graph into a separate dimension
        node_features = torch.zeros((graph_batch.num_graphs, self.mol_trunc_length, self.graph_hidden_channels)).to(self.device)
        for i, p in enumerate(graph_batch.ptr):
          if p == 0: 
            old_p = p
            continue
          node_features[i - 1, :p-old_p, :] = mol_x[old_p:torch.min(p, old_p + self.mol_trunc_length), :]
          old_p = p
        node_features = torch.transpose(node_features, 0, 1)

        text_output = self.text_transformer_decoder(text_encoder_output['last_hidden_state'].transpose(0,1), node_features, 
                                                            tgt_key_padding_mask = text_mask == 0, memory_key_padding_mask = ~molecule_mask) 


        #Readout layer
        x = global_mean_pool(mol_x, batch)  # [batch_size, graph_hidden_channels]

        x = self.mol_hidden1(x)
        x = x.relu()
        x = self.mol_hidden2(x)

        text_x = torch.tanh(self.text_hidden1(text_output[0,:,:])) #[CLS] pooler
        text_x = self.text_hidden2(text_x)

        x = self.ln1(x)
        text_x = self.ln2(text_x)

        x = x * torch.exp(self.temp)
        text_x = text_x * torch.exp(self.temp)

        return text_x, x
