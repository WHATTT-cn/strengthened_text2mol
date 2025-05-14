import pymysql
import numpy as np
import os
import sys
import pickle

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

def import_embeddings():
    try:
        # 建立连接
        conn = pymysql.connect(
            host='obmt6cbqtuc0dj0g-mi.aliyun-cn-hangzhou-internet.oceanbase.cloud',
            port=3306,
            user='whattt',
            password='@WAng123456!',
            database='text2mol',
            charset='utf8mb4'
        )
        
        cursor = conn.cursor()
        
        # 加载数据
        emb_dir = os.path.join(project_root, 'test_output', 'embeddings')
        data_path = os.path.join(project_root, 'data')
        
        # 检查文件是否存在
        descriptions_file = os.path.join(data_path, 'ChEBI_defintions_substructure_corpus.cp')
        if not os.path.exists(descriptions_file):
            print(f"错误：找不到描述文件 {descriptions_file}")
            return
            
        print(f"正在加载描述文件: {descriptions_file}")
        
        # 尝试加载描述数据
        try:
            with open(descriptions_file, 'rb') as f:
                descriptions = pickle.load(f)
            print("描述文件加载成功")
        except Exception as e:
            print(f"加载描述文件时出错: {e}")
            print("将使用空描述继续")
            descriptions = {}
        
        # 准备SQL语句
        insert_sql = """
        INSERT INTO molecule_embeddings (id, embedding, description, embedding_type)
        VALUES (%s, %s, %s, %s)
        """
        
        # 检查并加载训练集数据
        train_cids_file = os.path.join(emb_dir, "cids_train.npy")
        train_emb_file = os.path.join(emb_dir, "chem_embeddings_train.npy")
        
        if os.path.exists(train_cids_file) and os.path.exists(train_emb_file):
            print("开始导入训练集数据...")
            cids_train = np.load(train_cids_file, allow_pickle=True)
            chem_embeddings_train = np.load(train_emb_file)
            
            for i, cid in enumerate(cids_train):
                try:
                    embedding = chem_embeddings_train[i].tobytes()
                    description = descriptions.get(cid, '')
                    cursor.execute(insert_sql, (cid, embedding, description, 'train'))
                    if (i + 1) % 100 == 0:
                        print(f"已导入 {i + 1} 条训练集数据")
                        conn.commit()  # 每100条提交一次
                except Exception as e:
                    print(f"导入训练集数据时出错 (索引 {i}): {e}")
                    continue
        else:
            print("警告：找不到训练集数据文件")
        
        # 检查并加载验证集数据
        val_cids_file = os.path.join(emb_dir, "cids_val.npy")
        val_emb_file = os.path.join(emb_dir, "chem_embeddings_val.npy")
        
        if os.path.exists(val_cids_file) and os.path.exists(val_emb_file):
            print("开始导入验证集数据...")
            cids_val = np.load(val_cids_file, allow_pickle=True)
            chem_embeddings_val = np.load(val_emb_file)
            
            for i, cid in enumerate(cids_val):
                try:
                    embedding = chem_embeddings_val[i].tobytes()
                    description = descriptions.get(cid, '')
                    cursor.execute(insert_sql, (cid, embedding, description, 'val'))
                    if (i + 1) % 100 == 0:
                        print(f"已导入 {i + 1} 条验证集数据")
                        conn.commit()  # 每100条提交一次
                except Exception as e:
                    print(f"导入验证集数据时出错 (索引 {i}): {e}")
                    continue
        else:
            print("警告：找不到验证集数据文件")
        
        # 检查并加载测试集数据
        test_cids_file = os.path.join(emb_dir, "cids_test.npy")
        test_emb_file = os.path.join(emb_dir, "chem_embeddings_test.npy")
        
        if os.path.exists(test_cids_file) and os.path.exists(test_emb_file):
            print("开始导入测试集数据...")
            cids_test = np.load(test_cids_file, allow_pickle=True)
            chem_embeddings_test = np.load(test_emb_file)
            
            for i, cid in enumerate(cids_test):
                try:
                    embedding = chem_embeddings_test[i].tobytes()
                    description = descriptions.get(cid, '')
                    cursor.execute(insert_sql, (cid, embedding, description, 'test'))
                    if (i + 1) % 100 == 0:
                        print(f"已导入 {i + 1} 条测试集数据")
                        conn.commit()  # 每100条提交一次
                except Exception as e:
                    print(f"导入测试集数据时出错 (索引 {i}): {e}")
                    continue
        else:
            print("警告：找不到测试集数据文件")
        
        conn.commit()
        print("所有数据导入完成！")
        
    except Exception as e:
        print(f"导入数据时出错: {e}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    import_embeddings() 