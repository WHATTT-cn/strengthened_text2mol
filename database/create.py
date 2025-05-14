import pymysql

def create_tables():
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
        
        # 创建游标
        cursor = conn.cursor()
        
        # 创建表
        sql = """
        CREATE TABLE IF NOT EXISTS molecule_embeddings (
            id VARCHAR(50) PRIMARY KEY,
            embedding BLOB,
            description TEXT,
            embedding_type ENUM('train', 'val', 'test')
        )
        """
        
        # 执行SQL
        cursor.execute(sql)
        conn.commit()
        print("表创建成功！")
        
        # 显示所有表
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()
        print("数据库中的表：", tables)
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    create_tables()
