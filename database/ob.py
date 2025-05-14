import pymysql

# 连接参数
conn = pymysql.connect(
    host='obmt6cbqtuc0dj0g-mi.aliyun-cn-hangzhou-internet.oceanbase.cloud',  # 云数据库地址
    port=3306,                     # 端口
    user='whattt',                 # 用户名
    password='@WAng123456!',      # 密码（需要您填入实际密码）
    database='text2mol',           # 数据库名
    charset='utf8mb4'              # 推荐字符集
)

cursor = conn.cursor()

# 测试查询
cursor.execute("SELECT VERSION();")
print("OB Cloud 版本:", cursor.fetchone())

# 关闭连接
cursor.close()
conn.close()
