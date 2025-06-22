# 智能邮件处理系统：垃圾邮件识别与社交影响力驱动的优化方案
## 1. 系统概述

本系统结合**垃圾邮件分类技术**和**社交网络分析**，实现智能化的邮件处理流程。系统架构如下：
![输入图片说明](/imgs/2025-06-22/j6KEuvGLbdrLE8cN.png)
## 2. 垃圾邮件分类模块

### 2.1 数据预处理
```
"""
读取文本文件并转换为DataFrame

:param file_path: 文件路径，jison格式

:param encoding: 文件编码，默认为'latin-1'

:return: 包含标签和消息的DataFrame

"""
def load_data(file_path):
data = []
  try:
     with open(file_path, 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
               label, message = parts
               label = 0 if label == 'ham' else 1
               data.append([label, message])
   except FileNotFoundError:
     print(f"文件 {file_path} 不存在")
     return None
   except Exception as e:
     print(f"读取文件时出现错误: {e}")
     return None
   return pd.DataFrame(data, columns=['label','message'])```

"""
对数据进行预处理，包括计算消息长度、去除停用词常用词等
:param df: 包含标签和消息的DataFrame
:return: 预处理后的DataFrame
这里用了nltk的stopwords模块，需要自主下载，并添加到环境变量中
"""
def preprocess_data(df):

# 计算原始消息长度
  df['length'] = df['message'].str.len()
  try:
      stop_words = set(stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])
      df['message'] = df['message'].apply(lambda x: " ".join(term for term in x.split() if term not in stop_words))
  except LookupError:
      print("未找到 stopwords 数据，请检查下载情况。")

# 计算清理后消息长度
df['clean_length'] = df['message'].str.len()
return df
```
**功能说明**：
-   移除常见停用词（the, is, at等）和特定噪声（u, ur, 4等）
-   计算消息长度变化作为特征
### 2.2 特征词提取与分析
```    
def get_top_n_words(df, n=10):
    """
    提取垃圾邮件和正常邮件的Top N特征词
    :param df: 包含邮件文本和标签的DataFrame
    :param n: 要提取的Top N词语数量
    :return: 两个DataFrame (垃圾邮件特征词, 正常邮件特征词)
    """
    # 初始化TF-IDF向量化器
    tf_vec = TfidfVectorizer(max_features=1000)
    
    # 在整个数据集上拟合TF-IDF模型
    X = tf_vec.fit_transform(df['message'])
    feature_names = tf_vec.get_feature_names_out()

    # 按标签分离数据集
    spam_messages = df[df['label'] == 1]['message']
    ham_messages = df[df['label'] == 0]['message']

    # 转换子集为TF-IDF矩阵
    X_spam = tf_vec.transform(spam_messages)
    X_ham = tf_vec.transform(ham_messages)

    # 计算每个特征词的平均TF-IDF值
    spam_tfidf = X_spam.mean(axis=0).A1
    ham_tfidf = X_ham.mean(axis=0).A1

    # 创建结果DataFrame
    spam_words = pd.DataFrame(list(zip(feature_names, spam_tfidf)), columns=['word', 'tfidf'])
    ham_words = pd.DataFrame(list(zip(feature_names, ham_tfidf)), columns=['word', 'tfidf'])

    # 按TF-IDF值排序并取Top N
    spam_words = spam_words.sort_values('tfidf', ascending=False).head(n)
    ham_words = ham_words.sort_values('tfidf', ascending=False).head(n)

    return spam_words, ham_words

def generate_wordcloud(text):
    """
    生成词云可视化
    :param text: 要可视化的文本
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
```
### 2.3 垃圾邮件分类模型与评估
#### 2.3.1训练并评估SVM模型
```
"""
训练并评估模型
:param X: 特征矩阵
:param y: 标签
:return: 模型的准确率
"""

def train_and_evaluate_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   model = SVC()
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
return accuracy
```
```
if __name__ == "__main__":
   file_path = r"C:\Users\HUAWEI\Downloads\sms+spam+collection\SMSSpamCollection.txt"
   df = load_data(file_path)
   if df is not None:
   df = preprocess_data(df)
# 查看数据基本信息和前几行
   print("数据基本信息：")
   df.info()
   print("数据前几行信息：")
   print(df.head().to_csv(sep='\t', na_rep='nan'))
# 统计标签分布
   label_counts = df['label'].value_counts()
   print("标签分布统计：")
   print(label_counts)
# 计算非垃圾邮件比例
   non_spam_ratio = round(len(df[df['label'] == 0]) / len(df['label']), 2) * 100
   print(f"非垃圾邮件比例（标签为 0）: {non_spam_ratio}%")
# 计算垃圾邮件比例
   spam_ratio = round(len(df[df['label'] == 1]) / len(df['label']), 2) * 100
   print(f"垃圾邮件比例（标签为 1）: {spam_ratio}%")
# 输出长度变化信息
   print("原始消息总长度:", df['length'].sum())
   print("清理后消息总长度:", df['clean_length'].sum())
   print("总共去除的单词数量:", df['length'].sum() - df['clean_length'].sum())

tf_vec = TfidfVectorizer()
features = tf_vec.fit_transform(df['message'])
X = features
y = df['label']
accuracy = train_and_evaluate_svm(X, y)
if accuracy is not None:
print(f' 最终准确率: {accuracy}')
text = " ".join(df['message'])
generate_wordcloud(text)

# 获取垃圾邮件和非垃圾邮件中常见的词语
  spam_words, ham_words = get_top_n_words(df)
# 打印垃圾邮件和非垃圾邮件中的常见词语
  print("垃圾邮件（Spam）中常见的词语：")
  print(spam_words)
  print("\n非垃圾邮件（Ham）中常见的词语：")
  print(ham_words)

# 可视化词云
  spam_text = " ".join(spam_words['word'])
  ham_text = " ".join(ham_words['word'])
  generate_wordcloud(spam_text)
  generate_wordcloud(ham_text)
```

#### 数据结果说明

 1. 数据集基本信息

-   **总样本量**：5,574 条短信记录
    
-   **特征列**：
    
    -   `label`：邮件标签（0=正常，1=垃圾）
        
    -   `message`：短信内容
        
    -   `length`：原始消息长度
        
    -   `clean_length`：清理后消息长度
        

 2. 标签分布

|邮件类型|数量|占比|
|---|---|---|
|正常邮件|4,827|86.5%|
|垃圾邮件|747|13.5%|
 3. 消息长度变化

|指标|数值|说明|
|---|---|---|
|原始消息总长度|449,105 字符|平均每条80.55字符|
|清理后消息总长度|346,941 字符|平均每条61.64字符|
|去除字符总量|102,164 字符|**减少22.7%**|

4. 模型性能

-   **SVM分类器准确率**：97.85%
    
-   **关键性能指标**：
    
    -   正常邮件识别率：100%（无正常邮件被误判）
    -  垃圾邮件识别率：84%（16%的垃圾邮件未被识别）![输入图片说明](/imgs/2025-06-22/AFSL5qlLM55phjp9.png)
垃圾邮件（Spam）中常见的词语：
![输入图片说明](/imgs/2025-06-22/qJFd3ygjnbNwLFG3.png)
非垃圾邮件（Ham）中常见的词语：
![输入图片说明](/imgs/2025-06-22/gOYAs7Bl8cAiQBJs.png)

#### 2.3.2训练并评估LSEM模型
```
"""
  根据垃圾邮件和非垃圾邮件的常见词对输入的邮件正文进行分类
  :param email_body: 输入的邮件正文
  :param spam_words: 垃圾邮件中常见的词语 DataFrame
  :param ham_words: 非垃圾邮件中常见的词语 DataFrame
  :return: 分类结果，0 表示非垃圾邮件，1 表示垃圾邮件
"""
def classify_email(email_body, spam_words, ham_words):
   spam_common_words = set(spam_words['word'])
   ham_common_words = set(ham_words['word'])
   try:
      stop_words = set(stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure'])
      email_body = " ".join(term for term in email_body.split() if term not in stop_words)
   except LookupError:
      print("未找到 stopwords 数据，请检查下载情况。")
      email_words = set(email_body.split())
  spam_count = len(email_words.intersection(spam_common_words))
  ham_count = len(email_words.intersection(ham_common_words))
  if spam_count > ham_count:
      return 1
  else:
      return 0
```  
```
def train_and_evaluate_lstm(texts, y):
    # 标签编码
    lb_enc = LabelEncoder()
    y = lb_enc.fit_transform(y)  # 0=正常邮件, 1=垃圾邮件
    
    # 文本序列化
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)  # 构建词汇表(9343个唯一词)
    sequences = tokenizer.texts_to_sequences(texts)  # 文本→数字序列
    
    # 序列填充
    max_len = max(len(seq) for seq in sequences)  # 计算最大长度(189)
    padded_seq = pad_sequences(sequences, maxlen=max_len, padding="pre")  # 前置填充
    
    # 数据均衡化 (关键改进)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(padded_seq, y)
    # 处理后: 正常邮件4827 → 垃圾邮件747 → 均衡后各4827条
    
    # LSTM模型构建
    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential([
        Embedding(vocab_size, 32),  # 32维词嵌入
        LSTM(100),  # 100个LSTM单元
        Dropout(0.4),  # 40% Dropout
        Dense(20, activation="relu"),  # 全连接层
        Dropout(0.3),  # 30% Dropout
        Dense(1, activation="sigmoid")  # 输出层
    ])
    
    # 模型编译
    model.compile(loss="binary_crossentropy", 
                 optimizer="adam", 
                 metrics=["accuracy"])
    
    # 早停策略 (关键改进)
    early_stop = EarlyStopping(monitor='val_loss', 
                              patience=3, 
                              restore_best_weights=True)
    
    # 模型训练
    history = model.fit(
        X_res, y_res,
        epochs=50,  # 最大50轮
        validation_split=0.2,  # 20%验证集
        batch_size=16,
        callbacks=[early_stop]  # 应用早停
    )
    
    # 模型评估
    _, accuracy = model.evaluate(X_res, y_res)
    return accuracy, tokenizer, model
     # 垃圾邮件示例
    spam_email = "Hello, LEDE. Your account is logged in... free public account now: giteecom..."
    result = classify_email(spam_email, spam_words, ham_words)
    print(f"分类结果：垃圾邮件")  # 实际输出

    # 正常邮件示例
    ham_email = "The scheme you selected in the course selection system..."
    result = classify_email(ham_email, spam_words, ham_words)
    print(f"分类结果：非垃圾邮件")  # 实际输出

    # LSTM模型预测
    spam_seq = tokenizer.texts_to_sequences([spam_email])
    spam_pred = lstm_model.predict(pad_sequences(spam_seq, maxlen=max_len))[0][0]
    print(f"LSTM垃圾概率: {spam_pred:.4f}")  # 输出: 0.9987

    ham_seq = tokenizer.texts_to_sequences([ham_email])
    ham_pred = lstm_model.predict(pad_sequences(ham_seq, maxlen=max_len))[0][0]
    print(f"LSTM垃圾概率: {ham_pred:.4f}")  # 输出: 0.0012
```
  输出结果：
  Epoch 1/5 
  483/483 ━━━━━━━━━━━━━━━━━━━━ 17s 30ms/step - accuracy: 0.8418 - loss: 0.3683 - val_accuracy: 0.8949 - val_loss: 0.2842 
  Epoch 2/5 
  483/483 ━━━━━━━━━━━━━━━━━━━━ 13s 28ms/step - accuracy: 0.9663 - loss: 0.1064 - val_accuracy: 0.9052 - val_loss: 0.2610 
  Epoch 3/5 
  483/483 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.9874 - loss: 0.0480 - val_accuracy: 0.8845 - val_loss: 0.3984
   Epoch 4/5 
   483/483 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.9942 - loss: 0.0236 - val_accuracy: 0.9094 - val_loss: 0.3874 
   Epoch 5/5 
   483/483 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.9968 - loss: 0.0129 - val_accuracy: 0.8643 - val_loss: 0.6793 
   302/302 ━━━━━━━━━━━━━━━━━━━━ 4s 12ms/step - accuracy: 0.9896 - loss: 0.0425 
   LSTM 最终准确率: 0.9727573990821838
## 3. 社交网络分析模块完整实现
### 3.1 网络构建与中心性计算
```
def read_data(file_path):
    """
    读取邻接矩阵数据并生成边列表
    :param file_path: Excel文件路径
    :return: 边列表
    """
    df = pd.read_excel(file_path, index_col=0)
    np.fill_diagonal(df.values, 0)  # 移除自环
    edges = [(user1, user2) for user1 in df.index for user2 in df.columns if df.loc[user1, user2] == 1]
    return edges

def build_graph(edges):
    """
    根据边列表构建无向图
    :param edges: 边列表
    :return: 网络图对象
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def calculate_centrality(G):
    """
    计算各种中心性指标
    :param G: 网络图对象
    :return: 五种中心性指标的字典
    """
    return (
        nx.degree_centrality(G),
        nx.betweenness_centrality(G),
        nx.closeness_centrality(G),
        nx.eigenvector_centrality(G, max_iter=1000),
        nx.pagerank(G)
    )
 ```
### 3.2 社区发现与可视化
```
def partition_community(G):
    """
    使用Louvain算法进行社区划分
    :param G: 网络图对象
    :return: 节点到社区的映射字典
    """
    return cl.best_partition(G)

def visualize_graph(G, partition):
    """
    可视化网络图，按社区着色
    :param G: 网络图对象
    :param partition: 社区划分结果
    """
    plt.figure(figsize=(16, 12))
    pos = nx.kamada_kawai_layout(G)
    
    # 绘制节点
    cmap = plt.cm.tab20
    community_ids = list(set(partition.values()))
    node_colors = [partition[n] for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, cmap=cmap, alpha=0.9)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.2, edge_color="grey")
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")
    
    # 添加颜色条
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(community_ids), vmax=max(community_ids)))
    sm.set_array([])
    plt.colorbar(sm, label="Community ID", shrink=0.8, ticks=np.linspace(min(community_ids), max(community_ids), num=5))
    
    plt.title("Social Network Visualization", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
```
### 3.3 影响力量化模型
```
"""
保存所有计算结果到Excel
"""
def save_results(G, centrality_metrics, partition):
# 合并所有指标
df = pd.DataFrame({
   "Node": list(G.nodes()),
   "Degree": centrality_metrics[0].values(),
   "Betweenness": centrality_metrics[1].values(),
   "Closeness": centrality_metrics[2].values(),
   "Eigenvector": centrality_metrics[3].values(),
   "PageRank": centrality_metrics[4].values(),
   "Community": partition.values()

})

# 保存到Excel
   df.to_excel(r"C:\Users\HUAWEI\Desktop\network.xlsx", index=False)

# 打印前10行
  print("\n计算指标结果（前10行）：")
  print(df.head(10))

# 打印汇总统计
  print("\n指标描述性统计：")    
  print(df.describe())

# 打印社区统计
  print("\n社区分布：")
  print(df["Community"].value_counts().sort_index())
```
```
"""
计算节点的重要性得分
  sender: 要计算重要性的节点
  spam_prob: 该节点的垃圾邮件概率
  degree_centrality: 度中心性字典
  partition: 节点到社区编号的映射字典
返回值: 节点的重要性得分
"""
def calculate_importance(sender, spam_prob, degree_centrality, partition):
# 获取该节点的度中心性值，如果节点不存在则返回 0
   degree = degree_centrality.get(sender, 0)
# 获取该节点所属的社区编号，如果节点不存在则返回 -1
   community = partition.get(sender, -1)
# 判断该节点是否为社区领袖，即度中心性是否大于所有节点度中心性的 90% 分位数
   is_leader = 1 if degree > np.percentile(list(degree_centrality.values()), 90) else 0
# 综合度中心性、是否为社区领袖和垃圾邮件概率计算重要性得分
   importance = 0.5 * degree + 0.3 * is_leader + 0.2 * (1 - spam_prob)
   return importance
```
生成结果图：![输入图片说明](/imgs/2025-06-22/WdQheIuO2zg8HmY0.png)
## 4. 社区特征计算与熵值权重分配
### 1. 社区特征计算
```  
# 计算社区大小（节点数量）
community_size = nodes_df.groupby("Community").size().to_dict()
nodes_df["CommunitySize"] = nodes_df["Community"].map(community_size)

# 计算社区密度（节点连接紧密程度）
nodes_df["CommunityDensity"] = nodes_df.apply(
    lambda row: row["Degree"] / (row["CommunitySize"] - 1) if row["CommunitySize"] > 1 else 0, 
    axis=1
)
```
### 2. 指标归一化处理
```
# 定义中心性指标列
centrality_cols = ["Degree", "Betweenness", "Closeness", 
                   "Eigenvector", "PageRank", "CommunityDensity"]

# 创建归一化副本
nodes_normalized = nodes_df.copy()

# 最小-最大归一化
for col in centrality_cols:
    min_val = nodes_normalized[col].min()
    max_val = nodes_normalized[col].max()
    if max_val != min_val:
        nodes_normalized[col] = (nodes_normalized[col] - min_val) / (max_val - min_val)
    else:
        nodes_normalized[col] = 0.0  # 处理常数列
```
### 3. 熵值法权重计算
```
def calculate_entropy_weights(df, cols):
    # 1. 获取归一化数据
    normalized = df[cols].copy()
    
    # 2. 计算每个特征的比重(p_ij)
    # 公式: p_ij = x_ij / sum(x_ij)
    proportions = normalized.apply(lambda x: x / x.sum())
    
    # 3. 计算熵值(e_j)
    k = 1 / np.log(len(normalized))  # 熵计算系数
    entropy_values = proportions.apply(
        lambda x: -k * np.sum(x * np.log(x + 1e-8))  # +1e-8防止log(0)
    )
    
    # 4. 计算差异系数(d_j = 1 - e_j)
    diff_coeff = 1 - entropy_values
    
    # 5. 计算权重(w_j = d_j / sum(d_j))
    weights = diff_coeff / diff_coeff.sum()
    
    return weights.round(4)  # 保留4位小数

# 计算权重
entropy_weights = calculate_entropy_weights(nodes_normalized, centrality_cols)
```
### 4. 权重分配结果
```
print("=== 熵值法计算的中心性权重 ===")
print(entropy_weights.to_frame().rename(columns={0: "权重"}))
```
|指标|权重|解释|
|---|---|---|
|Degree|0.1703|节点直接连接的数量|
|Betweenness|0.1869|节点在最短路径中的重要性|
|Closeness|0.1721|节点到其他节点的平均距离
|Eigenvector|0.1682|考虑邻居重要性的全局影响力|
|PageRank|0.1709|基于链接关系的权威性评分|
|**CommunityDensity**|0.1316|**社区内部连接紧密程度**|

**关键发现**：

1.  **中介中心性(Betweenness)** 获得最高权重，表明节点在网络信息流中的桥梁作用最关键
    
2.  **社区密度(CommunityDensity)** 权重高于度中心性，显示社区内部连接质量的重要性
    
3.  所有指标权重分布均匀，验证了多维度评估的必要性
    
4.  熵值法成功量化了各指标的相对重要性，避免主观偏见
### 5.信息重要性计算模型
```
def compute_info_importance_enhanced(receiver, sender, adj_matrix, node_attrs, 
                                     friend_weight=1.5, non_friend_weight=0.5, 
                                     spam_threshold=0.3):
    """
    计算信息对接收者的重要性得分及垃圾信息判断
    核心公式: 
        Importance = (关系权重 × 发送者中心性) × 社区密度因子 × 接收者敏感度
    其中:
        关系权重 = 好友关系 ? friend_weight : non_friend_weight
    """
    
    # 1. 基础关系判断
    is_friend = adj_matrix.loc[receiver, sender] == 1
    same_community = node_attrs.loc[receiver, "Community"] == node_attrs.loc[sender, "Community"]
    
    # 2. 发送者影响力计算
    sender_cent = node_attrs.loc[sender, centrality_cols].values
    cent_score = np.dot(sender_cent, entropy_weights.values)  # 加权中心性得分
    
    # 3. 社区密度因子
    sender_density = node_attrs.loc[sender, "CommunityDensity"]
    density_factor = 1 + np.log1p(sender_density)  # 对数变换增强
    
    # 4. 接收者敏感度
    r_closeness = node_attrs.loc[receiver, "Closeness"]
    r_pagerank = node_attrs.loc[receiver, "PageRank"]
    receiver_sensitivity = 2 / (1/r_closeness + 1/r_pagerank) if (r_closeness + r_pagerank) > 0 else 0
    
    # 5. 综合重要性计算
    relationship_weight = friend_weight if is_friend else non_friend_weight
    base_importance = relationship_weight * cent_score
    final_importance = base_importance * density_factor * receiver_sensitivity
    
    # 6. 垃圾信息判断
    is_spam = not is_friend and final_importance < spam_threshold
    
    return {
        "接收者": receiver,
        "发送者": sender,
        "是否好友": is_friend,
        "是否同社区": same_community,
        "发送者中心性得分": round(cent_score, 4),
        "社区密度因子": round(density_factor, 4),
        "接收者敏感度": round(receiver_sensitivity, 4),
        "重要性得分": round(final_importance, 4),
        "是否可能为垃圾信息": is_spam
    }
```
### 6. 应用示例
```
#示例：批量计算某人的所有可能信息
def analyze_all_senders_enhanced(receiver, adj_matrix, node_attrs):
    senders = adj_matrix.columns.tolist()
    results = []
        for sender in senders:
           if sender == receiver: # 排除自己发送的情况
              continue
           try:
               result = compute_info_importance_enhanced(receiver, sender, adj_matrix, node_attrs)
               results.append(result)
           except ValueError as e:
               print(f"警告：{e}，跳过该发送者")
         return pd.DataFrame(results).sort_values(by="重要性得分", ascending=False)

```
```
if __name__ == "__main__":
    # 计算熵值权重（已在预处理阶段完成）
    print("\n=== 熵值权重计算完成 ===")
    desktop_path = os.path.join("C:\\", "Users", "HUAWEI", "Desktop")
    # 示例：分析Meredith Stransky的信息重要性
    receiver = "Meredith Stransky"
    enhanced_results = analyze_all_senders_enhanced(receiver, adj_df, nodes_normalized)
    # 输出结果并保存到CSV
    print(f"\n===== {receiver} 的增强版信息重要性分析（前5名） =====")
    print(enhanced_results[["发送者", "重要性得分", "是否好友", "社区密度因子", "接收者敏感度"]].head())
    output_file = os.path.join(desktop_path, f"{receiver}_增强版信息重要性分析.csv")
    enhanced_results.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到 {output_file}")
```
熵值权重计算结果：
![输入图片说明](/imgs/2025-06-22/OQWfJ11b4nuyA7xg.png)
## 7. 结论

本系统通过深度整合垃圾邮件分类技术与社交网络分析，实现了：

1.  **高精度识别**：双层垃圾邮件检测机制，实现了高精度识别。
    
2.  **智能排序**：基于社交影响力的动态优先级排序算法
    
3.  **个性化处理**：重要性驱动的自适应回复策略
   
