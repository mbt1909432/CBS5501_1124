import os
import stanza
import nltk
from nltk.corpus import stopwords, wordnet as wn
from string import punctuation
import matplotlib.pyplot as plt
from nltk.text import Text
from nltk.draw.dispersion import dispersion_plot
import pandas as pd

# 指定数据目录
data_directory = "./the_Analects_corpus"
CONLL_directory="./result/stanza_pipeline_The_Analects"

# 获取文件夹中的所有 .txt 文件
txt_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.txt')]
ff=os.listdir(data_directory)
print(ff)
# 初始化 Stanza NLP 管道
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse')

def CONLL():
    global ff

    # 遍历每个文件并处理内容
    for index,file_path in enumerate(txt_files):
        result=""
        current_file=ff[index]
        print(current_file)
        file_name = fr"{CONLL_directory}/{current_file}"
        if os.path.exists(fr"{file_name}"):
            print(f"文件 {file_name} 存在")
            continue

        print(f"正在处理文件: {file_path}")
        # 打开并读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 使用 Stanza 处理文本
        doc = nlp(content)

        # 格式化输出处理结果
        print(f"文件: {file_path} 的处理结果:")
        result+=f"文件: {file_path} 的处理结果:\n"
        for sent in doc.sentences:
            for word in sent.words:
                print(f'{word.id}\t{word.text}\t{word.lemma}\t{word.pos}\t{word.head}\t{word.deprel}')
                result +=f'{word.id}\t{word.text}\t{word.lemma}\t{word.pos}\t{word.head}\t{word.deprel}\n'

        result+="\n" + "=" * 50 + "\n"
        print("\n" + "=" * 50 + "\n")  # 分隔每个文件的输出

        with open(file_name,'w',encoding='utf-8') as file:
            file.write(result)


def method():
    # 必要资源
    #stanza.download('en', verbose=False)
    #nltk.download('stopwords')
    #nltk.download('wordnet')
    #nltk.download('omw-1.4')

    result=""

    # 需要处理的文件
    target_files = [
        'yong-ye.txt', 'xue-er.txt', 'xian-wen.txt', 'ba-yi.txt',
        'wei-ling-gong.txt', 'tai-bo.txt', 'shu-er.txt', 'li-ren.txt',
        'zi-lu.txt', 'yang-huo.txt', 'yan-yuan.txt', 'ji-shi.txt',
        'xian-jin.txt', 'zi-han.txt', 'wei-zheng.txt', 'zi-zhang.txt',
        'yao-yue.txt', 'wei-zi.txt', 'gong-ye-chang.txt', 'xiang-dang.txt'
    ]

    # 拼接完整路径
    txt_files = [os.path.join(data_directory, f) for f in target_files]

    # 初始化 Stanza NLP pipeline
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', verbose=False)

    # 准备停用词和标点符号
    stops = stopwords.words('english')

    # 2. 读取文件并处理文本
    # 初始化存储所有文件的词汇的列表
    all_words = []

    for file_path in txt_files:
        print(f"正在处理文件: {file_path}")

        # 打开并读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 使用 Stanza 处理文本
        doc = nlp(content)

        # 将所有单词提取到 all_words 列表中
        for sent in doc.sentences:
            for word in sent.words:
                all_words.append(word.text.lower())

    # 3. 创建词频统计
    total = len(all_words)  # 总单词数
    myFreq = {}
    for word in all_words:
        if word in myFreq:
            myFreq[word] += 1
        else:
            myFreq[word] = 1

    # 对词频进行排序
    sorted_freq = sorted(myFreq.items(), key=lambda x: x[1], reverse=True)

    # 4. 计算文本指标

    # Type-token ratio (TTR)
    TTR_The_Analects = len(myFreq.keys()) / total
    print("Type-token ratio for The Analects: ", TTR_The_Analects)
    result+="Type-token ratio for The Analects:"+str(TTR_The_Analects)+"\n"

    # 实词比例 (Content Word Ratio)
    content_freqs = sum(myFreq[word] for word in myFreq if word not in stops and word not in punctuation)
    ratio = content_freqs / total if total > 0 else 0
    print("Ratio of content words: ", ratio)
    result+="Ratio of content words: "+ str(ratio)+"\n"

    # Hapax Legomena Ratio (仅出现一次的单词比例)
    hapax_freqs = [x for x in sorted_freq if x[1] == 1]
    HLR = len(hapax_freqs) / total if total > 0 else 0
    print("Hapax legomena ratio: ", HLR)
    result += "Hapax legomena ratio: " + str(HLR) + "\n"

    # 词汇量大小
    vocabulary_size = len(myFreq.keys())
    print("Vocabulary size: ", vocabulary_size)
    result += "Vocabulary size: " + str(vocabulary_size) + "\n"

    # 单词 "virtue" 的相对频率
    word = 'virtue'
    RL = myFreq[word] / total if word in myFreq.keys() else 0
    print("Relative frequency of 'virtue': ", RL)
    result += "Relative frequency of 'virtue': " + str(RL) + "\n"

    # 5. 数据可视化

    # 绘制 Zipf 分布图
    x_axis = [x + 1 for x in range(len(sorted_freq))]
    y_axis = [x[1] for x in sorted_freq]

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis[:500], y_axis[:500], label="Zipf's Law")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Zipf's Law Distribution")
    plt.legend()
    plt.show()

    # 创建 NLTK 的 Text 对象
    all_words_texts = Text(all_words)

    # 目标词列表
    targets = ['virtue', 'superior', 'government', 'master']

    # 绘制目标词的分布图 Plot
    plt.figure(figsize=(12, 6))
    dispersion_plot(all_words_texts, targets, ignore_case=True, title="Dispersion Plot for Target Words")
    plt.show()

    # 6. 使用 WordNet 进行语义分析

    # 要分析的单词
    word_to_analyze = "virtue"

    # 查找单词的同义词集（Synsets）
    synsets = wn.synsets(word_to_analyze)
    print(f"\nSynsets for '{word_to_analyze}':")
    result += f"\nSynsets for '{word_to_analyze}':\n"
    for syn in synsets:
        print(f"- {syn.name()} : {syn.definition()}")
        result+=f"- {syn.name()} : {syn.definition()}\n"

    # 选择一个同义词集进行深入分析
    if synsets:
        selected_synset = synsets[0]
        print("\nAnalyzing the first synset:")
        result += "\nAnalyzing the first synset:\n"
        print(f"Name: {selected_synset.name()}")
        result += f"Name: {selected_synset.name()}\n"
        print(f"Definition: {selected_synset.definition()}")
        result += f"Definition: {selected_synset.definition()}\n"
        print(f"Example sentences: {selected_synset.examples()}")
        result += f"Example sentences: {selected_synset.examples()}\n"

        # 获取同义词
        synonyms = selected_synset.lemma_names()
        print(f"\nSynonyms: {synonyms}")
        result+=f"\nSynonyms: {synonyms}\n"

        # 获取反义词（Antonyms）
        antonyms = []
        for lemma in selected_synset.lemmas():
            if lemma.antonyms():
                antonyms.extend(lemma.antonyms())
        print(f"Antonyms: {[ant.name() for ant in antonyms]}")
        result += f"Antonyms: {[ant.name() for ant in antonyms]}\n"

        # 获取上义词（Hypernyms）
        hypernyms = selected_synset.hypernyms()
        print(f"\nHypernyms (more general concepts):")
        result += f"\nHypernyms (more general concepts):\n"
        for hypernym in hypernyms:
            print(f"- {hypernym.name()} : {hypernym.definition()}")
            result += f"- {hypernym.name()} : {hypernym.definition()}\n"

        # 获取下义词（Hyponyms）
        hyponyms = selected_synset.hyponyms()
        print(f"\nHyponyms (more specific concepts):")
        result += f"\nHyponyms (more specific concepts):\n"
        for hyponym in hyponyms:
            print(f"- {hyponym.name()} : {hyponym.definition()}")
            result += f"- {hyponym.name()} : {hyponym.definition()}\n"

        # 获取同一层级的词（Coordinate terms）
        coordinates = selected_synset.hypernyms()[0].hyponyms() if hypernyms else []
        print(f"\nCoordinate terms (words sharing the same hypernym):")
        result += f"\nCoordinate terms (words sharing the same hypernym):\n"
        for coord in coordinates:
            print(f"- {coord.name()} : {coord.definition()}")
            result += f"- {coord.name()} : {coord.definition()}\n"
    else:
        print(f"No synsets found for '{word_to_analyze}'.")
        result+=f"No synsets found for '{word_to_analyze}'.\n"

    with open(fr"{CONLL_directory}/analysis_result2.txt",'w',encoding='utf-8') as file:
        file.write(result)

    targets = ['virtue', 'superior', 'government', 'master']

    # 绘制目标词的分布图 Plot
    plt.figure(figsize=(12, 6))
    dispersion_plot(all_words_texts, targets, ignore_case=True, title="Dispersion Plot for Target Words")
    plt.show()

def method2():
    # 初始化 Stanza NLP 管道
    nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse', verbose=False)

    # 要处理的 4 句话
    sentences = [
        """They are few who, being filial and fraternal, are fond of offending against their superiors. There have been none, who, not liking to offend against their superiors, have been fond of stirring up confusion. The superior man bends his attention to what is radical. That being established, all practical courses naturally grow up. Filial piety and fraternal submission! - are they not the root of all benevolent actions?""",
        """If the scholar be not grave, he will not call forth any veneration, and his learning will not be solid. Hold faithfulness and sincerity as first principles. Have no friends not equal to yourself. When you have faults, do not fear to abandon them.""",
        """In practicing the rules of propriety, a natural ease is to be prized. In the ways prescribed by the ancient kings, this is the excellent quality, and in things small and great we follow them. Yet it is not to be observed in all cases. If one, knowing how such ease should be prized, manifests it, without regulating it by the rules of propriety, this likewise is not to be done.""",
        """He who aims to be a man of complete virtue in his food does not seek to gratify his appetite, nor in his dwelling place does he seek the appliances of ease; he is earnest in what he is doing, and careful in his speech; he frequents the company of men of principle that he may be rectified - such a person may be said indeed to love to learn."""
    ]

    # 存储所有处理结果的列表
    data = []

    # 遍历每句话并处理
    for idx, sentence in enumerate(sentences):
        print(f"正在处理第 {idx + 1} 句话: {sentence[:50]}...")  # 打印前50个字符
        doc = nlp(sentence)  # 使用 Stanza 处理句子

        for sent in doc.sentences:
            for word in sent.words:
                # 将每个单词的分析结果存储为字典
                data.append({
                    "Sentence Index": idx + 1,  # 句子索引（从1开始）
                    "Word ID": word.id,  # 单词ID
                    "Word Text": word.text,  # 单词原文
                    "Lemma": word.lemma,  # 单词词干
                    "POS": word.pos,  # 词性
                    "Head": word.head,  # 依存关系的头部
                    "DepRel": word.deprel  # 依存关系类型
                })

    # 将数据转换为 Pandas DataFrame
    df = pd.DataFrame(data)

    # 保存到 Excel 文件
    output_file_path = "result/xue_er_4_sentences_nlp_analysis.xlsx"
    df.to_excel(output_file_path, index=False)

    print(f"\n处理完成！结果已保存到: {output_file_path}")





if __name__=="__main__":
    #pass
    CONLL()
    method()
    method2()
