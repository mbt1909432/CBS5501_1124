import sys
import nltk
import spacy
import os
import re
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from typing import Dict, List, Union


# 下载必要的NLTK数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# 加载spacy模型
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spacy model...")
    os.system(f"{sys.executable} -m spacy download en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

class TextAnalyzer:
    def __init__(self, data_dir="."):
        """
        初始化分析器
        :param data_dir: 文本文件所在的目录
        """
        self.data_dir = data_dir
        self.stop_words = self._get_extended_stopwords()
        self.target_concepts = {
            'virtue': ['virtue', 'virtuous', 'virtues'],
            'benevolence': ['benevolence', 'benevolent', 'jen', 'ren'],
            'ritual': ['ritual', 'rituals', 'li'],
            'wisdom': ['wisdom', 'wise', 'zhi'],
            'harmony': ['harmony', 'harmonious', 'he']
        }

    def _get_extended_stopwords(self):
        """获取扩展的停用词列表"""
        stop_words = set(stopwords.words('english'))
        additional_stops = {
            'said', 'saying', 'say', 'says',
            'would', 'could', 'should', 'shall',
            'may', 'might', 'must', 'one',
            'also', 'like', 'well', 'way',
            'make', 'made', 'makes', 'making',
            'zi', 'lu', 'gong', 'you', 'xian',
            'wen'
        }
        stop_words.update(additional_stops)
        return stop_words

    def get_text_files(self):
        """获取目录下所有txt文件"""
        text_files = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.txt'):
                full_path = os.path.join(self.data_dir, file)
                text_files.append(full_path)
        return sorted(text_files)

    def read_file_safe(self, file_path):
        """安全地读取文件，处理不同编码"""
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'gbk', 'gb18030', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Cannot read file {file_path} with any encoding")

    def combine_files(self):
        """合并所有txt文件的内容"""
        combined_text = ""
        text_files = self.get_text_files()

        if not text_files:
            print("No text files found in the specified directory!")
            return ""

        print(f"Combining {len(text_files)} text files:")
        for file_path in text_files:
            print(f"- Reading: {os.path.basename(file_path)}")
            try:
                text = self.read_file_safe(file_path)
                combined_text += text + "\n"
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")

        return combined_text

    def clean_text(self, text):
        """清理和标准化文本"""
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\n', os.linesep)

        # 确保标点符号后有空格
        punctuation_pattern = r'([.!?。！？।։۔؟၊။។፨;:])'
        text = re.sub(punctuation_pattern, r'\1' + os.linesep, text, flags=re.UNICODE)

        # 处理省略号
        text = re.sub(r'\.{3,}', '...', text)

        # 清理多余的空白字符
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def analyze_concordance(self, text, window_size=40):
        """
        对目标概念词进行concordance分析
        """
        concordance_results = {}
        words = word_tokenize(text.lower())

        for concept, variants in self.target_concepts.items():
            concordance_results[concept] = []
            for i, word in enumerate(words):
                if word in variants:
                    left = max(0, i - window_size)
                    right = min(len(words), i + window_size+ 1)

                    context = {
                        'left': ' '.join(words[left:i]),
                        'target': word,
                        'right': ' '.join(words[i+1:right])
                    }
                    concordance_results[concept].append(context)

        return concordance_results

    def analyze_collocations(self, text: str, measure: str = 'likelihood_ratio', freq_filter: int = 3) -> Dict[str, Union[List, Dict]]:
    # 检查输入是否为空
        if not text or not isinstance(text, str):
            raise ValueError("The input text must be a non-empty string.")

    # 分词并过滤停用词
        words = word_tokenize(text.lower())
        words = [w for w in words if w not in self.stop_words and w.isalnum()]

    # 创建双字搭配查找器
        bigram_finder = BigramCollocationFinder.from_words(words)
        bigram_finder.apply_freq_filter(freq_filter)

    # 定义可用的统计方法
        available_measures = {
            'pmi': BigramAssocMeasures.pmi,
            't_score': BigramAssocMeasures.student_t,
            'likelihood_ratio': BigramAssocMeasures.likelihood_ratio
    }

    # 检查传入的 measure 是否有效
        if measure not in available_measures:
            valid_measures = ', '.join(available_measures.keys())
            raise ValueError(f"Invalid measure '{measure}'. Valid options are: {valid_measures}.")

    # 选择统计方法
        score_fn = available_measures[measure]

    # 计算搭配分数
        collocations = bigram_finder.score_ngrams(score_fn)

    # 返回结果
        return {
            'top_collocations': collocations[:30],  # 返回前 30 个搭配
            'statistics': {
                'total_bigrams': bigram_finder.ngram_fd.N(),  # 总双字搭配数
                'unique_bigrams': len(bigram_finder.ngram_fd)  # 唯一双字搭配数
        }
    }

    def analyze_text(self, text):
        """综合文本分析"""
        # 清理文本
        cleaned_text = self.clean_text(text)

        # 基础分词
        words = word_tokenize(cleaned_text.lower())
        words_no_stop = [w for w in words if w not in self.stop_words and w.isalnum()]

        # 句子分析
        sentences = sent_tokenize(cleaned_text)

        # Concordance分析
        concordance_results = self.analyze_concordance(cleaned_text)

        # Collocation分析
        collocation_results = self.analyze_collocations(cleaned_text)

        # 词频统计
        word_freq = Counter(words_no_stop).most_common(30)

        return {
            'basic_stats': {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'unique_words': len(set(words_no_stop)),
                'avg_sentence_length': len(words) / len(sentences) if sentences else 0
            },
            'word_frequency': word_freq,
            'concordance': concordance_results,
            'collocations': collocation_results
        }

    def print_analysis(self, results):
        """打印分析结果"""
        print("\n" + "="*80)
        print("TEXT ANALYSIS RESULTS")
        print("="*80)

        # 打印基础统计
        stats = results['basic_stats']
        print("\nBASIC STATISTICS:")
        print(f"Total Words: {stats['word_count']}")
        print(f"Total Sentences: {stats['sentence_count']}")
        print(f"Unique Words: {stats['unique_words']}")
        print(f"Average Sentence Length: {stats['avg_sentence_length']:.2f} words")

        # 打印词频统计
        print("\nTOP 30 MOST FREQUENT WORDS:")
        print("-"*40)
        for word, count in results['word_frequency']:
            print(f"{word:<15} : {count:>5}")

        # 打印concordance分析
        print("\nCONCORDANCE ANALYSIS:")
        print("="*80)
        for concept, contexts in results['concordance'].items():
            print(f"\n{concept.upper()} ({len(contexts)} occurrences)")
            print("-"*60)
            for i, ctx in enumerate(contexts[:5], 1):
                print(f"{i}. ...{ctx['left']} [{ctx['target'].upper()}] {ctx['right']}...")

        # 打印collocation分析
        print("\nCOLLOCATION ANALYSIS:")
        print("="*80)
        collocations = results['collocations']
        print(f"\nTotal Bigrams: {collocations['statistics']['total_bigrams']}")
        print(f"Unique Bigrams: {collocations['statistics']['unique_bigrams']}")
        print("\nTop Significant Collocations:")
        for (w1, w2), score in collocations['top_collocations'][:15]:
            print(f"{w1:15} - {w2:15} : {score:.4f}")

    def save_analysis(self,results):

        stats = results['basic_stats']
        collocations = results['collocations']

        final_result="\n" + "=" * 80+"\n"+"TEXT ANALYSIS RESULTS"+'\n'+"=" * 80
        final_result+="\nBASIC STATISTICS:\n"
        final_result+=f"Total Words: {stats['word_count']}\n"
        final_result+=f"Total Sentences: {stats['sentence_count']}\n"
        final_result+=f"Unique Words: {stats['unique_words']}\n"
        final_result+=f"Average Sentence Length: {stats['avg_sentence_length']:.2f} words\n"

        final_result+="\nTOP 30 MOST FREQUENT WORDS:\n"
        final_result+="-" * 40+"\n"
        for word, count in results['word_frequency']:
            final_result+=f"{word:<15} : {count:>5}\n"

        final_result+="\nCONCORDANCE ANALYSIS:\n"
        final_result+="=" * 80+"\n"
        for concept, contexts in results['concordance'].items():
            final_result+=f"\n{concept.upper()} ({len(contexts)} occurrences)\n"
            final_result+="-" * 60+"\n"
            for i, ctx in enumerate(contexts[:5], 1):
                final_result+=f"{i}. ...{ctx['left']} [{ctx['target'].upper()}] {ctx['right']}...\n"

        final_result+="\nCOLLOCATION ANALYSIS:\n"
        final_result+="=" * 80+"\n"
        final_result+=f"\nTotal Bigrams: {collocations['statistics']['total_bigrams']}\n"
        final_result+=f"Unique Bigrams: {collocations['statistics']['unique_bigrams']}\n"
        final_result+="\nTop Significant Collocations:\n"

        for (w1, w2), score in collocations['top_collocations'][:15]:
            final_result+=f"{w1:15} - {w2:15} : {score:.4f}\n"

        with open(fr"result/analysis_result1.txt", "w", encoding='utf-8') as file:
            file.write(final_result)


def main():
    # 设置文件目录
    data_directory = "./the_Analects_corpus"  # 替换为实际路径
    analyzer = TextAnalyzer(data_dir=data_directory)

    # 合并文件
    print("Starting text analysis...")
    combined_text = analyzer.combine_files()

    if not combined_text:
        print("No text to analyze!")
        return

    # 分析文本
    print("\nAnalyzing combined text...")
    results = analyzer.analyze_text(combined_text)

    #打印结果
    analyzer.print_analysis(results)

    #保存结果
    analyzer.save_analysis(results)

if __name__ == "__main__":
    main()