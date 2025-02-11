import pandas as pd
from Bio.Align import PairwiseAligner
from Bio.Align.substitution_matrices import load as load_substitution_matrix

def calculate_blosum_similarity(seq1, seq2, matrix_name='BLOSUM62'):
    # 加载BLOSUM62矩阵
    matrix = load_substitution_matrix(matrix_name)
    # 初始化PairwiseAligner
    aligner = PairwiseAligner()
    aligner.substitution_matrix = matrix
    # 使用PairwiseAligner进行全局对齐并返回最高得分
    score = aligner.score(seq1, seq2)
    return score

def main():
    # 加载数据
    file_path = r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\nettcr_strict0Hetero_pred_fold0.tsv"
    df = pd.read_csv(file_path, sep='\t')

    # 根据条件筛选数据
    filtered_df = df[(df['true_label'] == 1) & (df['probability'] > 0.5)]

    # 查找唯一的TCR组合
    unique_pairs = filtered_df.drop_duplicates(subset=['tra_cdr3', 'trb_cdr3'])

    # 计算每对独特peptide间的BLOSUM62相似性得分
    peptides = unique_pairs['peptide'].unique()
    scores = {}
    for i in range(len(peptides)):
        for j in range(i + 1, len(peptides)):
            score = calculate_blosum_similarity(peptides[i], peptides[j])
            scores[(peptides[i], peptides[j])] = score

    # 打印相似性得分
    for pair, score in scores.items():
        print(f"Similarity score between {pair[0]} and {pair[1]}: {score}")

    # 保存数据到新的文件
    output_file = r"D:\xy\HeteroTCR-main\data\nettcr_strict0_5folds\predresult\strict0similayscore.tsv"
    columns = ['tra_cdr3', 'trb_cdr3', 'peptide', 'true_label', 'probability', 'weights_tra', 'weights_trb']
    filtered_df.to_csv(output_file, columns=columns, sep='\t', index=False)

if __name__ == "__main__":
    main()
