import argparse
import pandas as pd
from src.utils import get_news_info, combined_string, data_preprocess
from datasets import load_dataset, Dataset

def main(args):
    train_behaviors = pd.read_csv(args.input_path_train + '/train_behaviors1.tsv', sep='\t')
    train_news = pd.read_csv(args.input_path_train + '/train_news.tsv', sep='\t')
    test_behaviors = pd.read_csv(args.input_path_test + '/test_behaviors1.tsv', sep='\t')
    test_news = pd.read_csv(args.input_path_test + '/test_news.tsv', sep='\t')

    train_news['news_id'] = train_news['news_id'].astype(str)
    train_news_dict = train_news.set_index('news_id').to_dict(orient='index')

    test_news['news_id'] = test_news['news_id'].astype(str)
    test_news_dict = test_news.set_index('news_id').to_dict(orient='index')
    
    data_preprocess(train_behaviors, train_news, train_news_dict, args.output_path,"train")
    data_preprocess(test_behaviors, test_news, test_news_dict, args.output_path,"test")
    
    dataset = load_dataset("./dataset")
    dataset.push_to_hub("frank")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_train', type=str, required=True)
    parser.add_argument('--input_path_test', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    main(args)