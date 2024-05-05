import csv
import random
from datasets import load_dataset, Dataset
import os
import time

def get_news_info(click_news_info,news_dict):
    if click_news_info in news_dict:
        return f"{click_news_info}: [category is : {news_dict[click_news_info]['category']}; subcategory is : {news_dict[click_news_info]['subcategory']}; title is : {news_dict[click_news_info]['title']}]"
    else:
        print("news_id not found")
        return None
    
def combined_string(combined_string_for_click_news,impressions_split):
    return f"{combined_string_for_click_news} [SEP] {impressions_split}"

def output_csv(data, output_path):
    fieldnames = ['text', 'label']
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def data_preprocess(train_behaviors, train_news, news_dict, output_path, mode):
    random.seed(time.time())
    total_combined_data = []
    for i, index in train_behaviors.iterrows():
        print(i)
        click_news = index['clicked_news'].split(' ')
        impressions = index['impressions'].split(' ')
        click_news_info_list = []

        for click_news_info in click_news:
            click_news_info_list.append(get_news_info(click_news_info,news_dict))
        if mode == "train":
            for impressions_split in impressions:
                if len(click_news_info_list) > 5:
                    random_sample = random.sample(click_news_info_list, 5)
                else:
                    random_sample = click_news_info_list
                # random_sample = random.sample(click_news_info_list, 5)
                combined_string_for_click_news = ' '.join(random_sample)
                impressions_split = impressions_split.split('-')
                if impressions_split[1] == '1':
                    total_combined_data.append({
                        'text': combined_string(combined_string_for_click_news, get_news_info(impressions_split[0],news_dict)),
                        'label': impressions_split[1]
                    })
                elif impressions_split[1] == '0':
                    if(random.random() > 0.75):
                        total_combined_data.append({
                            'text': combined_string(combined_string_for_click_news, get_news_info(impressions_split[0],news_dict)),
                            'label': impressions_split[1]
                        })
            # print("\n".join(map(str, total_combined_data)))
        elif mode == "test":
            for impressions_split in impressions:
                if len(click_news_info_list) > 5:
                    random_sample = random.sample(click_news_info_list, 5)
                else:
                    random_sample = click_news_info_list
                combined_string_for_click_news = ' '.join(random_sample)
                total_combined_data.append({
                    'text': combined_string(combined_string_for_click_news, get_news_info(impressions_split,news_dict)),
                    'label': -1
                })
        else:  
            print("mode not found")
            return None
    if mode == "train":
        random.shuffle(total_combined_data)
        train_size = int(0.8 * len(total_combined_data))
        train_data = total_combined_data[:train_size]
        valid_data = total_combined_data[train_size:]
        output_csv(train_data, output_path + '/train.tsv')
        print("train_data done")
        output_csv(valid_data, output_path + '/valid.tsv')
        print("valid_data done")
    else:
        output_csv(total_combined_data, output_path + '/test.tsv')
        print("test_data done")