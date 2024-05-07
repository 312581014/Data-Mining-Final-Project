from datasets import load_dataset
imdb = load_dataset("DandinPower/recommendation-system-news-clicked")
import csv
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="D:/研究所/資料探勘/Final Project/deberta-v3-xsmall-checkpoint-200",device=0)
with open('predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id","p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12","p13","p14","p15"])

    for i in range(0, len(imdb["test"]), 15):
        print("Processing batch:", i)
        batch_texts = [imdb["test"][j]["text"] for j in range(i, min(i + 15, len(imdb["test"])))]
        
        batch_scores = []
        for text in batch_texts:
            sentiment_prediction = classifier(text)
            label = sentiment_prediction[0]["label"]
            score = sentiment_prediction[0]["score"]
            if label == "one":
                output_score = score
            else:
                output_score = 1 - score
            batch_scores.append(output_score)
        
        writer.writerow([f'{i // 15}'] + batch_scores)

print("write in predictions.csv successfully!")