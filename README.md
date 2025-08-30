# Bharat-Data-Science-Intern

Project Overview
This project detects emotions from text. Unlike sentiment analysis (positive/negative), it is a multi-label classification task where a text can have multiple emotions.  
We built baseline models using TF-IDF + Ridge Classifier and advanced models by fine-tuning BERT. The final model also applies class weighting and threshold optimization to handle imbalanced data. Fine-grained labels are mapped into six universal emotions: Joy, Sadness, Anger, Fear, Disgust, and Surprise.

Features
- Data preprocessing and exploratory analysis  
- Baseline: TF-IDF with Ridge Regression  
- Advanced: BERT fine-tuning with attention  
- Weighted Binary Cross-Entropy loss  
- Probability threshold tuning for multi-label classification  
- Mapping 28 emotions into 6 universal categories  

Tech Stack
- Python  
- PyTorch, HuggingFace Transformers  
- scikit-learn, pandas, numpy, matplotlib  

Results
- TF-IDF + Ridge provided a benchmark  
- BERT significantly improved accuracy and recall  
- Threshold tuning and class weights improved balance across emotions
