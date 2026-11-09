# Fake News Detector

B.Tech 2nd year AI & ML mini project. News text ko TF-IDF se numbers mein badalkar
Logistic Regression, Naive Bayes aur Passive Aggressive classifier se **Real / Fake** classify karta hai.
Saath mein ek Streamlit web app hai jahan user news paste karke result aur confidence dekh sakta hai.

## Project structure

```
fake-news-detector/
├── data/                 # Fake.csv aur True.csv yahan rakho (git mein upload mat karo)
├── utils.py              # shared cleaning functions
├── train.py              # training + evaluation, model.pkl aur tfidf.pkl banata hai
├── app.py                # Streamlit app
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Dataset: Kaggle par **"Fake and Real News Dataset" (ISOT)** download karo aur `Fake.csv`, `True.csv`
ko `data/` folder mein rakho.

## Chalane ka tareeka

```bash
python train.py --data data      # model train + results.csv + confusion matrix
streamlit run app.py             # web app
```

`train.py` ye files banata hai: `model.pkl`, `tfidf.pkl`, `results.csv`,
`confusion_matrix.png`, `confusion_matrix_with_source.png`, `top_words.csv`.

## Important: "Reuters" wali baat

ISOT dataset mein real news ke shuru mein aksar `(Reuters)` likha hota hai, aur model wahi pakadkar
98-99% accuracy de deta hai. `train.py` isliye **do experiment** chalata hai:
source words ke saath aur source words hatakar. Dono ka comparison `results.csv` mein milta hai.
Report aur viva mein dono score dikhao.

## Team

| Member | Kaam |
|---|---|
| A | Data load, EDA charts, cleaning |
| B | Models train, comparison table, model save |
| C | Streamlit app |
| D | Evaluation, report, PPT |

## Limitations

- Model sirf is dataset ke style se seekhta hai, doosre country ya topic ki khabar par kamzor ho sakta hai.
- Ye final saboot nahi, sirf ek andaza hai.
