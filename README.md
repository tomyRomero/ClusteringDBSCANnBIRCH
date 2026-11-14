# Welcome to my Machine Learning Research Project

## I Conducted This During Independent Research At the University of the Virgin Islands

A Python script that compares three clustering algorithms (K-means, DBSCAN and BIRCH) with scikit-learn, grouping 200 mall customers by annual income and spending score.

A school project from 2023.

(Please note all credit for the data collected goes to the following GitHub repo https://gist.github.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf), This research is for educational purposes only.

Within this Repo I have attached a PowerPoint presentation as well as The Research Report. I have also included the results of the code provided.

Here is the charts that are displayed as an outcome:

# K-Means:
![Local Image](Kmeans.png)

# DBSCAN
![Local Image](DBSCAN.png)

# BIRCH
![Local Image](BIRCH.png)

## Running it

You need Python 3.14 (the versions in `requirements.txt` were tested with it on macOS). From the repo folder:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

The script opens each chart in a window; close it to move on to the next. At the end it prints the time each algorithm took and the DBSCAN and BIRCH silhouette scores (0.114 and 0.323, the same as in `Code_Results.txt`). To run it without windows, use `MPLBACKEND=Agg python main.py`.

## Dataset

The dataset is not included in this repo: its source does not state terms that allow sharing it again. Download it yourself before running the script:

1. Get the CSV from the Kaggle page [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) (it is also in the gist credited above). The Kaggle page says it comes from Udemy's Machine Learning A-Z course and lists its licence as "Other (specified in description)" without stating any terms.
2. Save it as `Mall_Customers.csv` in the repo folder, beside `main.py`. The file name is in `.gitignore`, so your copy is not committed.
3. Check it is the right file: 200 rows of customers plus a header row with the columns `CustomerID`, `Gender`, `Age`, `Annual Income (k$)` and `Spending Score (1-100)`.

## What I would do differently today

- **Pinned versions.** The original run kept no list of versions. `requirements.txt` now pins the ones tested here, and yellowbrick 1.5 already breaks on the newest scikit-learn.
- **A fixed random seed.** K-means starts from random centres; setting `random_state` would make every run give the same curve and scores.
- **Fairer timings.** Each timer also covers the plotting and the time a chart window stays open, and the K-means timer covers ten fits, so the times compare more than the algorithms.
- **The limits of the study.** Two features, 200 customers, unscaled data, a hand-picked DBSCAN `eps` and one measure (the silhouette score) are enough for a class comparison, not for general claims about the algorithms.

## Licence

The code is under the MIT licence: see [LICENSE](LICENSE). The report and slides are my own writing. The dataset is not mine and keeps its own source's terms: it is the Kaggle [Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python), from Udemy's Machine Learning A-Z course, licence listed as "Other (specified in description)".
