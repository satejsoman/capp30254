from pathlib import Path

import matplotlib2tikz
import matplotlib.pyplot as plt
import pandas as pd


def main():
    pr1 = pd.read_csv('../code/output/donors_choose-LATEST/LinearSVC-C1-penaltyl2_pr-data_1.csv')
    pr2 = pd.read_csv('../code/output/donors_choose-LATEST/LinearSVC-C1-penaltyl2_pr-data_2.csv')
    pr3 = pd.read_csv('../code/output/donors_choose-LATEST/LinearSVC-C1-penaltyl2_pr-data_3.csv')

    plt.figure()
    pr1.plot(x="recall", y="precision", ax=plt.gca())
    pr2.plot(x="recall", y="precision", ax=plt.gca())
    pr3.plot(x="recall", y="precision", ax=plt.gca())
    plt.title('Precision-Recall Curve for Sample SVM')
    matplotlib2tikz.save('pr-curve-raw.tex')

if __name__ == "__main__":
    main()
