import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tmu.models.classification.vanilla_classifier import TMClassifier
import logging
import argparse
from tmu.tools import BenchmarkTimer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

df = pd.read_csv("diabetes.csv")

X = df.iloc[:, :8]  
Y = df["Outcome"].values.astype(np.uint32)

# Apply KBinsDiscretizer
discretizer = KBinsDiscretizer(n_bins=15, encode='onehot-dense', strategy='quantile')
X_discrete = discretizer.fit_transform(X).astype(np.uint32)



X_train, X_test, y_train, y_test = train_test_split(X_discrete, Y, test_size=0.2, random_state=42)

data = {'x_train' : X_train, 'x_test' : X_test, 'y_train' : y_train, 'y_test' : y_test}


number_of_clauses = 20

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=number_of_clauses, type=int)
    parser.add_argument("--T", default=2000, type=int)
    parser.add_argument("--s", default=15, type=float)
    parser.add_argument("--max_included_literals", default=300, type=int)
    parser.add_argument("--device", default="CPU", type=str, choices=["CPU", "CPU_sparse", "CUDA"])
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=50, type=int)
    args = parser.parse_args()

    tm = TMClassifier(
        type_iii_feedback=False,
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        seed=42,
    )

    maxx = 0

    for epoch in range(args.epochs):
        benchmark_total = BenchmarkTimer(logger=None, text="Epoch Time")
        with benchmark_total:
            benchmark1 = BenchmarkTimer(logger=None, text="Training Time")
            with benchmark1:
                res = tm.fit(
                    data["x_train"],
                    data["y_train"],
                    metrics=["update_p"],
                )
            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()
                if result > maxx:
                    maxx = result
            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")
    _LOGGER.info(maxx)

    number_of_features = X_train.shape[1]
    clauses_matrix = np.zeros((number_of_clauses, 2 * number_of_features), dtype=int)

    for clause_idx in range(number_of_clauses):
        for feature_idx in range(number_of_features):
            if tm.get_ta_action(clause_idx, feature_idx):
                clauses_matrix[clause_idx, 2 * feature_idx] = 1
            elif tm.get_ta_action(clause_idx, feature_idx+number_of_features): 
                clauses_matrix[clause_idx, 2*feature_idx + 1] = 1

    output_file = "learned_clauses1.txt"
    with open(output_file, "w") as file:
        # file.write(f"Number of Clauses = {number_of_clauses}\n")
        # file.write(f"Number of Features = {number_of_features}\n")
        # file.write(f"Number of Literals = {2 * number_of_features}\n")
        # file.write("Learned Clauses:\n\n")
        for clause_idx in range(number_of_clauses):
            bin = "".join(map(str, clauses_matrix[clause_idx]))
            file.write(f"{bin}\n")

    # # Save input literals for test data with interleaved negated bits
    # test_literals_file = "test_literals1.txt"
    # with open(test_literals_file, "w") as file:
    #     for i in range(len(X_test)):
    #         interleaved_features = "".join(
    #             f"{bit}{1 - bit}" for bit in X_test[i].astype(int)
    #         )
    #         expected_output = str(y_test[i])
    #         file.write(interleaved_features + expected_output + "\n")


    # for i in range(number_of_features):
    #     print(i, tm.get_ta_action(0, i))