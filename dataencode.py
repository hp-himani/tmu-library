import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tmu.models.classification.vanilla_classifier import TMClassifier
import logging
import argparse
from tmu.tools import BenchmarkTimer
from sklearn.model_selection import train_test_split



df = pd.read_csv("mushrooms.csv")
# Convert class labels: Edible (e) -> 0, Poisonous (p) -> 1
df["class"] = df["class"].map({"e": 0, "p": 1})

# Select categorical features (excluding target column)
X = df.iloc[:, 1:]  # Exclude "class" column

# Apply One-Hot Encoding
encoder = OneHotEncoder()
X_binary = encoder.fit_transform(X).toarray().astype(np.uint32)

Y1= df["class"].values.astype(np.uint32)


X_train, X_test, y_train, y_test = train_test_split(X_binary, Y1, test_size=0.2, random_state=42)

data = {'x_train' : X_train, 'x_test' : X_test, 'y_train' : y_train, 'y_test' : y_test}
    


number_of_clauses = 40


_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=number_of_clauses, type=int)
    parser.add_argument("--T", default=2000, type=int)
    parser.add_argument("--s", default=15, type=float)
    parser.add_argument("--max_included_literals", default=50, type=int)
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

    # _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
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
            # print(res)
            benchmark2 = BenchmarkTimer(logger=None, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()
                if result > maxx :
                    maxx = result

            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")
    _LOGGER.info(maxx)


    number_of_features = X_train.shape[1]  # Number of one-hot encoded features


    # Iterate through clauses and extract literals
    clauses = []

    # Initialize a binary matrix to store clauses
    clauses_matrix = np.zeros((number_of_clauses, 2*number_of_features), dtype=int)

    for clause_idx in range(number_of_clauses):
        literals = []
        for feature_idx in range(number_of_features):
            if tm.get_ta_state(clause_idx,feature_idx) > 0:  # If TA state is active
                literals.append(f"Feature_{feature_idx}")
                clauses_matrix[clause_idx, 2*feature_idx] = 1
            elif tm.get_ta_state(clause_idx, feature_idx + number_of_features) > 0:  # If negation is active
                literals.append(f"NOT-Feature_{feature_idx}")
                clauses_matrix[clause_idx, 2*feature_idx+1] = 1

        clause_str = f"Clause {clause_idx}: {' AND '.join(literals)}"
        clauses.append(clause_str)


    output_file = "learned_clauses.txt"

    # Write the learned clauses to the file
    with open(output_file, "w") as file:
        file.write(f"Number of Clauses = {number_of_clauses}\n")
        file.write(f"Number of Features = {number_of_features}\n")
        file.write(f"Number of Literals = {2*number_of_features}\n\n")

        file.write("Learned Clauses in Binary format:\n\n")
        for clause_idx in range(number_of_clauses):
        # Convert the row of clauses_matrix to a binary string
            bin = "".join(map(str, clauses_matrix[clause_idx]))
            file.write(f"Clause {clause_idx} {bin}\n")
        
        file.write("\nLearned Clauses:\n\n")
        for clause in clauses:
            file.write(clause + "\n\n")


    # Create a DataFrame for CSV storage
    column_names = ["clause_idx"] + [f"feature_{i}" if j % 2 == 0 else f"~feature_{i}" for i in range(number_of_features) for j in range(2)]
    df = pd.DataFrame(clauses_matrix, columns=column_names[1:])

    # Add clause index as first column
    df.insert(0, "clause_idx", range(number_of_clauses))

    # Save DataFrame to CSV
    output_csv = "clauses.csv"
    df.to_csv(output_csv, index=False)
