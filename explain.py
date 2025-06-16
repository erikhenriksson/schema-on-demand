import json

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

# Map register codes to full names
map_full_names = {
    "MT": "Machine translated (MT)",
    "LY": "Lyrical (LY)",
    "SP": "Spoken (SP)",
    "it": "Interview (it)",
    "os": "Other SP",
    "ID": "Interactive discussion (ID)",
    "NA": "Narrative (NA)",
    "ne": "News report (ne)",
    "sr": "Sports report (sr)",
    "nb": "Narrative blog (nb)",
    "on": "Other NA",
    "HI": "How-to or instructions (HI)",
    "re": "Recipe (re)",
    "oh": "Other HI",
    "IN": "Informational description (IN)",
    "en": "Encyclopedia article (en)",
    "ra": "Research article (ra)",
    "dtp": "Description: thing / person (dtp)",
    "fi": "FAQ (fi)",
    "lt": "Legal (lt)",
    "oi": "Other IN",
    "OP": "Opinion (OP)",
    "rv": "Review (rv)",
    "ob": "Opinion blog (ob)",
    "rs": "Religious blog / sermon (rs)",
    "av": "Advice (av)",
    "oo": "Other OP",
    "IP": "Informational persuasion (IP)",
    "ds": "Description: intent to sell (ds)",
    "ed": "News & opinion blog / editorial (ed)",
    "oe": "Other IP",
}


def load_and_process_data(
    filepath="data/processed/descriptors_with_explainers.jsonl", register_threshold=0.4
):
    """Load and process the dataset"""
    print(f"Loading data from {filepath}...")

    documents = []
    with open(filepath, "r") as f:
        for line in f:
            documents.append(json.loads(line.strip()))

    print(f"Loaded {len(documents)} documents")

    # Get register types from first document
    register_types = list(documents[0]["register_probabilities"].keys())
    register_types.sort()

    # Extract features and labels
    feature_dicts = []
    register_labels = []

    for doc in documents:
        # Create feature dictionary from descriptors (colon-delimited)
        feature_dict = {}
        for descriptor in doc["descriptors"]:
            prefix = descriptor.split(":")[0].lower().strip()
            if prefix:
                feature_dict[prefix] = 1

        feature_dicts.append(feature_dict)

        # Convert register probabilities to binary labels
        register_label = []
        for register_type in register_types:
            prob = doc["register_probabilities"].get(register_type, 0.0)
            register_label.append(1 if prob >= register_threshold else 0)

        register_labels.append(register_label)

    register_labels = np.array(register_labels)

    # Filter out registers with too few examples
    min_examples = 10
    label_counts = np.sum(register_labels, axis=0)
    valid_mask = label_counts >= min_examples

    # Apply filtering
    filtered_register_types = [
        reg for i, reg in enumerate(register_types) if valid_mask[i]
    ]
    register_labels = register_labels[:, valid_mask]

    print(f"Using {len(filtered_register_types)} registers")

    # Vectorize features
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(feature_dicts)

    return X, register_labels, filtered_register_types, vectorizer


def analyze_feature_importance(
    X, y, register_types, vectorizer, class_weight={0: 1, 1: 2}, C=1.0, top_n=20
):
    """Train model and analyze feature importance"""
    print(f"\nTraining model with class_weight={class_weight}, C={C}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Train model
    model = MultiOutputClassifier(
        LogisticRegression(
            C=C,
            class_weight=class_weight,
            penalty="l2",
            max_iter=1000,
            solver="liblinear",
            random_state=42,
        )
    )

    model.fit(X_train, y_train)

    # Get predictions for F1 scores
    y_pred = model.predict(X_test)

    # Print multilabel classification report
    print(f"\nMULTILABEL CLASSIFICATION REPORT")
    print("=" * 80)

    # Create target names with full names
    target_names = [map_full_names.get(reg, reg) for reg in register_types]

    # Generate classification report for each register
    print(
        classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )
    )

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    print(f"\nFEATURE ANALYSIS - TOP {top_n} PREDICTORS PER REGISTER")
    print("=" * 80)

    # Analyze each register
    for i, register_type in enumerate(register_types):
        # Calculate F1 score for this register
        register_f1 = f1_score(y_test[:, i], y_pred[:, i], average="binary")

        # Get full name from map
        full_name = map_full_names.get(register_type, register_type)

        print(f"\n{full_name} (F1: {register_f1:.4f})")
        print("-" * 60)

        # Get coefficients for this register
        coefficients = model.estimators_[i].coef_[0]

        # Get indices sorted by coefficient value
        sorted_indices = np.argsort(coefficients)

        # Most positive predictors (strongly predict PRESENCE of this register)
        print(f"TOP {top_n} POSITIVE PREDICTORS (predict {full_name}):")
        most_positive = sorted_indices[-top_n:][::-1]  # Reverse to get highest first
        for j, idx in enumerate(most_positive, 1):
            feature_name = feature_names[idx]
            coef_value = coefficients[idx]
            print(f"  {j:2d}. {feature_name:<30} (coef: {coef_value:+.4f})")

        print("\n" + "=" * 80)


def main():
    """Main function"""
    print("Register Classification - Feature Importance Analysis")
    print("=" * 60)

    # Load and process data
    X, y, register_types, vectorizer = load_and_process_data()

    # TODO: Replace these with your best parameters from the grid search
    best_class_weight = {0: 1, 1: 2}  # Default: favor positive class
    best_C = 1.0  # Replace with your best result

    print(f"Using best parameters: class_weight={best_class_weight}, C={best_C}")

    # Analyze feature importance
    analyze_feature_importance(
        X,
        y,
        register_types,
        vectorizer,
        class_weight=best_class_weight,
        C=best_C,
        top_n=15,  # Show top 15 features per register
    )


if __name__ == "__main__":
    main()
