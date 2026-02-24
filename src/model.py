from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline


# --------------------------------------------------
# Baseline Random Forest
# --------------------------------------------------
def build_baseline_model(random_state):
    return OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=500,
            n_jobs=-1,
            random_state=random_state
        )
    )


# --------------------------------------------------
# Random Forest with Class Weighting
# --------------------------------------------------
def build_weighted_model(random_state):
    return OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        )
    )


# --------------------------------------------------
# Random Forest with Random Oversampling
# --------------------------------------------------
def build_oversampled_model(random_state):
    base_rf = RandomForestClassifier(
        n_estimators=500,
        n_jobs=-1,
        random_state=random_state
    )

    ros = RandomOverSampler(random_state=random_state)

    pipeline = Pipeline([
        ("oversample", ros),
        ("rf", base_rf)
    ])

    return OneVsRestClassifier(pipeline)
