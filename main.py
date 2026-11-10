import pandas as pd
from itertools import product

# compute stats data relative to overall 

train_df = pd.read_csv("train.csv")

stats = ["pace", "shooting", "dribbling", "passing", "defending", "physical"]

def compute_relative (df):
    df["overall"] = 0

    for stat in stats:
        df["overall"] += df[stat]
    df["overall"] /= len(stats)

    for stat in stats:
        if stat == "overall":
            continue
        df[f"relative_{stat}"] = df[stat] - df["overall"]

levels = ["low", "medium", "high"]
low_threshold = -5
high_threshold = 6
bins = [-100, low_threshold, high_threshold, 100]

# apply labels to every stat
def apply_labels(df):
    for stat in stats:
        if stat == "overall":
            continue
        df[f"relative_{stat}"] = pd.cut(df[f"relative_{stat}"], bins=bins, labels=levels, include_lowest=True)

    
# preprocess train data
compute_relative(train_df)
apply_labels(train_df)

# from train data
attackers = train_df[train_df["position"] == "attacker"]
midfielders = train_df[train_df["position"] == "midfielder"]
defenders = train_df[train_df["position"] == "defender"]

p_att = len(attackers) / len(train_df) 
p_mid = len(midfielders) / len(train_df)
p_def = len(defenders) / len(train_df)


def relative_stat_probabilities(players, stat) -> list: # probability of players from a certain category to have stat set to one of the levels
    relative_column = f"relative_{stat}"
    probabilities = {}

    for level in levels:
        count = len(players[players[relative_column] == level])
        probabilities[level] = (count + 1) / (len(players) + len(levels)) # add-one smoothing

    return probabilities


# based on train data compute probability for every stat and position
groups = {
    "att" : attackers,
    "mid" : midfielders,
    "def" : defenders,
}

broad_p = {
    "att" : p_att,
    "mid" : p_mid,
    "def" : p_def
}

p = {
    position: {
        stat: relative_stat_probabilities(players, stat)
        for stat in stats
    }
    for position, players in groups.items()
}

# calculate probabilities for smart model

parents = {
    "dribbling": ["pace"],
    "passing": ["dribbling"],
    "shooting": ["passing", "dribbling", "pace"],
    "defending": ["physical", "pace"],
}

conditional_p = {}

for pos, players in groups.items():
    conditional_p[pos] = {}

    for stat, parent_stats in parents.items():
        table = {}

        for parent_levels in product(levels, repeat=len(parent_stats)):
            subset = players

            for parent_stat, level in zip(parent_stats, parent_levels):
                subset = subset[subset[f"relative_{parent_stat}"] == level]

            table[parent_levels] = relative_stat_probabilities(subset, stat)

        conditional_p[pos][stat] = table

# test data
test_df = pd.read_csv("test.csv")

# preprocess test data
compute_relative(test_df)
apply_labels(test_df)

positions = ["att", "mid", "def"]

# naive model

for pos in positions:
    pos_key = f"scor_{pos}"
    test_df[pos_key] = broad_p[pos]

    for stat in stats:
        stat_levels = test_df[f"relative_{stat}"]
        probs = stat_levels.map(p[pos][stat]).astype(float)
        test_df[pos_key] *= probs

score_to_position = {
    "scor_att" : "attacker",
    "scor_mid" : "midfielder",
    "scor_def" : "defender"
}

score_columns = list(score_to_position.keys())

test_df["prediction"] = test_df[score_columns].idxmax(axis=1).map(score_to_position)

correct = test_df["prediction"] == test_df["position"]
accuracy = correct.mean()

print(f"Naive model accuracy: {accuracy * 100:.2f}%")

# smart model
smart_score_to_position = {}

for pos in positions:
    score_column = f"smart_scor_{pos}"
    smart_score_to_position[score_column] = score_to_position[f"scor_{pos}"]
    test_df[score_column] = broad_p[pos]

    for stat in stats:
        stat_levels = test_df[f"relative_{stat}"]

        if stat in parents:
            parent_columns = [f"relative_{parent}" for parent in parents[stat]]
            parent_rows = test_df[parent_columns].itertuples(index=False, name=None)
            table = conditional_p[pos][stat]

            probs = pd.Series(
                [
                    table[parent_levels][stat_level]
                    for parent_levels, stat_level in zip(parent_rows, stat_levels)
                ],
                index=test_df.index,
                dtype=float,
            )
        else:
            probs = stat_levels.map(p[pos][stat]).astype(float)

        test_df[score_column] *= probs

smart_score_columns = list(smart_score_to_position)

test_df["smart_prediction"] = (
    test_df[smart_score_columns]
    .idxmax(axis=1)
    .map(smart_score_to_position)
)

smart_accuracy = (test_df["smart_prediction"] == test_df["position"]).mean()
print(f"Smart model accuracy: {smart_accuracy:.2%}")
print(f"Difference: {(smart_accuracy - accuracy) * 100:.2f} percentage points")