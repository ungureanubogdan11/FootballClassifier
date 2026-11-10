import pandas as pd

input_file = "players_22.csv"
train_file = "train.csv"
test_file = "test.csv"

# cleans data from players_22.csv and keeps only stats and position
# creates training data and test data

positions_by_group = {
    "goalkeeper": ["GK"],
    "defender": ["CB", "LB", "RB", "LWB", "RWB"],
    "midfielder": ["CDM", "CM", "CAM", "LM", "RM"],
    "attacker": ["LW", "RW", "CF", "ST"],
}

group_by_positions = {
    position : group 
    for group, positions in positions_by_group.items()
    for position in positions
}


stats = ["pace", "shooting", "dribbling", "passing", "defending", "physic"]
information = ["short_name", "club_name", "nationality_name", "player_positions"]


df = pd.read_csv("players_22.csv")

df["player_positions"] = df["player_positions"].str.split(",").str[0].str.strip().str.upper()

clean_df = df.loc[:, information + stats]
clean_df["player_positions"] = df["player_positions"].map(group_by_positions).fillna("unknown")
clean_df = clean_df[clean_df["player_positions"] != "goalkeeper"]

clean_df = clean_df.rename(columns={
    "player_positions" : "position", 
    "short_name" : "name",
    "club_name" : "club",
    "nationality_name" : "nationality",
    "physic" : "physical"
})


shuffled = clean_df.sample(frac=1, random_state=51).reset_index(drop=True)
split_percentage = 0.8
split_index = int(split_percentage * len(shuffled))

train_df = shuffled.iloc[:split_index]
test_df = shuffled.iloc[split_index:]

train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)