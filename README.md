# FIFA Player Position Classifier

## Task and data

This project compares two Bayesian classifiers that predict the position group of a FIFA 22 outfield player. The target column, `position`, contains one of three labels: `defender`, `midfielder`, or `attacker`. The inputs are the six card ratings, `pace`, `shooting`, `passing`, `dribbling`, `defending`, and `physical`.

Goalkeepers are excluded because their cards use a different set of ratings. Name, club, and nationality are metadata. The classifiers estimate their probabilities from `train.csv` and are evaluated on the same players in `test.csv`.

## Preprocessing

For each player, the script calculates the mean of their six ratings and stores it as `overall`. It subtracts this mean from each rating to describe the player's profile relative to their own card. For example, a player with a mean of 68 and a defending rating of 52 has a relative defending value of −16.

Both CSVs are transformed using the same thresholds.

| Level | Rating minus player mean |
| --- | --- |
| `low` | At or below −6 |
| `medium` | Above −6 and at or below +5 |
| `high` | Above +5 |

Discretizing the ratings into three levels lets the models estimate probabilities by counting players in the training data. Ratings within the same interval contribute the same factor to the score.

## Naive Bayes

For each position, the model first estimates a **prior probability** from its share of the training set. It then estimates how often each rating level occurs within that position. For example, the shooting table for attackers contains the probabilities of relative shooting being `low`, `medium`, or `high` given that the player is an attacker.

To score a position for a test player, the model multiplies the prior by the six probabilities corresponding to the player's observed levels. This uses the **conditional independence assumption**. Once a position is given, the model treats each rating level as independent of the others. The position with the highest score becomes the prediction.

## Dependency-aware Bayes (smart model)

The smart model uses the same priors and scoring rule, but estimates some rating probabilities from narrower groups. Each group is defined by the candidate position and the observed levels of the rating's **parents**.

| Rating | Additional ratings used to estimate its probability |
| --- | --- |
| `pace` | None |
| `dribbling` | `pace` |
| `passing` | `dribbling` |
| `shooting` | `passing` and `dribbling` |
| `physical` | `pace` |
| `defending` | `physical` |

For example, consider a test player with `medium` shooting, `medium` passing, and `high` dribbling. To score the player as an attacker, the naive model estimates the shooting factor from all attackers. The smart model uses only attackers with `medium` passing and `high` dribbling. If 8 of those 20 attackers have `medium` shooting, the smoothed shooting factor is `(8 + 1) / (20 + 3) = 9/23`. That value supplies the single shooting factor in the product.

Both models use **Laplace smoothing** for rating levels. They add one to each level's count and three to the group total. This keeps a level absent from a training group from setting the entire position score to zero.

## Evaluation and comparison

Accuracy is the fraction of test players whose predicted position matches their label. The dependency-aware model achieved about **2 percentage points higher test accuracy** than the naive baseline in the reported run.

The gain is consistent with some rating combinations carrying information about position beyond their individual frequencies. The trade-off is that the smart model estimates probabilities from smaller groups. For example, the shooting table has nine passing–dribbling combinations per position, and rare combinations have fewer examples to learn from.

## How to run

Place the classifier script, `train.csv`, and `test.csv` in the same directory. Each CSV needs the six numeric rating columns and the `position` label. From that directory, install the dependency and run the script.

```bash
python -m pip install pandas
python main.py
```

Substitute your script's filename if it differs from `main.py`. The script trains both models from `train.csv` and prints their accuracy on `test.csv`.
