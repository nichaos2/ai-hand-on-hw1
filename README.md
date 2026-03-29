# AI Hands-on Assignement 1


## Problem Description

Predict the winner of a chess game.

## Dataset Description

- Domain: Chess
- Source [URL](https://wwwkagglecom/datasets/datasnaek/chess)
    - _Note_: the generation method did not really work due to authentication for some reason
- Rows: 20058
- Columns: 16

- Description of each feature

| Feature Name | Type | Description | Note |
| ------------ | ---- | ----------- | ---- |
| id           | Categorical | Unique game identifier | unique identifier - drop not for ML |
| rated        | Boolean | True if the game affects player ratings | categorical feature |
| created_at   | Numerical | Unix timestamp (start time)| conversion to datetime - difference with last_move_at|
| last_move_at | Numerical | Unix timestamp (end time)| conversion to datetime - difference with  created_at (if the game finished late at night mate in one could arrive due to fatigue ) |
| turns       | Numerical | Number of moves played| indicator for likelihood of mate in early stages - "hanging mate in one"|
| victory_status | Categorical |How the game ended among "mate- resign-outoftime-draw" |
| winner       | Categorical | white - black - draw | Target (draw is only 5% in this dataset) |
| increment_code| Categorical| Time control | High cardinality (8000+ types) - fast games (bullet-blitz) or long games (rapid-classical) |
| white_id     | Categorical | Username of White player | unique identifier - drop not for ML |
| white_rating | Numerical | level of White player| numerical feature (usually high ranks lead to less mating results as they tend to resign) |
| black_id     |Categorical|Username of Black player| unique identifier - drop not for ML |
| black_rating |Numerical|Skill level of Black player|same as white ranking |
| moves        | Text | game history in notation  | Needs Regex or string splits, but how useful it is? |
| opening_eco  |Categorical|Opening classification code| certain openings lead to more "edgy" games - Good for categorical encoding |
| opening_name |Categorical|Human-readable opening name|High cardinality |
| opening_ply  | Numerical | Length of the opening sequence | Number of theoretical moves -  sticking to theory leads to more balanced games positionally and games could go on more |

- The goal is to predict the _winner_ of the game (or if it was a draw).

From the dataset we see only 5% of the games end in draws; this actually pretty normal due to the nature of online chess with beginners and intermediate players.


## Preprocessing Approach

### Splitting the data

We split the with the 80/10/10 approach.
- we drop the target column winner from the df
- we use the method `train_test_split` of the scikit-learn library
- caution needed for the few rows where we have the value "draw" in our target; for this we use `stratify`.

The results for the splitting process are as follows

| Set |  Number of rows (percentage on df rows) | Draw Percentage |
| --- | ---------------------------- | --------------- |
| Training set  | 16046 rows (80%) | 4.74% |
| Validation set| 2006 rows (10%) | 4.74% | 
| Test set      | 2006 rows (10%) | 4.74%|


## Feature Engineering

## PCA Insights

## Model Comparison

## Best Model Designation

## Installation and Execution