# AI Hands-on Assignement 1


## Problem Description

Predict the winner of a chess game.

## Dataset Description

- Domain: Chess
- Source [URL](https://wwwkagglecom/datasets/datasnaek/chess)
    - _Note_: the generation method did not really work due to authentication for some reason
- Rows: 20058
- Columns: 
    - 16, original file
    - 11, after dropping columns not used in the training (see Notes below)

- Description of each feature

| Feature Name | Type | Description | Note |
| ------------ | ---- | ----------- | ---- |
| id           | Categorical | Unique game identifier | unique identifier - **drop** for ML |
| rated        | Boolean | True if the game affects player ratings | categorical feature |
| created_at   | Numerical | Unix timestamp (start time)| conversion to datetime - difference with last_move_at|
| last_move_at | Numerical | Unix timestamp (end time)| conversion to datetime - difference with  created_at (if the game finished late at night mate in one could arrive due to fatigue ) |
| turns       | Numerical | Number of moves played| indicator for likelihood of mate in early stages - "hanging mate in one"|
| victory_status | Categorical |How the game ended among "mate- resign-outoftime-draw" | In case of draw this gives away the result; in othere cases it would be interesting to use is as a target if we are interested on how the game ends, e.g. the target is a synthetic of winner-victory status, but this gets to complicated so we migth as well **drop** it for the sake of the exercise.|
| winner       | Categorical | white - black - draw | Target (draw is only 5% in this dataset) |
| increment_code| Categorical| Time control | High cardinality - fast games (bullet-blitz) or long games (rapid-classical). The format of this column follows a pattern "number+number", like "10+0" or "5+2".|
| white_id     | Categorical | Username of White player | unique identifier - **drop** for ML |
| white_rating | Numerical | level of White player| numerical feature (usually high ranks lead to less mating results as they tend to resign) |
| black_id     |Categorical|Username of Black player| unique identifier - **drop** for ML |
| black_rating |Numerical|Skill level of Black player|same as white ranking |
| moves        | Text | game history in notation  | Needs Regex or string splits, but how useful it is? -> we could extract the captures done by the highest rated player or the checks given by some player, but on the other hand this gives the history of the game so it is not a generalistic behaviour of the model -> final verdict: **drop** |
| opening_eco  |Categorical|Opening classification code| certain openings lead to more "edgy" games - Good for categorical encoding |
| opening_name |Categorical|Human-readable opening name|High cardinality |
| opening_ply  | Numerical | Length of the opening sequence | Number of theoretical moves -  sticking to theory leads to more balanced games positionally and games could go on more |

- The goal is to predict the _winner_ of the game: white - black - draw.

From the dataset we see only 5% of the games end in draws; this actually pretty normal due to the nature of online chess with beginners and intermediate players.

_Note_: the columns notes above are dropped during the loading of teh dataset. The approach of the target to be picked out of two columns (winner+victory_status) seems really attractive, but it is left out for now for the sake of time limitations (but will be revisited)

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

### Missing values

Although the dataset we are using does not have missing values we need to create an actual strategy to handle it,
in case the dataset we will use in the future has missing values.

We use `SimpleImputer` from the scikit-learn library, as the instructions advice to use mean or median values; an alternative would be the KNNImputer. we `fit` only to the training data to avoid data-leakage and then transform all three datasets (training, test, validation).

The approach we use is the following:

| Feature Type | Columns | Strategy Applied  | Reason |
| ------------ | ------- | ----------------  | ------ |
| Target       | winner  | Drop missing rows | As stated in the instruction; a supervised model cannot learn without the answer key |
| Numerical    | turns, ratings, opening_ply | Median Imputation | Chess _ratings_ and _turn_ counts are often skewed by outliers (e.g., 2700-rated GMs or 2-turn games). Median is safer than Mean.|
| Categorical  | rated, opening_eco, etc.    | Mode: Most Frequent | Replaces missing text with the most common category found in the training data.|

_Notes_:
1. some times Scikit behaves "strange" and automatically assigns columns a different type to the data than is is actually from the `df.info()`. To this end, we explicitly assign the caterorical columns as objects using this piece of code.
    ```python
    for df in [X_train, X_validation, X_test]:
        df[cat_cols] = df[cat_cols].astype("object")
    ```

### Outlier Detection and Treatment

We adopt the IQR method since it is more appropriate for the data, as they are not uniformly distributed (bell curve).

The detection of the outlier are applied to the numerical features, except "*created_at*" and "*last_move_at*". The values of these columns are "massive" integers and are used as timestamps; timestamps represent specific points in history. Capping them to a median "date" doesn't make physical sense for a model and therefore we convert them to duration later in the preprocessing.

After the detection of the outliers we "Winsorize" the outliers rather than remove them. We follow this strategy as  to preserve the integrity of the Validation and Test sets. In a real-world scenario, if a user inputs a highly unusual 250-turn game, the model cannot simply 'drop' the user's request; it should provide a prediction. By capping, we train the model to treat extreme outliers as simply 'very high' values without allowing them to distort the mathematical weights of the Neural Network.

## Feature Engineering

## PCA Insights

## Model Comparison

## Best Model Designation

## Installation and Execution