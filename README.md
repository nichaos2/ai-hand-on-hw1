# AI Hands-on Assignement 1


## 1. Problem Description

Predict the winner of a chess game.

## 2. Dataset Description

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
| rated        | Categorical - Boolean | True if the game affects player ratings | categorical feature |
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

## 3. Preprocessing Approach

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

The detection of the outlier are applied to the following numerical features:
- turns
- white_rating
- black_rating
- opening_ply

The numerical features "*created_at*" and "*last_move_at*" are not treated in the outlier detection process. The values of these columns are "massive" integers and are used as timestamps; timestamps represent specific points in history. Capping them to a median "date" doesn't make physical sense for a model and therefore we convert them to duration later in the preprocessing.

After the detection of the outliers we "Winsorize" the outliers rather than remove them. We follow this strategy as  to preserve the integrity of the Validation and Test sets. In a real-world scenario, if a user inputs a highly unusual 250-turn game, the model cannot simply 'drop' the user's request; it should provide a prediction. By capping, we train the model to treat extreme outliers as simply 'very high' values without allowing them to distort the mathematical weights of the Neural Network.

### Encoding Categorical Variables

The encoding method depends on the nature of the categorical variables:
- rated: boolean (True-False)
- victory status, it gives away the result if draw - dropped (although the result in a win would be a good feature to train upon.)
- winner is the target (3 labels)
- increment_code : high cardinality, a lot of game formats
- opening_eco: high cardinality 
- opening_name: high cardinality

#### One-hot encoding

Ideal for the target which has values "white", "black" and "draw".

Because it is in the target, the output from the encoding process will give as an array of 0s 1s and 2s.

#### Label encoding

Ideal for the rated, which is True or False, so it will be 1 or 0.

#### Target endoding

Target encoding basically calculates a possibility based on the target, so we will have 3 new columns for each categorical features with high cardinality. So after the encoding we have 16 columns.

Problems to overcome with the Targer encoding: overfitting, never seen categories in new datasets. 

We use the `TargetEncoder` class of the `scikit-learn` library. A very good description of how the class works can be found [here](https://towardsdatascience.com/encoding-categorical-variables-a-deep-dive-into-target-encoding-2862217c2753/). `scikit-learn` and `feature-engine` can automatically detect the optimal smoothing parameter using empirical Bayes variance estimates. To avoid data leakage we use `shuffle` and thus we set the `random_state` parameter to 42, as specified in the exercise.

_Note 1_ : This could be a good time to break the feature "*increament_code*" in two different numerical features "*base_time_mins*" and "*added_times_secs*"; this way any new time format that comes with a new dataset will be taken care by the model.

_Note 2_: the features "*opening_name*" and "*opening_eco*" have conceptually high correlation; e.g. the value D10 of the opening eco always related to the opening name "Slav defence" + some variation. However, there might be a case where opening name "Slav defence" + some other variation, does not relate to D10. (ecos and names are just for example). It might seem that we could drop one of those features as it does not give much more info. This is a coming topic on the PCA analysis.


## 4. Feature Engineering

The two features that can encode domain knowledge is the duration of a game and the difference between rated player.

- "last_move_at" - "created_at": The format is in Unix timestamps we will convert to Date/Time and then to minutes
- "white_rating" - "black_rating": is the differnce between the capabilities between the playes; a higher rated player will most probably win, and this is much more visible when the difference is very high. It is unlikely that a 500 player will win a 1500 player. However, it is not unlikely for a player of 2200 to win a player of 2300.

_Note_: due to the early format of the output in the dataset, in some rows, the time features "created_at" and "last_move_at" have the same value, and thus we get a "game_duration_mins" = 0. To that end we create a function to fix this 0 and set it to the median of the durations.

_Note 2_: however the duration can be more precisely "guessed" at the preprocessing by looking at the feature "increment_code", because a "5+2" game would have a duration of 5-8 minutes (median) and a "10+5" game probably a duration of 11-15. So it would be advantageous to have a median grouped by the feature "increment_code" and then drop the feature. This step takes place before encoding the categorical features. This step can be done before the encoding in our case.


## 5. Feature Scaling

The basic idea of the scaling is normalizing ratings (which range from 800 to 2700), and the date related features "created_at" and "last_move_at" so the Neural Network doesn't get overwhelmed by large numbers.

We select the `StandardScaler` to scale the numerical features, because we already treated our extreme outliers using the IQR capping method (Winsorization) in the previous step. Thus, a `RobustScaler` is no longer necessary. The `StandardScaler` transforms the features to have a mean of 0 and a standard deviation of 1 using the formula $z = \frac{x - \mu}{\sigma}$. This is the preferred method for Neural Networks, ensuring that massively scaled columns like created_at or white_rating do not artificially dominate smaller scaled columns like "turns" or "opening_ply".

## 6. PCA Insights

### Scree plot

<img src="./images/scree_plot.png" style="width:700px"/>

The scree plot reveals that the dataset's variance is relatively distributed across multiple dimensions. It requires 6 principal components to capture 90% of the total cumulative variance. This indicates that a chess game is a complex, multi-dimensional event; the data cannot be aggressively compressed into just 2 or 3 features without losing highly significant information.

### PCA loadings

Number of components needed to explain 90% of variance: 6

Top Features driving Principal Component 1 (PC1)

| feature         | Weight |
| --------------- | ------ |
| last_move_at    |0.555886|
| created_at      |0.555885|
| black_rating    |0.391142|
| white_rating    |0.362634|
| opening_ply     |0.253568|

Top Features driving Principal Component 2 (PC2)
| feature         | Weight |
| --------------- | ------ |
| white_rating    | 0.549390|
| black_rating    | 0.458896|
| created_at      | 0.427984|
| last_move_at    | 0.427976|
| opening_ply     | 0.288961|

By inspecting the component weights (loadings), we can interpret the real-world meaning of the newly created synthetic axes:
- PC1 is heavily dominated by the time related features ("last_move_at", "created_at") and the players' overall skill levels (black_rating, white_rating).
- PC2 is driven by the exact same set of features, but with a slightly heavier emphasis on the ratings over the timestamps.
- The fact that the highly correlated timestamps dominated the primary axes of variance validates our earlier feature engineering decision to extract game_duration_mins. In a strict dimensionality reduction scenario, the raw timestamps would likely be dropped to prevent them from overwhelming the principal components.


### PC1 - PC2 scatter plot

<img src="./images/scatter_plot.png" style="height:500px"/>

The data was projected onto a 2D scatter plot using PC1 and PC2, colored by the target class (Winner). The plot displays a dense, heavily overlapping cloud of data points with no distinct clusters or linear boundaries between the classes. Because PC1 and PC2 primarily represent the duration of the game and Skill Level (Rating) of the players which are factors that dictate the environment of the game rather than the outcome, it is mathematically logical that they do not perfectly separate the winner. This confirms that predicting the outcome of a chess game is a highly non-linear classification problem that will require an algorithm capable of learning complex, higher-dimensional interactions (such as a Random Forest or Neural Network).


## 7. Model Comparison

## 8. Best Model Designation

## 9. Installation and Execution