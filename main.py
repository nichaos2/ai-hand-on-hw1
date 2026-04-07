import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing import engineer_domain_features

app = FastAPI()

# Load artifacts
best_model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")


class ChessMatchInput(BaseModel):
    # Booleans
    rated: bool = Field(..., description="True if the game is rated, False otherwise")

    # Timestamps (usually floats or ints in this dataset)
    created_at: float = Field(..., description="Start time of the match")
    last_move_at: float = Field(..., description="End time of the match")

    # Numerical
    turns: int = Field(..., description="Total number of moves/turns in the game")
    white_rating: int = Field(..., description="Elo rating of the White player")
    black_rating: int = Field(..., description="Elo rating of the Black player")
    opening_ply: int = Field(..., description="Number of moves in the opening phase")

    # Categorical / Text
    increment_code: str = Field(..., description="Time control format, e.g., '15+2'")
    opening_eco: str = Field(..., description="Standardized ECO code, e.g., 'D04'")
    opening_name: str = Field(..., description="Name of the opening played")

    model_config = {
        # This creates the default example in your Swagger UI documentation
        "json_schema_extra": {
            "examples": [
                {
                    "rated": True,
                    "created_at": 1504210000000.0,
                    "last_move_at": 1504210000000.0,
                    "turns": 13,
                    "increment_code": "15+2",
                    "white_rating": 1500,
                    "black_rating": 1191,
                    "opening_eco": "D10",
                    "opening_name": "Slav Defense",
                    "opening_ply": 5,
                }
            ]
        }
    }


@app.post("/predict")
def predict_outcome(match_data: ChessMatchInput):
    try:
        # Convert JSON to a Pandas DataFrame
        input_dict = match_data.model_dump()
        input_df = pd.DataFrame([input_dict])
        print("df  from body")
        print(input_df)

        input_df_, _, _ = engineer_domain_features(input_df, None, None)
        print("df from adding features")
        print(input_df_)

        # encode_categories
        ## binary
        input_df_["rated"] = input_df_["rated"].astype(int)
        high_cardinality_features = ["increment_code", "opening_eco", "opening_name"]

        # Transform the text strings into their historical numerical averages
        # we do not pass 'y' here. The encoder uses its saved memory.
        new_col_names = target_encoder.get_feature_names_out(high_cardinality_features)
        encoded_data = target_encoder.transform(input_df_[high_cardinality_features])
        encoded_df = pd.DataFrame(
            encoded_data, columns=new_col_names, index=input_df_.index
        )
        input_df_ = input_df_.drop(columns=high_cardinality_features)
        input_df_ = pd.concat([input_df_, encoded_df], axis=1)

        # --- SCALING ---
        # Ensure input_df_ columns exactly match what the scaler expects
        cols_to_scale = [
            "turns",
            "white_rating",
            "black_rating",
            "opening_ply",
            "created_at",
            "last_move_at",
            "rating_advantage",
            "game_duration_mins",
        ]
        input_df_[cols_to_scale] = scaler.transform(input_df_[cols_to_scale])

        # XGBoost is strict and need columns in specific order
        expected_columns = [
            "rated",
            "created_at",
            "last_move_at",
            "turns",
            "white_rating",
            "black_rating",
            "opening_ply",
            "rating_advantage",
            "game_duration_mins",
            "increment_code_0",
            "increment_code_1",
            "increment_code_2",
            "opening_eco_0",
            "opening_eco_1",
            "opening_eco_2",
            "opening_name_0",
            "opening_name_1",
            "opening_name_2",
        ]

        # --- PREDICTION ---
        input_df_ = input_df_[expected_columns]
        prediction_num = best_model.predict(input_df_)[0]
        class_mapping = {0: "Black Wins", 1: "Draw", 2: "White Wins"}
        return {
            "status": "success",
            "predicted_outcome": class_mapping.get(int(prediction_num), "Unknown"),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
