from fastapi import FastAPI, HTTPException, Form
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import uvicorn
from input_options import options  # options is a dict containing allowed values for each field

# Adjust sys.path so we can import our ANN architecture from training.ann
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from training.ann import ANN

# -----------------------------------------------------
# HELPER FUNCTION: One-hot encoder
# -----------------------------------------------------
def one_hot_encode(value: str, categories: list) -> list:
    """Return a one-hot encoded vector for the given value based on the provided categories."""
    vector = [0] * len(categories)
    try:
        index = categories.index(value)
        vector[index] = 1
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid value '{value}'. Expected one of {categories}."
        )
    return vector

# -----------------------------------------------------
# Model configuration and loading.
# The input vector consists of:
# - One-hot encoded categorical features: Pool, Project, Category, Chain, Outlook
# - 5 numeric features: TVL, APY, APY Base, APY Mean 30d, APY Reward
# -----------------------------------------------------
INPUT_DIM = 622  # Must match your training configuration
HIDDEN_DIM = 64  # Must match the training configuration
OUTPUT_DIM = 3   # For example, 3 classes: conservative, balanced, aggressive.

model = ANN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.load_state_dict(torch.load("../model/risk-assess-model.pth", map_location=torch.device("cpu")))
model.eval()  # Set the model to evaluation mode

# -----------------------------------------------------
# Create the FastAPI instance.
# -----------------------------------------------------
app = FastAPI()

@app.post("/predict")
async def predict(
    pool: str = Form(..., description=f"Allowed options: {', '.join(options['Pool'])}"),
    project: str = Form(..., description=f"Allowed options: {', '.join(options['Project'])}"),
    category: str = Form(..., description=f"Allowed options: {', '.join(options['Category'])}"),
    chain: str = Form(..., description=f"Allowed options: {', '.join(options['Chain'])}"),
    tvl: float = Form(..., description="Numeric value for TVL"),
    apy: float = Form(..., description="Numeric value for APY"),
    apy_base: float = Form(..., description="Numeric value for APY Base"),
    apy_mean_30d: float = Form(..., description="Numeric value for APY Mean 30d"),
    apy_reward: float = Form(..., description="Numeric value for APY Reward"),
    outlook: str = Form(..., description=f"Allowed options: {', '.join(options['Outlook'])}")
):
    # One-hot encode the categorical features using the allowed options
    pool_vec = one_hot_encode(pool, options["Pool"])
    project_vec = one_hot_encode(project, options["Project"])
    category_vec = one_hot_encode(category, options["Category"])
    chain_vec = one_hot_encode(chain, options["Chain"])
    outlook_vec = one_hot_encode(outlook, options["Outlook"])
    
    # Collect numeric features.
    numeric_features = [
        tvl,
        apy,
        apy_base,
        apy_mean_30d,
        apy_reward
    ]
    
    # Combine all features into one input vector.
    input_vector = np.array(
        pool_vec + project_vec + category_vec + chain_vec + outlook_vec + numeric_features,
        dtype=np.float32
    )
    
    # Check that the input vector matches the expected dimension.
    if input_vector.shape[0] != INPUT_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Input vector has incorrect dimensions. Expected {INPUT_DIM} features, got {input_vector.shape[0]}."
        )
    
    # Reshape to (1, INPUT_DIM) for a single sample.
    data_tensor = torch.FloatTensor(input_vector).reshape(1, -1)
    
    # Run the model to obtain predictions.
    with torch.no_grad():
        outputs = model(data_tensor)
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
        predicted_class = int(np.argmax(probabilities)) + 1  # Adjust if necessary
    
    risk_levels = ["High", "Medium", "Low"]
    risk = risk_levels[predicted_class - 1]
    
    return {
        "predicted_class": predicted_class,
        "risk_level": risk,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
