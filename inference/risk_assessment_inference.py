from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import uvicorn

# Define the request model using Pydantic
class InputData(BaseModel):
    # Example: expecting a list of floats for a single sample
    data: list

# Define your model architecture (must match your exported model)
class SimpleANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Set model dimensions (adjust these based on your training)
INPUT_DIM = 622     # number of features
HIDDEN_DIM = 64     # same as used during training
OUTPUT_DIM = 3      # number of classes (e.g., 0,1,2)

# Instantiate and load the model
model = SimpleANN(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
model.load_state_dict(torch.load("../model/risk-assess-model.pth", map_location=torch.device("cpu")))
model.eval()  # set the model to evaluation mode

# Create FastAPI instance
app = FastAPI()

@app.post("/predict")
async def predict(input_data: InputData):
    # Convert the input list into a numpy array and reshape it to a single sample (1, INPUT_DIM)
    data = np.array(input_data.data).reshape(1, -1)
    data_tensor = torch.FloatTensor(data)
    
    with torch.no_grad():
        outputs = model(data_tensor)
        # Get probabilities using softmax
        probabilities = torch.softmax(outputs, dim=1).numpy()[0]
        predicted_class = int(np.argmax(probabilities))
    
    return {"predicted_class": predicted_class, "probabilities": probabilities.tolist()}

# Run the API with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
