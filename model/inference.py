import numpy as np
import json

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model_fn(model_dir):
    w = np.load(f"{model_dir}/logreg_weights.npy")
    b = np.load(f"{model_dir}/logreg_bias.npy")[0]
    return w, b

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return np.array(data["features"])
    else:
        raise ValueError("Unsupported content type")

def predict_fn(input_data, model):
    w, b = model
    prob = sigmoid(input_data @ w + b)
    return {"heart_disease_probability": float(prob)}

def output_fn(prediction, response_content_type):
    return json.dumps(prediction)