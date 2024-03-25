import base64
import io
import torch

def serialize_model(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()

def deserialize_model(data_b64, device):
    decoded_data = base64.b64decode(data_b64)
    buffer = io.BytesIO(decoded_data)
    model = torch.load(buffer, map_location=torch.device(device))
    return model