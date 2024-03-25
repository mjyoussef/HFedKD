import base64
import io
import torch

def serialize_model(model: torch.nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()

def deserialize_model(data_b64: str, device: str) -> torch.nn.Module:
    decoded_data = base64.b64decode(data_b64)
    buffer = io.BytesIO(decoded_data)
    model_state_dict = torch.load(buffer, map_location=torch.device(device))
    return model_state_dict