import base64
import io
import torch

def serialize_model(model: torch.nn.Module) -> str:
    '''Attempts to serialize a torch model; returns a base64 encoding
    of the model or an empty string if serialization fails.'''
    try:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()
    except:
        return ""

def deserialize_model(data_b64: str, device: str) -> None | torch.nn.Module:
    '''Attempts to deserialize a base64 encoding of a model into a torch model; returns
    the model or None if deserialization fails.'''
    try:
        decoded_data = base64.b64decode(data_b64)
        buffer = io.BytesIO(decoded_data)
        model_state_dict = torch.load(buffer, map_location=torch.device(device))
        return model_state_dict
    except:
        return None