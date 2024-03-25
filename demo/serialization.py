import base64
import io
import torch

def serialize_to_b64(model, args):
    buffer = io.BytesIO()
    torch.save({
        "student_model": model.state_dict(),
        **args,
    }, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()

def deserialize_model(data_b64, device):
    decoded_data = base64.b64decode(data_b64)
    buffer = io.BytesIO(decoded_data)
    loaded_data = torch.load(buffer, map_location=torch.device(device))
    model = loaded_data.pop("student_model")
    args = loaded_data

    return model, args