import httpx
import asyncio
import torch.nn as nn
from models.vgg import vggStudent
from serialization import *
from utils.training import average_weights
import os

async def make_rpc_call( 
        client_id: int,
        student_model: nn.Module,
    ) -> None | nn.Module:

    url = f"http://{os.environ['ip']}:{int(os.environ['base_port']) + client_id + 1}/rpc"

    async with httpx.AsyncClient() as client:
        # serialize and send to the client
        data_b64 = serialize_model(student_model)
        response = await client.post(url, json={"student_model": data_b64})

        if response.status_code != 200:
            return None
        
        # get the updated student model weights
        response_data = await response.json()
        
        if "student_model" not in response_data:
            return None
        
        # deserialize and return the student model; return None if there is problem
        student_model_weights = deserialize_model(response_data.get("student_model"), os.environ['device'])
        return student_model_weights
    
async def main() -> None:
    student_model = vggStudent()
    epochs = int(os.environ['epochs'])
    num_clients = int(os.environ['num_clients'])

    for _ in range(epochs):
        tasks = []
        for client_id in range(num_clients):
            task = asyncio.create_task(make_rpc_call(client_id, student_model))
            tasks.append(task)
        
        # wait on all of the calls
        updated_student_models = await asyncio.gather(*tasks)

        # get rid of any tasks that failed (ie. returned None)
        valid_student_models = [w for w in updated_student_models if w != None]

        # although rare, its possible for all the calls to fail; in this
        # case, just move onto the next epoch.
        if len(valid_student_models) == 0:
            continue

        # average the weights and load into the student model
        avg_student_model = average_weights(valid_student_models)
        student_model.load_state_dict(avg_student_model)

if __name__ == '__main__':
    # training
    asyncio.run(main())