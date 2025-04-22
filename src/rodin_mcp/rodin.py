import os
from typing import Literal, Optional
from mcp.server.fastmcp import FastMCP, Image as MCPImage
import httpx
from pydantic import Field
from .custom_types import DownloadRequestParameters, RodinParameters
from .exceptions import RodinAPIException
import asyncio

server = FastMCP('rodin-mcp')

API_KEY = os.getenv(
    'RODIN_API_KEY', 'g4FQt6KvTGXre2i98GopyXdTdddOnFTBcQF5Hfiu7z5asMm5bqoy7f3L1gFU4fZ0'
)

def error_handle_response(response):
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise RodinAPIException(
            f"HTTP error occurred: {e}",
            response=e.response
        ) from e
    
    except httpx.RequestError as e:
        raise RodinAPIException(
            f"Request error occurred: {e}",
            response=e.response if hasattr(e, 'response') else None
        ) from e
    
    except httpx.TimeoutException as e:
        raise RodinAPIException(
            f"Timeout occurred: {e}",
            response=e.response if hasattr(e, 'response') else None
        ) from e

@server.tool()
async def generate_3d_model(parameters: RodinParameters) -> dict | str:
    """
    This tool will call the Rodin API to generate a mesh and textures for the given images and prompt.

    [Notes]
        - When generating an model prompt, try to be concise, and do not describe the model by saying what this thing is not.
        - Try to ask for a reference image.
            - If user does not have the image. It's OK, but remember to provide the prompt.
            - If user did give you image(s), do the following:
                - If you decided to upload image(s), ALWAYS ask for the path from the user.
                - All provided image_paths must be actual paths, given by the user.
                - Guide the user to find the absolute image path if they don't know how to.
                - The prompt parameter will not help much if image(s) are given, so omit the prompt parameter.
        - Confirm the model format and other noticeable parameters with the user before generating.

    [Returns]
        If generation task created, the tool will return an object containing important information to retrieve the result after the generation finished:
        - message: A template message.
        - prompt The final prompt used by Rodin. The prompt is refined by Rodin to meet specific needs.
        - submit_time: A timestamp of Rodin's server receiving the request.
        - uuid: The task's UUID. Important for retrieving the results.
        - jobs: An object, containing information of the jobs of the task.
            - uuids: The UUIDs of the jobs
            - subscription_key: A JWT, important for retrieving the result.

    """
    url = "https://hyperhuman.deemos.com/api/v2/rodin"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }
    files = parameters.convert_to_files()
    async with httpx.AsyncClient(headers=headers) as client:
        response = await client.post(url, files=files)
        error_handle_response(response)

        try:
            json_data = response.json()
            return json_data
        except:
            return response.text

async def download_file(client: httpx.AsyncClient, url: str, file_path: str, retries: int = 3, delay: float = 1.0):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            response = await client.get(url)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)
            return
        except Exception as e:
            last_exc = e
            if attempt < retries:
                await asyncio.sleep(delay)
            else:
                raise last_exc

@server.tool()
async def try_download_result(
    parameters: DownloadRequestParameters
) -> str | tuple[str, Optional[MCPImage], str]:
    """
    This tool will try to download the generated assets if the Rodin completed the generation task.

    [Notes]
        - Always make sure an absolute path is given bu the user before calling this tool.
        - The tool expects a directory for the download_to_path parameter.
            - The tool will download files to {download_to_path}/{uuid}/ directory
            - So when giving example, don't include the filename
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }
    async with httpx.AsyncClient(headers=headers) as client:
        # Polling
        # for _ in range(parameters.retry_count):
        for _ in range(3):
            response = await client.post(
                "https://hyperhuman.deemos.com/api/v2/status",
                data={
                    "subscription_key": parameters.subscription_key
                }
            )

            error_handle_response(response)
            json_data = response.json()
            try:
                status_list = [i['status'] for i in json_data["jobs"]]
            except:
                raise RodinAPIException(f"Unexpected response: {json_data}")
            if "Failed" in status_list or "Canceled" in status_list:
                raise RodinAPIException("Generation task failed!", response)
            
            if all(i == "Done" for i in status_list):
                break
            await asyncio.sleep(1.5)
            # await asyncio.sleep(parameters.poll_interval)
        else:
            return "Task not finished yet. Try again later."
        
        # Get asset links
        response = await client.post(
            "https://hyperhuman.deemos.com/api/v2/download",
            data={
                "task_uuid": parameters.uuid
            }
        )
        error_handle_response(response)
        json_data = response.json()

    # Actual download
    asset_dict = {
        i['name']: i['url']
        for i in json_data['list']
    } # Assuming every file has unique name

    os.makedirs(parameters.target_directory_path, exist_ok=True)
    async with httpx.AsyncClient() as client:
        tasks = [
            download_file(client, url, os.path.join(parameters.target_directory_path, file_name))
            for file_name, url in asset_dict.items()
        ]
        await asyncio.gather(*tasks)

    returning_image = None
    if "preview.webp" in asset_dict:
        returning_image = MCPImage(path=os.path.join(parameters.target_directory_path, "preview.webp"))

    return (
        returning_image,
        "The image above is an preview image of the generated model, without texture and material.\n"
        f"The assets are downloaded to {parameters.target_directory_path}"
    )

def main(transport: Literal["stdio", "sse"] = "stdio"):
    """Run the MCP server"""
    server.run(transport)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start MCP with specified transport")
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: 'stdio' or 'sse' (default: stdio)"
    )
    args = parser.parse_args()

    transport = args.transport
    main(transport)
