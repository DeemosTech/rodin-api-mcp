from typing import Any, Optional
import httpx


class RodinAPIException(Exception):
    def __init__(self, message: str, response: Optional[httpx.Response] = None, *args: Any):
        self.response = response
        super().__init__(message, *args)
    
    def __str__(self):
        if self.response:
            return f"{self.args[0]}\nResponse Status Code: {self.response.status_code}\nResponse Body: {self.response.text}"
        return self.args[0]