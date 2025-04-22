from typing import Literal
from rodin_mcp.rodin import main as server_main

def main(transport: Literal["stdio", "sse"] = "stdio"):
    """Entry point for the rodin_mcp package"""
    server_main(transport)

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
