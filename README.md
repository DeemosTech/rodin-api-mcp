# Rodin API MCP

## Project Overview

Rodin API MCP is a service based on the Model Context Protocol (MCP) that exposes Rodin's API to AI models. This service aims to simplify the interaction between AI models and the Rodin API.

## Features

- Provides an MCP interface for Rodin API
- Supports integration with various AI models
- Offers efficient data transmission and processing capabilities

## Dependencies Installation

For installing `uv`, please refer to the official installation guide: [uv Installation Guide](https://docs.astral.sh/uv/getting-started/installation/)

## Configuration for Claude Desktop

To configure Claude Desktop to support MCP, follow these steps:

1. Go to Claude > Settings > Developer > Edit Config > `claude_desktop_config.json` and include the following:

   ```json
   {
       "mcpServers": {
           "blender": {
               "command": "uvx",
               "args": [
                   "run",
                   "github.com/yourusername/rodin-api-mcp"
               ],
               "env": {
                    "RODIN_API_KEY": <PROVIDE_YOUR_API_KEY_HERE>
               }
           }
       }
   }
   ```

2. If Claude Deskop is opened, quit it and restart Claude Desktop.
