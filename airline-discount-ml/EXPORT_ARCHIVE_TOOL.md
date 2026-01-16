# Export Archive Tool

## Overview

The `export_archive` tool has been added to the MCP Synth server. It allows you to create zip archives of synthetic data output files.

## Features

- **Flexible file patterns**: Specify which file types to include (e.g., `*.json`, `*.csv`, `*.txt`)
- **Custom archive naming**: Choose your own zip file name
- **Automatic logging**: Creates a log file with archive details
- **Security**: Restricts access to `data/` and `synth_models/` directories only
- **Size reporting**: Shows archive size in human-readable format (KB/MB)

## Usage

### Via GitHub Copilot (Recommended)

Once the MCP server is running and connected to VS Code, you can use natural language:

```
#mcp_synth Create a zip archive of all JSON files in data/synthetic_output
```

```
#mcp_synth Export synthetic data to backup.zip including json and txt files
```

### Via MCP JSON-RPC

```bash
curl -X POST http://127.0.0.1:8010/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "export_archive",
      "arguments": {
        "source_dir": "data/synthetic_output",
        "archive_name": "my_export.zip",
        "include_patterns": ["*.json", "*.csv", "*.txt"]
      }
    }
  }'
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_dir` | string | `"data/synthetic_output"` | Directory containing files to archive |
| `archive_name` | string | `"synthetic_data_export.zip"` | Name of the output zip file |
| `include_patterns` | array | `["*.json", "*.csv", "*.txt"]` | File patterns to include |
| `log_file` | string | `""` | Optional custom path for log file |

## Response

The tool returns:

- **success**: Boolean indicating if archive was created
- **message**: Summary message
- **archive_path**: Full path to the created zip file
- **files_archived**: List of files included in the archive
- **archive_size_bytes**: Size of the archive in bytes

## Examples

### Example 1: Archive all JSON files (default)

```json
{
  "source_dir": "data/synthetic_output",
  "archive_name": "data_backup.zip",
  "include_patterns": ["*.json"]
}
```

**Output:**
```
📦 Archive created successfully!
📁 Archive: /path/to/data/data_backup.zip
📊 Size: 0.56 KB (571 bytes)
📄 Files archived (1):
  - generated_data.json
```

### Example 2: Archive multiple file types

```json
{
  "source_dir": "data/synthetic_output",
  "archive_name": "full_export.zip",
  "include_patterns": ["*.json", "*.txt", "*.csv"]
}
```

**Output:**
```
📦 Archive created successfully!
📁 Archive: /path/to/data/full_export.zip
📊 Size: 0.83 KB (847 bytes)
📄 Files archived (2):
  - generated_data.json
  - model_inspection.txt
```

### Example 3: Using with custom log file

```json
{
  "source_dir": "data/synthetic_output",
  "archive_name": "export_2024.zip",
  "include_patterns": ["*.json"],
  "log_file": "data/logs/export_2024_log.txt"
}
```

## Testing

### 1. Start the MCP server

```bash
cd airline-discount-ml
source venv/bin/activate
uvicorn src.mcp_synth.server:app --host 127.0.0.1 --port 8010 --reload
```

### 2. Test the tool

```bash
# List all tools (should include export_archive)
curl -X POST http://127.0.0.1:8010/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | jq '.result.tools[].name'

# Create a test archive
curl -X POST http://127.0.0.1:8010/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "export_archive",
      "arguments": {
        "source_dir": "data/synthetic_output",
        "archive_name": "test_export.zip",
        "include_patterns": ["*.json"]
      }
    }
  }' | jq .

# Verify the archive
ls -lh data/test_export.zip
unzip -l data/test_export.zip
```

## Integration with VS Code

The tool is automatically available in GitHub Copilot when:

1. The MCP server is running on `http://127.0.0.1:8010`
2. The `.vscode/mcp.json` file includes the server configuration
3. VS Code has been reloaded to pick up the MCP configuration

## Security Features

- **Path traversal prevention**: Only allows access to `data/` and `synth_models/` directories
- **Input validation**: Validates all parameters using Pydantic models
- **Error handling**: Provides clear error messages for invalid inputs

## Error Handling

The tool handles various error scenarios:

- **Source directory not found**: Returns 404 error
- **No matching files**: Returns 404 error with patterns used
- **Permission denied**: Returns 403 error for restricted paths
- **Archive creation failure**: Returns 500 error with details

## Files Modified

- `/airline-discount-ml/src/mcp_synth/server.py`: Added export_archive functionality
  - `ExportArchiveRequest` class (lines 170-185)
  - `ExportArchiveResponse` class (lines 186-192)
  - `export_archive()` function (lines 194-255)
  - `EXPORT_ARCHIVE_SCHEMA` constant (lines 333-354)
  - Tool registration in `tools/list` (line 420-422)
  - Tool handler in `tools/call` (lines 505-536)

## Implementation Details

The tool uses Python's built-in `zipfile` module with `ZIP_DEFLATED` compression for optimal file sizes. Files are stored with their relative paths from the source directory to maintain structure.
