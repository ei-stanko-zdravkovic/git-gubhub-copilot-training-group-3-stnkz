from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from . import __version__

app = FastAPI(title="MCP Synth Server", version=__version__)

# Add CORS middleware for VS Code MCP client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Health & version (unchanged)
# -------------------------

class HealthResponse(BaseModel):
    status: str

@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(status="ok")

class VersionResponse(BaseModel):
    version: str

@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(version=__version__)

# -------------------------
# Tool: synth_generate (existing logic)
# -------------------------

class GenerateRequest(BaseModel):
    """Request to generate synthetic data via Synth CLI."""
    model_dir: str = Field(default="synth_models/airline_data", description="Path to Synth schemas")
    out_dir: str = Field(default="data/synthetic_output", description="Output directory")
    size: int = Field(default=5, ge=1, le=10000, description="Records per collection")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    log_file: str = Field(default="", description="Optional path to save formatted log output")

class GenerateResponse(BaseModel):
    """Response from data generation."""
    success: bool
    message: str
    files_created: List[str]
    data: Dict[str, Any] = Field(default_factory=dict, description="Generated data preview")
    command: str = Field(description="Synth CLI command that was executed")

@app.post("/synth_generate", response_model=GenerateResponse)
def synth_generate(req: GenerateRequest) -> GenerateResponse:
    """Generate synthetic data using Synth CLI."""
    try:
        cmd = [
            "synth", "generate",
            req.model_dir,
            "--size", str(req.size),
            "--seed", str(req.seed),
        ]
        cmd_str = " ".join(cmd)
        # Run Synth
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Persist output as a single JSON file
        os.makedirs(req.out_dir, exist_ok=True)
        out_file = os.path.join(req.out_dir, "generated_data.json")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        # Parse the generated data to include in response
        generated_data = json.loads(result.stdout)

        return GenerateResponse(
            success=True,
            message=f"Generated {req.size} records per collection",
            files_created=[out_file],
            data=generated_data,
            command=cmd_str,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Synth command failed: {e.stderr}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e


class InspectModelRequest(BaseModel):
    model_dir: str = Field(default="synth_models/airline_data", description="Path to Synth schemas")
    log_file: str = Field(default="", description="Optional path to save inspection output")

class InspectModelResponse(BaseModel):
    model_dir: str
    files: List[str]

def synth_inspect_model(req: InspectModelRequest) -> InspectModelResponse:
    p = Path(req.model_dir)
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail=f"Model dir not found: {p}")
    files = sorted([str(x) for x in p.glob("**/*") if x.is_file()])
    return InspectModelResponse(model_dir=str(p), files=files)

class PreviewHeadRequest(BaseModel):
    path: str = Field(
        default="data/synthetic_output/generated_data.json",
        description="Path to JSON/NDJSON/CSV file"
    )
    n: int = Field(default=10, ge=1, le=200, description="Number of rows to preview")
    log_file: str = Field(default="", description="Optional path to save preview output")

class PreviewHeadResponse(BaseModel):
    path: str
    rows: List[dict]

def preview_table_head(req: PreviewHeadRequest) -> PreviewHeadResponse:
    p = Path(req.path)
    
    # Security: prevent directory traversal and restrict to safe paths
    try:
        resolved = p.resolve()
        # Allow only data/ and synth_models/ directories (relative to project root)
        allowed_prefixes = [
            Path("data").resolve(),
            Path("synth_models").resolve(),
        ]
        if not any(str(resolved).startswith(str(prefix)) for prefix in allowed_prefixes):
            raise HTTPException(
                status_code=403, 
                detail=f"Access denied: path must be under data/ or synth_models/"
            )
    except (OSError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e
    
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {resolved}")
    rows: List[dict] = []

    # Simple preview for JSON arrays or NDJSON; CSV fallback.
    try:
        text = resolved.read_text(encoding="utf-8")
        text_stripped = text.strip()
        if text_stripped.startswith("["):
            data = json.loads(text_stripped)
            if isinstance(data, list):
                rows = data[: req.n]
            else:
                rows = [data]
        elif "\n" in text_stripped and text_stripped.startswith("{"):
            # NDJSON
            lines = [ln for ln in text.splitlines() if ln.strip()]
            for ln in lines[: req.n]:
                rows.append(json.loads(ln))
        elif resolved.suffix.lower() == ".csv":
            import csv
            with resolved.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for i, rec in enumerate(reader):
                    if i >= req.n:
                        break
                    rows.append(rec)
        else:
            # Fallback: return first N lines as opaque text
            rows = [{"line": ln} for ln in text.splitlines()[: req.n]]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not preview file: {e}") from e

    return PreviewHeadResponse(path=str(resolved), rows=rows)

class SynthStatsRequest(BaseModel):
    """Request to compute statistics on a data file."""
    path: str = Field(
        default="data/synthetic_output/generated_data.json",
        description="Path to JSON/NDJSON/CSV file"
    )
    log_file: str = Field(default="", description="Optional path to save stats output")

class SynthStatsResponse(BaseModel):
    """Response with data statistics."""
    path: str
    total_records: int
    collections: Dict[str, Any] = Field(default_factory=dict, description="Statistics per collection")

def synth_stats(req: SynthStatsRequest) -> SynthStatsResponse:
    """Compute statistics on generated data file."""
    p = Path(req.path)
    
    # Security: prevent directory traversal and restrict to safe paths
    try:
        resolved = p.resolve()
        allowed_prefixes = [
            Path("data").resolve(),
            Path("synth_models").resolve(),
        ]
        if not any(str(resolved).startswith(str(prefix)) for prefix in allowed_prefixes):
            raise HTTPException(
                status_code=403, 
                detail=f"Access denied: path must be under data/ or synth_models/"
            )
    except (OSError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e
    
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {resolved}")
    
    try:
        text = resolved.read_text(encoding="utf-8")
        data = json.loads(text.strip())
        
        collections_stats = {}
        total_records = 0
        
        if isinstance(data, dict):
            # Multi-collection format
            for collection_name, records in data.items():
                if isinstance(records, list):
                    count = len(records)
                    total_records += count
                    
                    field_stats = {}
                    if records:
                        all_keys = set()
                        for record in records:
                            if isinstance(record, dict):
                                all_keys.update(record.keys())
                        
                        for key in all_keys:
                            values = [r.get(key) for r in records if isinstance(r, dict) and key in r]
                            non_null_values = [v for v in values if v is not None]
                            
                            field_stat = {
                                "count": len(non_null_values),
                                "null_count": len(values) - len(non_null_values),
                                "types": list(set(type(v).__name__ for v in non_null_values)),
                            }
                            
                            # Numeric statistics
                            numeric_values = [v for v in non_null_values if isinstance(v, (int, float))]
                            if numeric_values:
                                field_stat["min"] = min(numeric_values)
                                field_stat["max"] = max(numeric_values)
                                field_stat["mean"] = sum(numeric_values) / len(numeric_values)
                            
                            # String statistics
                            string_values = [v for v in non_null_values if isinstance(v, str)]
                            if string_values:
                                field_stat["unique_count"] = len(set(string_values))
                                field_stat["sample_values"] = list(set(string_values))[:5]
                            
                            field_stats[key] = field_stat
                    
                    collections_stats[collection_name] = {
                        "count": count,
                        "fields": field_stats
                    }
        
        return SynthStatsResponse(
            path=str(resolved),
            total_records=total_records,
            collections=collections_stats
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to compute stats: {e}") from e

class ImportSchemaRequest(BaseModel):
    """Request to import database schema into Synth."""
    db_path: str = Field(default="data/airline_discount.db", description="Path to SQLite database")
    output_dir: str = Field(default="synth_models/imported", description="Output directory for Synth schemas")
    log_file: str = Field(default="", description="Optional path to save import log")

class ImportSchemaResponse(BaseModel):
    """Response from schema import."""
    success: bool
    message: str
    schema_files: List[str]

def synth_import_schema(req: ImportSchemaRequest) -> ImportSchemaResponse:
    """Import database schema into Synth format."""
    db_path = Path(req.db_path)
    
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"Database not found: {db_path}")
    
    try:
        # Create output directory
        output_path = Path(req.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run synth import command
        cmd = ["synth", "import", str(db_path), "--to", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # List generated schema files
        schema_files = [str(f.relative_to(output_path)) for f in output_path.glob("*.json")]
        
        return ImportSchemaResponse(
            success=True,
            message=f"Imported schema from {db_path} to {output_path}",
            schema_files=schema_files
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Synth import failed: {e.stderr}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Import failed: {e}") from e

class ValidateFKRequest(BaseModel):
    """Request to validate foreign key constraints."""
    path: str = Field(
        default="data/synthetic_output/generated_data.json",
        description="Path to generated data file"
    )
    log_file: str = Field(default="", description="Optional path to save validation log")

class ValidateFKResponse(BaseModel):
    """Response from FK validation."""
    valid: bool
    message: str
    violations: List[Dict[str, Any]] = Field(default_factory=list)

def validate_fk(req: ValidateFKRequest) -> ValidateFKResponse:
    """Validate foreign key constraints in generated data."""
    p = Path(req.path)
    
    # Security check
    try:
        resolved = p.resolve()
        allowed_prefixes = [Path("data").resolve(), Path("synth_models").resolve()]
        if not any(str(resolved).startswith(str(prefix)) for prefix in allowed_prefixes):
            raise HTTPException(status_code=403, detail="Access denied")
    except (OSError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e
    
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {resolved}")
    
    try:
        with open(resolved) as f:
            data = json.load(f)
        
        violations = []
        
        # Check if data has expected collections
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="Expected multi-collection JSON object")
        
        # Extract IDs from each collection
        passenger_ids = set()
        route_ids = set()
        
        if "passengers" in data:
            passenger_ids = {p.get("id") for p in data["passengers"] if isinstance(p, dict)}
        if "routes" in data:
            route_ids = {r.get("id") for r in data["routes"] if isinstance(r, dict)}
        
        # Validate discounts reference valid passengers and routes
        if "discounts" in data:
            for i, discount in enumerate(data["discounts"]):
                if not isinstance(discount, dict):
                    continue
                
                passenger_id = discount.get("passenger_id")
                route_id = discount.get("route_id")
                
                if passenger_id and passenger_id not in passenger_ids:
                    violations.append({
                        "collection": "discounts",
                        "index": i,
                        "field": "passenger_id",
                        "value": passenger_id,
                        "message": f"passenger_id {passenger_id} not found in passengers"
                    })
                
                if route_id and route_id not in route_ids:
                    violations.append({
                        "collection": "discounts",
                        "index": i,
                        "field": "route_id",
                        "value": route_id,
                        "message": f"route_id {route_id} not found in routes"
                    })
        
        return ValidateFKResponse(
            valid=len(violations) == 0,
            message=f"Found {len(violations)} FK violations" if violations else "All FK constraints valid",
            violations=violations
        )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Validation failed: {e}") from e

class DryRunRequest(BaseModel):
    """Request to preview data generation without creating files."""
    model_dir: str = Field(default="synth_models/airline_data", description="Path to Synth schemas")
    size: int = Field(default=5, ge=1, le=100, description="Preview sample size (max 100)")
    seed: int = Field(default=42, description="Random seed")
    log_file: str = Field(default="", description="Optional path to save preview")

class DryRunResponse(BaseModel):
    """Response from dry run."""
    success: bool
    message: str
    preview: Dict[str, Any] = Field(default_factory=dict)

def synth_dry_run(req: DryRunRequest) -> DryRunResponse:
    """Preview data generation without creating files."""
    try:
        cmd = ["synth", "generate", req.model_dir, "--size", str(req.size), "--seed", str(req.seed)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        preview_data = json.loads(result.stdout)
        
        return DryRunResponse(
            success=True,
            message=f"Preview generated {req.size} records per collection (not saved)",
            preview=preview_data
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Synth command failed: {e.stderr}") from e
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON output: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Dry run failed: {e}") from e

class PolicyCheckRequest(BaseModel):
    """Request to check generation against organizational policies."""
    size: int = Field(description="Requested generation size")
    model_dir: str = Field(default="synth_models/airline_data", description="Model directory")
    max_size: int = Field(default=10000, description="Maximum allowed size")
    log_file: str = Field(default="", description="Optional path to save policy check log")

class PolicyCheckResponse(BaseModel):
    """Response from policy check."""
    allowed: bool
    message: str
    warnings: List[str] = Field(default_factory=list)

def policy_check(req: PolicyCheckRequest) -> PolicyCheckResponse:
    """Enforce organizational limits on data generation."""
    warnings = []
    
    # Check size limit
    if req.size > req.max_size:
        return PolicyCheckResponse(
            allowed=False,
            message=f"Request size {req.size} exceeds maximum {req.max_size}",
            warnings=[]
        )
    
    # Warn if size is large
    if req.size > req.max_size * 0.8:
        warnings.append(f"Large generation request: {req.size} records (80%+ of maximum)")
    
    # Check model directory exists
    model_path = Path(req.model_dir)
    if not model_path.exists():
        return PolicyCheckResponse(
            allowed=False,
            message=f"Model directory not found: {req.model_dir}",
            warnings=warnings
        )
    
    # Check for required schema files
    schema_files = list(model_path.glob("*.json"))
    if not schema_files:
        return PolicyCheckResponse(
            allowed=False,
            message=f"No schema files found in {req.model_dir}",
            warnings=warnings
        )
    
    return PolicyCheckResponse(
        allowed=True,
        message=f"Policy check passed for {req.size} records",
        warnings=warnings
    )

class ExportArchiveRequest(BaseModel):
    """Request to export output files as a zip archive."""
    source_dir: str = Field(
        default="data/synthetic_output",
        description="Directory containing files to archive"
    )
    archive_name: str = Field(
        default="synthetic_data_export.zip",
        description="Name of the output zip file"
    )
    include_patterns: List[str] = Field(
        default=["*.json", "*.csv", "*.txt"],
        description="File patterns to include (e.g., ['*.json', '*.csv'])"
    )
    log_file: str = Field(default="", description="Optional path to save archive log output")

class ExportArchiveResponse(BaseModel):
    """Response from archive export."""
    success: bool
    message: str
    archive_path: str
    files_archived: List[str]
    archive_size_bytes: int

def export_archive(req: ExportArchiveRequest) -> ExportArchiveResponse:
    """Create a zip archive of output files."""
    import zipfile
    from datetime import datetime
    
    source = Path(req.source_dir)
    
    # Security: prevent directory traversal
    try:
        resolved_source = source.resolve()
        allowed_prefixes = [
            Path("data").resolve(),
            Path("synth_models").resolve(),
        ]
        if not any(str(resolved_source).startswith(str(prefix)) for prefix in allowed_prefixes):
            raise HTTPException(
                status_code=403,
                detail="Access denied: source must be under data/ or synth_models/"
            )
    except (OSError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid source path: {e}") from e
    
    if not resolved_source.exists() or not resolved_source.is_dir():
        raise HTTPException(status_code=404, detail=f"Source directory not found: {resolved_source}")
    
    # Create archive in the same parent directory as source
    archive_path = resolved_source.parent / req.archive_name
    
    # Collect files matching patterns
    files_to_archive: List[Path] = []
    for pattern in req.include_patterns:
        files_to_archive.extend(resolved_source.glob(pattern))
    
    # Remove duplicates and sort
    files_to_archive = sorted(set(files_to_archive))
    
    if not files_to_archive:
        raise HTTPException(
            status_code=404,
            detail=f"No files matching patterns {req.include_patterns} found in {resolved_source}"
        )
    
    # Create zip archive
    try:
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_archive:
                if file_path.is_file():
                    # Store with relative path from source directory
                    arcname = file_path.relative_to(resolved_source)
                    zipf.write(file_path, arcname)
        
        archive_size = archive_path.stat().st_size
        files_archived = [str(f.relative_to(resolved_source)) for f in files_to_archive]
        
        return ExportArchiveResponse(
            success=True,
            message=f"Successfully archived {len(files_archived)} files",
            archive_path=str(archive_path),
            files_archived=files_archived,
            archive_size_bytes=archive_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create archive: {e}") from e

# -------------------------
# MCP JSON-RPC endpoint
# -------------------------

# JSON Schemas for tool parameters (derived from Pydantic)
# Following JSON Schema Draft 7 specification for MCP compatibility
SYNTH_GENERATE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "model_dir": {
            "type": "string", 
            "description": "Path to Synth schemas",
            "default": "synth_models/airline_data"
        },
        "out_dir": {
            "type": "string", 
            "description": "Output directory",
            "default": "data/synthetic_output"
        },
        "size": {
            "type": "integer",
            "description": "Records per collection",
            "minimum": 1, 
            "maximum": 10000,
            "default": 1000
        },
        "seed": {
            "type": "integer",
            "description": "Random seed for reproducibility",
            "default": 42
        },
        "log_file": {
            "type": "string", 
            "description": "Optional path to save formatted log output",
            "default": ""
        }
    },
    "required": []
}

SYNTH_INSPECT_MODEL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "model_dir": {
            "type": "string", 
            "description": "Path to Synth schemas"
        },
        "log_file": {
            "type": "string", 
            "description": "Optional path to save inspection output"
        }
    }
}

PREVIEW_TABLE_HEAD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string", 
            "description": "Path to JSON/NDJSON/CSV file (default: data/synthetic_output/generated_data.json)"
        },
        "n": {
            "type": "integer",
            "description": "Number of rows to preview",
            "minimum": 1, 
            "maximum": 200
        },
        "log_file": {
            "type": "string", 
            "description": "Optional path to save preview output"
        }
    },
    "required": []
}

SYNTH_STATS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string", 
            "description": "Path to JSON/NDJSON/CSV file",
            "default": "data/synthetic_output/generated_data.json"
        },
        "log_file": {
            "type": "string", 
            "description": "Optional path to save stats output",
            "default": ""
        }
    },
    "required": []
}

IMPORT_SCHEMA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "db_path": {
            "type": "string",
            "description": "Path to SQLite database",
            "default": "data/airline_discount.db"
        },
        "output_dir": {
            "type": "string",
            "description": "Output directory for Synth schemas",
            "default": "synth_models/imported"
        },
        "log_file": {
            "type": "string",
            "description": "Optional path to save import log",
            "default": ""
        }
    },
    "required": []
}

VALIDATE_FK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to generated data file",
            "default": "data/synthetic_output/generated_data.json"
        },
        "log_file": {
            "type": "string",
            "description": "Optional path to save validation log",
            "default": ""
        }
    },
    "required": []
}

DRY_RUN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "model_dir": {
            "type": "string",
            "description": "Path to Synth schemas",
            "default": "synth_models/airline_data"
        },
        "size": {
            "type": "integer",
            "description": "Preview sample size (max 100)",
            "minimum": 1,
            "maximum": 100,
            "default": 5
        },
        "seed": {
            "type": "integer",
            "description": "Random seed",
            "default": 42
        },
        "log_file": {
            "type": "string",
            "description": "Optional path to save preview",
            "default": ""
        }
    },
    "required": []
}

POLICY_CHECK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "size": {
            "type": "integer",
            "description": "Requested generation size"
        },
        "model_dir": {
            "type": "string",
            "description": "Model directory",
            "default": "synth_models/airline_data"
        },
        "max_size": {
            "type": "integer",
            "description": "Maximum allowed size",
            "default": 10000
        },
        "log_file": {
            "type": "string",
            "description": "Optional path to save policy check log",
            "default": ""
        }
    },
    "required": ["size"]
}

EXPORT_ARCHIVE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "source_dir": {
            "type": "string",
            "description": "Directory containing files to archive",
            "default": "data/synthetic_output"
        },
        "archive_name": {
            "type": "string",
            "description": "Name of the output zip file",
            "default": "synthetic_data_export.zip"
        },
        "include_patterns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "File patterns to include (e.g., ['*.json', '*.csv'])",
            "default": ["*.json", "*.csv", "*.txt"]
        },
        "log_file": {
            "type": "string",
            "description": "Optional path to save archive log output",
            "default": ""
        }
    },
    "required": []
}

def mcp_ok(id_: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}

def mcp_err(id_: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}

@app.post("/mcp")
async def mcp(request: Request):
    """
    Minimal MCP JSON-RPC endpoint supporting:
      - tools/list
      - tools/call
    """
    try:
        payload = await request.json()
    except Exception:
        return mcp_err(None, -32700, "Parse error")

    method = payload.get("method")
    rpc_id = payload.get("id")

    # Basic lifecycle methods used by some MCP clients
    if method == "initialize":
        # Return a minimal successful init response so clients don't error
        return mcp_ok(rpc_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "logging": {}
            },
            "serverInfo": {
                "name": "mcp-synth",
                "version": __version__
            }
        })

    if method == "shutdown":
        # Client requests server shutdown; acknowledge. We don't stop the process here.
        return mcp_ok(rpc_id, {"shutdown": True})

    if method == "tools/list":
        return mcp_ok(rpc_id, {
            "tools": [
                {
                    "name": "synth_generate",
                    "description": "Generate synthetic data via Synth CLI",
                    "inputSchema": SYNTH_GENERATE_SCHEMA,
                },
                {
                    "name": "synth_inspect_model",
                    "description": "List files under the Synth model directory",
                    "inputSchema": SYNTH_INSPECT_MODEL_SCHEMA,
                },
                {
                    "name": "preview_table_head",
                    "description": "Preview the first N rows of a generated file (JSON/NDJSON/CSV)",
                    "inputSchema": PREVIEW_TABLE_HEAD_SCHEMA,
                },
                {
                    "name": "synth_stats",
                    "description": "Compute statistics on a generated data file (counts, types, ranges, distributions)",
                    "inputSchema": SYNTH_STATS_SCHEMA,
                },
                {
                    "name": "synth_import_schema",
                    "description": "Import database schema into Synth format",
                    "inputSchema": IMPORT_SCHEMA_SCHEMA,
                },
                {
                    "name": "validate_fk",
                    "description": "Validate foreign key constraints in generated data",
                    "inputSchema": VALIDATE_FK_SCHEMA,
                },
                {
                    "name": "synth_dry_run",
                    "description": "Preview data generation without creating files",
                    "inputSchema": DRY_RUN_SCHEMA,
                },
                {
                    "name": "policy_check",
                    "description": "Check generation request against organizational policies",
                    "inputSchema": POLICY_CHECK_SCHEMA,
                },
                {
                    "name": "export_archive",
                    "description": "Zip output files into a compressed archive",
                    "inputSchema": EXPORT_ARCHIVE_SCHEMA,
                },
            ]
        })

    if method == "tools/call":
        params = payload.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}

        try:
            if name == "synth_generate":
                req = GenerateRequest(**args)
                resp = synth_generate(req)
                data = resp.model_dump()
                # Format output with generated data preview
                command_text = data.get("command", "(unknown)")
                text_output = (
                    f"🛠 Command: {command_text}\n\n"
                    f"✅ {data['message']}\n\n"
                    f"📁 Files created: {', '.join(data['files_created'])}\n\n"
                    f"📊 Generated Data:\n{json.dumps(data.get('data', {}), indent=2)}"
                )
                
                # Save formatted output to a log file
                if req.log_file:
                    log_file = Path(req.log_file)
                else:
                    log_dir = Path(req.out_dir)
                    log_file = log_dir / "generation_log.txt"
                
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(text_output)
                
                text_output += f"\n\n💾 Output also saved to: {log_file}"
                return mcp_ok(rpc_id, {"content": [
                    {"type": "text", "text": text_output},
                    {"type": "json", "data": data}
                ]})
            elif name == "synth_inspect_model":
                req = InspectModelRequest(**args)
                resp = synth_inspect_model(req)
                data = resp.model_dump()
                text_output = f"📂 Model directory: {data['model_dir']}\n\n📄 Files ({len(data['files'])}):\n" + "\n".join(f"  - {f}" for f in data['files'])
                
                # Save inspection output
                if req.log_file:
                    log_file = Path(req.log_file)
                else:
                    log_file = Path("data/synthetic_output/model_inspection.txt")
                
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(text_output)
                
                text_output += f"\n\n💾 Output also saved to: {log_file}"
                return mcp_ok(rpc_id, {"content": [
                    {"type": "text", "text": text_output},
                    {"type": "json", "data": data}
                ]})
            elif name == "preview_table_head":
                req = PreviewHeadRequest(**args)
                resp = preview_table_head(req)
                data = resp.model_dump()
                text_output = f"📄 Preview of {data['path']} (first {len(data['rows'])} rows):\n\n{json.dumps(data['rows'], indent=2)}"
                
                # Save preview output
                if req.log_file:
                    log_file = Path(req.log_file)
                else:
                    preview_path = Path(req.path)
                    log_file = preview_path.parent / f"{preview_path.stem}_preview.txt"
                
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(text_output)
                
                text_output += f"\n\n💾 Output also saved to: {log_file}"
                return mcp_ok(rpc_id, {"content": [
                    {"type": "text", "text": text_output},
                    {"type": "json", "data": data}
                ]})
            elif name == "synth_stats":
                req = SynthStatsRequest(**args)
                resp = synth_stats(req)
                data = resp.model_dump()
                
                # Format statistics for readability
                text_parts = [f"📊 Statistics for {data['path']}\n"]
                text_parts.append(f"📈 Total records: {data['total_records']:,}\n")
                
                for collection_name, collection_stats in data['collections'].items():
                    text_parts.append(f"\n📦 Collection: {collection_name}")
                    text_parts.append(f"   Records: {collection_stats['count']:,}")
                    text_parts.append(f"   Fields: {len(collection_stats['fields'])}\n")
                    
                    for field_name, field_stat in collection_stats['fields'].items():
                        text_parts.append(f"   • {field_name}:")
                        text_parts.append(f"     - Count: {field_stat['count']:,}, Nulls: {field_stat['null_count']}")
                        text_parts.append(f"     - Types: {', '.join(field_stat['types'])}")
                        if 'min' in field_stat:
                            text_parts.append(f"     - Range: [{field_stat['min']}, {field_stat['max']}], Mean: {field_stat['mean']:.2f}")
                        if 'unique_count' in field_stat:
                            text_parts.append(f"     - Unique: {field_stat['unique_count']}")
                            if 'sample_values' in field_stat:
                                samples = ', '.join(str(v) for v in field_stat['sample_values'][:3])
                                text_parts.append(f"     - Samples: {samples}")
                
                text_output = "\n".join(text_parts)
                
                # Save stats output
                if req.log_file:
                    log_file = Path(req.log_file)
                else:
                    stats_path = Path(req.path)
                    log_file = stats_path.parent / f"{stats_path.stem}_stats.txt"
                
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(text_output)
                
                text_output += f"\n\n💾 Stats saved to: {log_file}"
                return mcp_ok(rpc_id, {"content": [
                    {"type": "text", "text": text_output},
                    {"type": "json", "data": data}
                ]})
            elif name == "synth_import_schema":
                req = ImportSchemaRequest(**args)
                resp = synth_import_schema(req)
                data = resp.model_dump()
                text_output = f"📥 {data['message']}\n\n📄 Schema files: {', '.join(data['schema_files'])}"
                return mcp_ok(rpc_id, {"content": [
                    {"type": "text", "text": text_output},
                    {"type": "json", "data": data}
                ]})
            elif name == "validate_fk":
                req = ValidateFKRequest(**args)
                resp = validate_fk(req)
                data = resp.model_dump()
                
                if data['valid']:
                    text_output = f"✅ {data['message']}"
                else:
                    text_output = f"❌ {data['message']}\n\n"
                    text_output += f"Violations ({len(data['violations'])}):\n"
                    for v in data['violations'][:10]:  # Show first 10
                        text_output += f"  • {v['collection']}[{v['index']}].{v['field']}: {v['message']}\n"
                    if len(data['violations']) > 10:
                        text_output += f"  ... and {len(data['violations']) - 10} more"
                
                return mcp_ok(rpc_id, {"content": [
                    {"type": "text", "text": text_output},
                    {"type": "json", "data": data}
                ]})
            elif name == "synth_dry_run":
                req = DryRunRequest(**args)
                resp = synth_dry_run(req)
                data = resp.model_dump()
                
                text_output = f"🔍 {data['message']}\n\n"
                text_output += f"Preview:\n{json.dumps(data['preview'], indent=2)[:500]}..."
                
                return mcp_ok(rpc_id, {"content": [
                    {"type": "text", "text": text_output},
                    {"type": "json", "data": data}
                ]})
            elif name == "policy_check":
                req = PolicyCheckRequest(**args)
                resp = policy_check(req)
                data = resp.model_dump()
                
                if data['allowed']:
                    text_output = f"✅ {data['message']}"
                    if data['warnings']:
                        text_output += "\n\n⚠️ Warnings:\n"
                        for w in data['warnings']:
                            text_output += f"  • {w}\n"
                else:
                    text_output = f"🚫 {data['message']}"
                
                return mcp_ok(rpc_id, {"content": [
                    {"type": "text", "text": text_output},
                    {"type": "json", "data": data}
                ]})
            elif name == "export_archive":
                req = ExportArchiveRequest(**args)
                resp = export_archive(req)
                data = resp.model_dump()
                
                # Format size for human readability
                size_kb = data['archive_size_bytes'] / 1024
                size_mb = size_kb / 1024
                size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{size_kb:.2f} KB"
                
                text_output = (
                    f"📦 Archive created successfully!\n\n"
                    f"📁 Archive: {data['archive_path']}\n"
                    f"📊 Size: {size_str} ({data['archive_size_bytes']:,} bytes)\n"
                    f"📄 Files archived ({len(data['files_archived'])}):\n" +
                    "\n".join(f"  - {f}" for f in data['files_archived'])
                )
                
                # Save archive log
                if req.log_file:
                    log_file = Path(req.log_file)
                else:
                    archive_path = Path(data['archive_path'])
                    log_file = archive_path.parent / f"{archive_path.stem}_log.txt"
                
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(text_output)
                
                text_output += f"\n\n💾 Log saved to: {log_file}"
                return mcp_ok(rpc_id, {"content": [
                    {"type": "text", "text": text_output},
                    {"type": "json", "data": data}
                ]})
            else:
                return mcp_err(rpc_id, -32601, f"Unknown tool: {name}")

            # MCP result payload: array of content items
            return mcp_ok(rpc_id, {"content": [{"type": "json", "data": data}]})

        except ValidationError as ve:
            return mcp_err(rpc_id, -32602, f"Invalid params: {ve}")
        except HTTPException as he:
            return mcp_err(rpc_id, he.status_code, he.detail)
        except Exception as e:  # noqa: BLE001
            return mcp_err(rpc_id, -32000, f"Server error: {e}")

    # Handle logging/setLevel method (VS Code MCP client compatibility)
    if method == "logging/setLevel":
        # Accept the request but do nothing - we use Python's logging
        return mcp_ok(rpc_id, {})

    # Unknown method
    return mcp_err(rpc_id, -32601, f"Unknown method: {method}")
