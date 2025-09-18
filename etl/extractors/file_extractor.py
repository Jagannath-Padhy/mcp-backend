"""
File Extractor

Extracts data from static files (JSON, CSV, etc.).
Useful for importing existing catalog data or test datasets.
"""

import json
import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd

from .base_extractor import BaseExtractor, ExtractionResult, ExtractionConfig

logger = logging.getLogger(__name__)


class FileExtractor(BaseExtractor):
    """
    Extractor for file-based data sources
    
    Supports:
    - JSON files (single objects or arrays)
    - CSV files 
    - Excel files (via pandas)
    - Directory scanning for multiple files
    """
    
    def __init__(self, config: ExtractionConfig, file_config: Dict[str, Any]):
        super().__init__(config, "file_system")
        
        # File system configuration
        self.data_path = file_config.get("data_path", "./data")
        self.supported_formats = file_config.get("formats", ["json", "csv", "xlsx"])
        self.watch_for_changes = file_config.get("watch_for_changes", False)
        
        # File processing settings
        self.encoding = file_config.get("encoding", "utf-8")
        self.csv_delimiter = file_config.get("csv_delimiter", ",")
        self.json_lines = file_config.get("json_lines", False)  # For JSONL format
        
        # Data type mappings (file patterns to data types)
        self.file_patterns = {
            "products": ["product", "item", "catalog"],
            "categories": ["category", "categories", "taxonomy"],
            "providers": ["provider", "seller", "vendor", "merchant"]
        }
        
    async def health_check(self) -> bool:
        """Check if data directory is accessible"""
        try:
            data_dir = Path(self.data_path)
            return data_dir.exists() and data_dir.is_dir()
        except Exception as e:
            logger.error(f"File extractor health check failed: {e}")
            return False
            
    async def extract_products(self, **kwargs) -> ExtractionResult:
        """
        Extract product data from files
        
        Args:
            file_path (str): Specific file path (optional)
            pattern (str): File pattern to match (optional)
        """
        return await self._extract_by_type("products", **kwargs)
        
    async def extract_categories(self, **kwargs) -> ExtractionResult:
        """Extract category data from files"""
        return await self._extract_by_type("categories", **kwargs)
        
    async def extract_providers(self, **kwargs) -> ExtractionResult:
        """Extract provider data from files"""
        return await self._extract_by_type("providers", **kwargs)
        
    async def _extract_by_type(self, data_type: str, **kwargs) -> ExtractionResult:
        """
        Extract data of a specific type from files
        """
        try:
            logger.info(f"Extracting {data_type} from files")
            
            # Get target files
            file_path = kwargs.get("file_path")
            pattern = kwargs.get("pattern")
            
            if file_path:
                # Extract from specific file
                files = [Path(file_path)]
            else:
                # Find files by pattern
                files = await self._find_files_by_type(data_type, pattern)
                
            if not files:
                return ExtractionResult(
                    success=False,
                    data=[],
                    errors=[f"No files found for {data_type}"],
                    metadata={"search_path": self.data_path},
                    extracted_at=datetime.utcnow(),
                    source=self.source_name,
                    total_records=0
                )
                
            all_data = []
            errors = []
            processed_files = []
            
            for file_path in files:
                try:
                    logger.info(f"Processing file: {file_path}")
                    
                    file_data = await self._extract_from_file(file_path)
                    
                    if file_data:
                        # Add file metadata to each record
                        for record in file_data:
                            if isinstance(record, dict):
                                record["source_file"] = str(file_path)
                                record["extracted_at"] = datetime.utcnow().isoformat()
                                
                        all_data.extend(file_data)
                        processed_files.append(str(file_path))
                        
                except Exception as e:
                    error_msg = f"Error processing file {file_path}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    
            # Validate extracted data
            valid_data, validation_errors = self.validate_data(all_data)
            errors.extend(validation_errors)
            
            return ExtractionResult(
                success=len(valid_data) > 0,
                data=valid_data,
                errors=errors,
                metadata={
                    "files_processed": processed_files,
                    "data_type": data_type
                },
                extracted_at=datetime.utcnow(),
                source=self.source_name,
                total_records=len(valid_data)
            )
            
        except Exception as e:
            logger.error(f"File extraction failed for {data_type}: {e}")
            return ExtractionResult(
                success=False,
                data=[],
                errors=[str(e)],
                metadata={"data_type": data_type},
                extracted_at=datetime.utcnow(),
                source=self.source_name,
                total_records=0
            )
            
    async def _find_files_by_type(self, 
                                 data_type: str, 
                                 pattern: Optional[str] = None) -> List[Path]:
        """
        Find files matching a data type pattern
        """
        data_dir = Path(self.data_path)
        
        if not data_dir.exists():
            logger.warning(f"Data directory does not exist: {data_dir}")
            return []
            
        files = []
        patterns = self.file_patterns.get(data_type, [data_type])
        
        if pattern:
            patterns.append(pattern)
            
        for file_path in data_dir.rglob("*"):
            if not file_path.is_file():
                continue
                
            # Check file extension
            if file_path.suffix[1:].lower() not in self.supported_formats:
                continue
                
            # Check if filename matches any pattern
            filename_lower = file_path.stem.lower()
            
            for pattern in patterns:
                if pattern.lower() in filename_lower:
                    files.append(file_path)
                    break
                    
        return files
        
    async def _extract_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Extract data from a single file based on its format
        """
        file_extension = file_path.suffix[1:].lower()
        
        if file_extension == "json":
            return await self._extract_from_json(file_path)
        elif file_extension == "csv":
            return await self._extract_from_csv(file_path)
        elif file_extension in ["xlsx", "xls"]:
            return await self._extract_from_excel(file_path)
        elif file_extension == "jsonl":
            return await self._extract_from_jsonl(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
            
    async def _extract_from_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract data from JSON file"""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Try common array keys
                for key in ['data', 'items', 'products', 'records', 'results']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                        
                # Single object - wrap in array
                return [data]
            else:
                logger.warning(f"Unexpected JSON structure in {file_path}")
                return []
                
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return []
            
    async def _extract_from_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract data from JSONL (JSON Lines) file"""
        try:
            data = []
            with open(file_path, 'r', encoding=self.encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        record = json.loads(line)
                        data.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                        
            return data
            
        except Exception as e:
            logger.error(f"Error reading JSONL file {file_path}: {e}")
            return []
            
    async def _extract_from_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract data from CSV file"""
        try:
            data = []
            with open(file_path, 'r', encoding=self.encoding, newline='') as f:
                # Try to detect delimiter
                sample = f.read(1024)
                f.seek(0)
                sniffer = csv.Sniffer()
                try:
                    delimiter = sniffer.sniff(sample).delimiter
                except:
                    delimiter = self.csv_delimiter
                    
                reader = csv.DictReader(f, delimiter=delimiter)
                
                for row_num, row in enumerate(reader, 1):
                    try:
                        # Clean up row data
                        cleaned_row = {}
                        for key, value in row.items():
                            if key is not None:  # Skip None keys
                                # Convert string representations
                                cleaned_value = self._convert_csv_value(value)
                                cleaned_row[key.strip()] = cleaned_value
                                
                        if cleaned_row:  # Only add non-empty rows
                            data.append(cleaned_row)
                            
                    except Exception as e:
                        logger.warning(f"Error processing row {row_num} in {file_path}: {e}")
                        
            return data
            
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return []
            
    async def _extract_from_excel(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract data from Excel file"""
        try:
            # Use pandas for Excel files
            df = pd.read_excel(file_path)
            
            # Convert to list of dictionaries
            data = df.to_dict('records')
            
            # Clean up the data
            cleaned_data = []
            for record in data:
                cleaned_record = {}
                for key, value in record.items():
                    # Handle NaN values
                    if pd.isna(value):
                        value = None
                    # Convert numpy types to Python types
                    elif hasattr(value, 'item'):
                        value = value.item()
                        
                    cleaned_record[str(key)] = value
                    
                cleaned_data.append(cleaned_record)
                
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {e}")
            return []
            
    def _convert_csv_value(self, value: str) -> Union[str, int, float, bool, None]:
        """
        Convert string values from CSV to appropriate Python types
        """
        if not isinstance(value, str):
            return value
            
        value = value.strip()
        
        # Handle empty values
        if not value or value.lower() in ['null', 'none', 'n/a', '']:
            return None
            
        # Handle boolean values
        if value.lower() in ['true', 'yes', '1']:
            return True
        elif value.lower() in ['false', 'no', '0']:
            return False
            
        # Try to convert to number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
            
        # Return as string
        return value
        
    async def list_available_files(self) -> Dict[str, List[str]]:
        """
        List all available files organized by data type
        """
        file_lists = {}
        
        for data_type in ["products", "categories", "providers"]:
            files = await self._find_files_by_type(data_type)
            file_lists[data_type] = [str(f) for f in files]
            
        return file_lists
        
    async def get_file_stats(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get statistics about a data file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": "File not found"}
            
        try:
            stats = {
                "path": str(file_path),
                "size_bytes": file_path.stat().st_size,
                "modified_at": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "format": file_path.suffix[1:].lower()
            }
            
            # Try to get record count
            data = await self._extract_from_file(file_path)
            stats["record_count"] = len(data)
            
            if data:
                # Get sample of field names
                sample_record = data[0]
                if isinstance(sample_record, dict):
                    stats["fields"] = list(sample_record.keys())
                    
            return stats
            
        except Exception as e:
            return {"error": str(e)}