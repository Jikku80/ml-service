from fastapi import APIRouter, Body, HTTPException, Query, Path, Response
from pydantic import BaseModel, field_validator, Field
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
import logging
import os
import csv
from dotenv import load_dotenv
import pymongo

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("multi_db_api")

# Create directory for CSV exports
CSV_EXPORT_DIR = os.getenv("CSV_EXPORT_DIR", "data_exports")
os.makedirs(CSV_EXPORT_DIR, exist_ok=True)

router = APIRouter(
    prefix="/dbconnect",
    tags=["dbconnect"],
    responses={404: {"description": "Not found"}},
)

# Database connection models
class DatabaseConnection(BaseModel):
    db_type: str = Field(..., description="Database type: postgres, mysql, sqlite, or mongodb")
    host: Optional[str] = Field(None, description="Database host (not needed for SQLite)")
    port: Optional[int] = Field(None, description="Database port (not needed for SQLite)")
    username: Optional[str] = Field(None, description="Database username (not needed for SQLite)")
    password: Optional[str] = Field(None, description="Database password (not needed for SQLite)")
    database: str = Field(..., description="Database name or SQLite file path")
    
    @field_validator('db_type')
    def validate_db_type(cls, v):
        if v not in ['postgres', 'mysql', 'sqlite', 'mongodb']:
            raise ValueError('db_type must be postgres, mysql, sqlite, or mongodb')
        return v
    
    @field_validator('port', 'host', 'username', 'password')
    def validate_connection_params(cls, v, info):
        # Get the values from the validation context
        data = info.data
        
        # Check if db_type exists in the data
        if 'db_type' not in data:
            # Can't validate without db_type, so return as is
            return v
            
        db_type = data.get('db_type')
        field_name = info.field_name
        
        # For non-sqlite databases, these fields are required
        if db_type != 'sqlite' and v is None and field_name in ['host', 'port', 'username', 'password']:
            # MongoDB can have anonymous authentication (no username/password)
            if db_type == 'mongodb' and field_name in ['username', 'password']:
                return v
            else:
                raise ValueError(f"{field_name} is required for {db_type}")
            
        return v

# Connection manager for database operations
class DatabaseManager:
    def __init__(self):
        # User connection cache to avoid recreating engines
        self.connection_cache = {}
    
    def get_connection_string(self, connection: DatabaseConnection) -> str:
        """Create a connection string based on the database type"""
        if connection.db_type == 'postgres':
            return f"postgresql://{connection.username}:{connection.password}@{connection.host}:{connection.port}/{connection.database}"
        elif connection.db_type == 'mysql':
            return f"mysql+pymysql://{connection.username}:{connection.password}@{connection.host}:{connection.port}/{connection.database}"
        elif connection.db_type == 'sqlite':
            return f"sqlite:///{connection.database}"
        elif connection.db_type == 'mongodb':
            # MongoDB doesn't use SQLAlchemy connection strings
            # Just return None as we'll handle MongoDB differently
            return None
        else:
            raise ValueError(f"Unsupported database type: {connection.db_type}")
    
    def get_engine(self, user_id: str, connection: DatabaseConnection):
        """Get or create an engine for the specified user and connection"""
        cache_key = f"{user_id}_{connection.db_type}_{connection.database}"
        
        if cache_key not in self.connection_cache:
            if connection.db_type == 'mongodb':
                # For MongoDB, create a pymongo client instead of SQLAlchemy engine
                auth_part = ""
                if connection.username and connection.password:
                    mongo_client = pymongo.MongoClient(
                        host=connection.host,
                        port=connection.port,
                        username=connection.username,
                        password=connection.password,
                        authSource=connection.database
                    )
                else:
                    mongo_client = pymongo.MongoClient(
                        host=connection.host,
                        port=connection.port
                    )
                # Test the connection
                mongo_client.admin.command('ping')
                self.connection_cache[cache_key] = mongo_client
            else:
                # For SQL databases, use SQLAlchemy as before
                connection_string = self.get_connection_string(connection)
                engine = create_engine(
                    connection_string,
                    pool_pre_ping=True,  # Verify connection before using
                    pool_recycle=3600    # Recycle connections after an hour
                )
                self.connection_cache[cache_key] = engine
            
        return self.connection_cache[cache_key]
    
    @contextmanager
    def get_connection(self, user_id: str, connection: DatabaseConnection):
        """Context manager for database connections"""
        if connection.db_type == 'mongodb':
            # For MongoDB, yield the database from the client
            mongo_client = self.get_engine(user_id, connection)
            try:
                db = mongo_client[connection.database]
                yield db
            except Exception as e:
                logger.error(f"MongoDB error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"MongoDB connection error: {str(e)}")
        else:
            # For SQL databases, use SQLAlchemy connection
            engine = self.get_engine(user_id, connection)
            db_conn = None
            try:
                db_conn = engine.connect()
                yield db_conn
            except SQLAlchemyError as e:
                logger.error(f"Database error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
            finally:
                if db_conn:
                    db_conn.close()

# Initialize the database manager
db_manager = DatabaseManager()

# Request and response models
class TableListResponse(BaseModel):
    tables: List[str]

class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool

class TableSchemaResponse(BaseModel):
    table_name: str
    columns: List[ColumnInfo]

class QueryRequest(BaseModel):
    sql: str

class QueryResponse(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    csv_export_path: Optional[str] = None

# Database connection storage (in-memory for this example - in production use a proper database)
# Key is user_id, value is dict of named connections
user_connections = {}

# API endpoints
@router.post("/users/{user_id}/connections/{connection_name}", response_model=dict)
async def create_connection(
    user_id: str = Path(..., description="User ID"),
    connection_name: str = Path(..., description="Connection name"),
    connection: DatabaseConnection = Body(...)
):
    """Create or update a database connection for a user"""
    if user_id not in user_connections:
        user_connections[user_id] = {}
        
    # Store the connection
    user_connections[user_id][connection_name] = connection
    
    # Test the connection to make sure it works
    try:
        if connection.db_type == 'mongodb':
            # For MongoDB, just getting the engine will test the connection
            # as we do a ping in the get_engine method
            db_manager.get_engine(user_id, connection)
        else:
            # For SQL databases, execute a simple query
            engine = db_manager.get_engine(user_id, connection)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
    except Exception as e:
        # If connection fails, remove it from storage
        del user_connections[user_id][connection_name]
        raise HTTPException(status_code=400, detail=f"Connection test failed: {str(e)}")
        
    return {"status": "success", "message": f"Connection '{connection_name}' created successfully"}

@router.get("/users/{user_id}/connections", response_model=List[str])
async def list_connections(user_id: str = Path(..., description="User ID")):
    """List all connections for a user"""
    if user_id not in user_connections:
        raise HTTPException(status_code=404, detail=f"No connections found for user {user_id}")
    
    return list(user_connections[user_id].keys())

@router.get("/users/{user_id}/connections/{connection_name}/tables", response_model=TableListResponse)
async def list_tables(
    user_id: str = Path(..., description="User ID"),
    connection_name: str = Path(..., description="Connection name")
):
    """List all tables in the database for the given connection"""
    # Check if user and connection exist
    if user_id not in user_connections or connection_name not in user_connections[user_id]:
        raise HTTPException(status_code=404, detail=f"Connection '{connection_name}' not found for user {user_id}")
    
    connection = user_connections[user_id][connection_name]
    
    try:
        if connection.db_type == 'mongodb':
            # For MongoDB, get collections instead of tables
            mongo_client = db_manager.get_engine(user_id, connection)
            db = mongo_client[connection.database]
            tables = db.list_collection_names()
            return TableListResponse(tables=tables)
        else:
            # For SQL databases, use SQLAlchemy inspector
            engine = db_manager.get_engine(user_id, connection)
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            return TableListResponse(tables=tables)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tables: {str(e)}")

@router.get("/users/{user_id}/connections/{connection_name}/tables/{table_name}/schema", response_model=TableSchemaResponse)
async def get_table_schema(
    user_id: str = Path(..., description="User ID"),
    connection_name: str = Path(..., description="Connection name"),
    table_name: str = Path(..., description="Table name")
):
    """Get the schema of a specific table"""
    # Check if user and connection exist
    if user_id not in user_connections or connection_name not in user_connections[user_id]:
        raise HTTPException(status_code=404, detail=f"Connection '{connection_name}' not found for user {user_id}")
    
    connection = user_connections[user_id][connection_name]
    
    try:
        if connection.db_type == 'mongodb':
            # For MongoDB, infer schema from a sample document
            mongo_client = db_manager.get_engine(user_id, connection)
            db = mongo_client[connection.database]
            
            # Check if collection exists
            if table_name not in db.list_collection_names():
                raise HTTPException(status_code=404, detail=f"Collection '{table_name}' not found")
            
            collection = db[table_name]
            
            # Get a sample document to infer schema
            sample_doc = collection.find_one()
            if not sample_doc:
                # Empty collection
                return TableSchemaResponse(table_name=table_name, columns=[])
            
            # Infer schema from the sample document
            columns = []
            for key, value in sample_doc.items():
                # Skip MongoDB's internal _id field if desired
                if key == '_id':
                    columns.append(ColumnInfo(name=key, type="ObjectId", nullable=False))
                else:
                    columns.append(ColumnInfo(
                        name=key,
                        type=type(value).__name__,
                        nullable=True  # MongoDB fields are generally nullable
                    ))
            
            return TableSchemaResponse(table_name=table_name, columns=columns)
        else:
            # For SQL databases, use SQLAlchemy inspector
            engine = db_manager.get_engine(user_id, connection)
            inspector = inspect(engine)
            
            # Check if table exists
            if table_name not in inspector.get_table_names():
                raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found")
            
            # Get column info
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append(
                    ColumnInfo(
                        name=column['name'],
                        type=str(column['type']),
                        nullable=column.get('nullable', True)
                    )
                )
            
            return TableSchemaResponse(table_name=table_name, columns=columns)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting table schema: {str(e)}")

def save_to_csv(data: List[Dict[str, Any]], columns: List[str], user_id: str, description: str = "query") -> str:
    """Save query results to a CSV file and return the file path"""
    filename = f"{user_id}_{description}.csv"
    filepath = os.path.join(CSV_EXPORT_DIR, filename)
    
    try:
        # First pass: collect all possible field names across all rows
        all_columns = set(columns)
        for row in data:
            all_columns.update(row.keys())
        
        # Convert to list and ensure consistent ordering
        all_columns = list(all_columns)
        
        # Second pass: write with all columns
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_columns, extrasaction='ignore')
            writer.writeheader()
            
            # Create sanitized rows where we ensure every column exists in every row
            sanitized_rows = []
            for row in data:
                # Create a new dict with None for missing fields
                sanitized_row = {col: row.get(col, None) for col in all_columns}
                sanitized_rows.append(sanitized_row)
            
            writer.writerows(sanitized_rows)
            
        logger.info(f"Data exported to {filepath}")
        return filename
    except Exception as e:
        logger.error(f"Error saving to CSV: {str(e)}")
        return None

@router.post("/users/{user_id}/connections/{connection_name}/query", response_model=QueryResponse)
async def execute_query(
    user_id: str = Path(..., description="User ID"),
    connection_name: str = Path(..., description="Connection name"),
    query: QueryRequest = Body(...),
    export_csv: bool = Query(False, description="Export results to CSV file")
):
    """Execute a SQL query on the database"""
    # Check if user and connection exist
    if user_id not in user_connections or connection_name not in user_connections[user_id]:
        raise HTTPException(status_code=404, detail=f"Connection '{connection_name}' not found for user {user_id}")
    
    connection = user_connections[user_id][connection_name]
    
    # MongoDB doesn't support SQL queries directly
    if connection.db_type == 'mongodb':
        raise HTTPException(
            status_code=400, 
            detail="MongoDB doesn't support SQL queries. Use the MongoDB-specific endpoints instead."
        )
    
    try:
        with db_manager.get_connection(user_id, connection) as conn:
            # Execute the query
            result = conn.execute(text(query.sql))
            
            # Convert result to dict
            columns = result.keys()
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Export to CSV if requested
            csv_export_path = None
            if export_csv and rows:
                csv_export_path = save_to_csv(
                    data=rows,
                    columns=list(columns),
                    user_id=user_id,
                    description=f"{connection_name}_query"
                )
            
            return QueryResponse(
                columns=list(columns),
                rows=rows,
                row_count=len(rows),
                csv_export_path=csv_export_path
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query execution error: {str(e)}")

@router.post("/users/{user_id}/connections/{connection_name}/mongodb/find", response_model=QueryResponse)
async def mongodb_find(
    user_id: str = Path(..., description="User ID"),
    connection_name: str = Path(..., description="Connection name"),
    collection: str = Query(..., description="MongoDB collection name"),
    filter_query: Dict[str, Any] = Body({}, description="MongoDB filter query"),
    projection: Dict[str, Any] = Body(None, description="MongoDB projection"),
    limit: int = Query(100, description="Maximum number of documents to return"),
    skip: int = Query(0, description="Number of documents to skip"),
    export_csv: bool = Query(False, description="Export results to CSV file")
):
    """Execute a MongoDB find operation"""
    # Check if user and connection exist
    if user_id not in user_connections or connection_name not in user_connections[user_id]:
        raise HTTPException(status_code=404, detail=f"Connection '{connection_name}' not found for user {user_id}")
    
    connection = user_connections[user_id][connection_name]
    
    # Ensure this is a MongoDB connection
    if connection.db_type != 'mongodb':
        raise HTTPException(
            status_code=400, 
            detail="This endpoint is only for MongoDB connections."
        )
    
    try:
        with db_manager.get_connection(user_id, connection) as db:
            # Get the collection
            coll = db[collection]
            
            # Execute the find operation
            cursor = coll.find(
                filter=filter_query,
                projection=projection
            ).limit(limit).skip(skip)
            
            # Convert results to list of dicts
            rows = list(cursor)
            
            # Handle ObjectId and other MongoDB-specific types 
            for row in rows:
                for key, value in row.items():
                    if key == '_id' and value is not None:
                        row[key] = str(value)
            
            # Collect all unique keys across all documents for MongoDB
            columns = set()
            for row in rows:
                columns.update(row.keys())
            columns = list(columns)
            
            # Export to CSV if requested 
            csv_export_path = None
            if export_csv and rows:
                csv_export_path = save_to_csv(
                    data=rows,
                    columns=columns,
                    user_id=user_id,
                    description=f"{connection_name}_{collection}_find"
                )
            
            return QueryResponse(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                csv_export_path=csv_export_path
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"MongoDB find error: {str(e)}")

@router.get("/users/{user_id}/connections/{connection_name}/tables/{table_name}/preview", response_model=QueryResponse)
async def preview_table_data(
    user_id: str = Path(..., description="User ID"),
    connection_name: str = Path(..., description="Connection name"),
    table_name: str = Path(..., description="Table name"),
    limit: int = Query(10, description="Number of rows to preview"),
    export_csv: bool = Query(False, description="Export results to CSV file")
):
    """Preview data from a table (first n rows)"""
    # Check if user and connection exist
    if user_id not in user_connections or connection_name not in user_connections[user_id]:
        raise HTTPException(status_code=404, detail=f"Connection '{connection_name}' not found for user {user_id}")
    
    connection = user_connections[user_id][connection_name]
    
    try:
        if connection.db_type == 'mongodb':
            # For MongoDB, use find operation
            mongo_client = db_manager.get_engine(user_id, connection)
            db = mongo_client[connection.database]
            collection = db[table_name]
            
            # Get the first n documents
            cursor = collection.find().limit(limit)
            rows = list(cursor)
            
            # Handle ObjectId and other MongoDB-specific types
            for row in rows:
                for key, value in row.items():
                    if key == '_id' and value is not None:
                        row[key] = str(value)
            
            # Get columns from first row or provide empty list
            columns = list(rows[0].keys()) if rows else []
            
            # Export to CSV if requested
            csv_export_path = None
            if export_csv and rows:
                csv_export_path = save_to_csv(
                    data=rows,
                    columns=columns,
                    user_id=user_id,
                    description=f"{connection_name}_{table_name}_preview"
                )
            
            return QueryResponse(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                csv_export_path=csv_export_path
            )
        else:
            # For SQL databases, use SQL query
            with db_manager.get_connection(user_id, connection) as conn:
                # Execute the query
                result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT {limit}"))
                
                # Convert result to dict
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                
                # Export to CSV if requested
                csv_export_path = None
                if export_csv and rows:
                    csv_export_path = save_to_csv(
                        data=rows,
                        columns=list(columns),
                        user_id=user_id,
                        description=f"{connection_name}_{table_name}_preview"
                    )
                
                return QueryResponse(
                    columns=list(columns),
                    rows=rows,
                    row_count=len(rows),
                    csv_export_path=csv_export_path
                )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error previewing table data: {str(e)}")

@router.get("/users/{user_id}/connections/{connection_name}/tables/{table_name}/export", response_model=QueryResponse)
async def export_table_data(
    user_id: str = Path(..., description="User ID"),
    connection_name: str = Path(..., description="Connection name"),
    table_name: str = Path(..., description="Table name"),
    limit: Optional[int] = Query(None, description="Optional limit for number of rows")
):
    """Export entire table data to CSV file"""
    # Check if user and connection exist
    if user_id not in user_connections or connection_name not in user_connections[user_id]:
        raise HTTPException(status_code=404, detail=f"Connection '{connection_name}' not found for user {user_id}")
    
    connection = user_connections[user_id][connection_name]
    
    try:
        if connection.db_type == 'mongodb':
            # For MongoDB, use find operation
            mongo_client = db_manager.get_engine(user_id, connection)
            db = mongo_client[connection.database]
            collection = db[table_name]
            
            # Get documents with optional limit
            cursor = collection.find()
            if limit:
                cursor = cursor.limit(limit)
            
            rows = list(cursor)
            
            # Handle ObjectId and other MongoDB-specific types
            for row in rows:
                for key, value in row.items():
                    if key == '_id' and value is not None:
                        row[key] = str(value)
            
            # Get columns from first row or provide empty list
            columns = list(rows[0].keys()) if rows else []
            
            # Always export to CSV for this endpoint
            csv_export_path = save_to_csv(
                data=rows,
                columns=columns,
                user_id=user_id,
                description=f"{connection_name}_{table_name}_export"
            )
            
            return QueryResponse(
                columns=columns,
                rows=rows,
                row_count=len(rows),
                csv_export_path=csv_export_path
            )
        else:
            # For SQL databases
            with db_manager.get_connection(user_id, connection) as conn:
                # Build query with optional limit
                query = f"SELECT * FROM {table_name}"
                if limit:
                    query += f" LIMIT {limit}"
                
                # Execute the query
                result = conn.execute(text(query))
                
                # Convert result to dict
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                
                # Always export to CSV for this endpoint
                csv_export_path = save_to_csv(
                    data=rows,
                    columns=list(columns),
                    user_id=user_id,
                    description=f"{connection_name}_{table_name}_export"
                )
                
                return QueryResponse(
                    columns=list(columns),
                    rows=rows,
                    row_count=len(rows),
                    csv_export_path=csv_export_path
                )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error exporting table data: {str(e)}")

@router.get("/users/{user_id}/exports/{filename}")
async def download_csv_file(
    user_id: str = Path(..., description="User ID"),
    filename: str = Path(..., description="CSV filename")
):
    """Download a previously generated CSV file"""
    # Verify filename belongs to the user (basic security)
    if not filename.startswith(f"{user_id}_"):
        raise HTTPException(status_code=403, detail="Access denied to this file")
    
    file_path = os.path.join(CSV_EXPORT_DIR, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Read file content
    with open(file_path, "rb") as f:
        content = f.read()
    
    os.remove(file_path)
    # Return the file as a downloadable response
    return Response(
        content=content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

@router.delete("/users/{user_id}/connections/{connection_name}")
async def delete_connection(
    user_id: str = Path(..., description="User ID"),
    connection_name: str = Path(..., description="Connection name")
):
    """Delete a database connection for a user"""
    if user_id not in user_connections or connection_name not in user_connections[user_id]:
        raise HTTPException(status_code=404, detail=f"Connection '{connection_name}' not found for user {user_id}")
    
    # Remove connection from storage
    del user_connections[user_id][connection_name]
    
    # Clean up empty user dict if needed
    if not user_connections[user_id]:
        del user_connections[user_id]
    
    return {"status": "success", "message": f"Connection '{connection_name}' deleted successfully"}