import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

class QueryIntent(BaseModel):
    """Structured query intent for ARGO data"""
    intent: str = Field(description="Type of query: filter_profiles, compare_regions, nearest_floats, summary, timeseries")
    filters: Dict[str, Any] = Field(description="Filters to apply to the data")
    visualization: str = Field(description="Type of visualization: profile_plot, map, timeseries, table, comparison")
    k: int = Field(default=50, description="Maximum number of profiles to return")

class ArgoQueryProcessor:
    """LangChain-powered query processor for ARGO data"""
    
    def __init__(self, processed_folder: str, use_openai: bool = True, openai_api_key: str = None):
        self.processed_folder = processed_folder
        
        # Initialize LLM
        if use_openai and openai_api_key:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                openai_api_key=openai_api_key
            )
        else:
            # Fallback to Ollama (requires local installation)
            self.llm = Ollama(model="llama3.2:3b", temperature=0.1)
        
        # Initialize parsers
        self.json_parser = JsonOutputParser(pydantic_object=QueryIntent)
        self.str_parser = StrOutputParser()
        
        # Load data
        self.metadata_df = None
        self.full_df = None
        self._load_data()
        
        # Create prompt templates
        self._setup_prompts()
    
    def _load_data(self):
        """Load processed ARGO data"""
        try:
            metadata_file = os.path.join(self.processed_folder, "argo_profiles_metadata_indexed.parquet")
            full_data_file = os.path.join(self.processed_folder, "argo_combined_data.parquet")
            
            if os.path.exists(metadata_file):
                self.metadata_df = pd.read_parquet(metadata_file)
                print(f"Loaded metadata for {len(self.metadata_df)} profiles")
            
            if os.path.exists(full_data_file):
                self.full_df = pd.read_parquet(full_data_file)
                print(f"Loaded full dataset with {len(self.full_df)} measurements")
                
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def _setup_prompts(self):
        """Setup LangChain prompt templates"""
        
        # Query parsing prompt
        self.query_parse_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert oceanographer and data analyst. Extract structured information from user queries about ARGO ocean float data.

ARGO floats measure temperature, salinity, and pressure at different depths in the ocean. Users may ask about:
- Finding profiles with specific conditions
- Comparing different regions or time periods  
- Analyzing trends over time
- Locating floats near specific coordinates

Extract the following JSON structure:
{format_instructions}

Guidelines:
- intent: One of "filter_profiles", "compare_regions", "nearest_floats", "summary", "timeseries"
- lat_range: [min_lat, max_lat] in decimal degrees (-90 to 90)
- lon_range: [min_lon, max_lon] in decimal degrees (-180 to 180)  
- date_range: [start_date, end_date] in YYYY-MM-DD format
- depth_range: [min_depth, max_depth] in meters (pressure converted to depth)
- variables: List from ["temperature", "salinity", "pressure"] 
- float_ids: Specific float IDs if mentioned
- visualization: "profile_plot", "map", "timeseries", "table", "comparison"
- k: Number of results (default 50, max 200)

Examples:
"Show salinity profiles near equator" -> lat_range: [-5, 5], variables: ["salinity"], visualization: "profile_plot"
"Temperature trends in Arabian Sea 2019" -> lat_range: [10, 25], lon_range: [50, 80], variables: ["temperature"], visualization: "timeseries"
"Map floats in Indian Ocean" -> lat_range: [-30, 30], lon_range: [40, 120], visualization: "map"
"""),
            ("human", "{query}")
        ])
        
        # Response generation prompt  
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are FloatChat, an AI oceanographer assistant. Provide informative responses about ARGO ocean data analysis results.

Based on the user query, retrieved context, and data analysis results, provide a comprehensive but concise response that:
1. Summarizes what was found
2. Highlights key oceanographic insights
3. Explains any notable patterns or anomalies
4. Suggests follow-up analyses if relevant

Keep responses scientific but accessible. Use oceanographic terminology appropriately.
"""),
            ("human", """User Query: {query}

Retrieved Context: {context}

Analysis Results Summary:
- Number of profiles found: {num_profiles}
- Date range: {date_range}  
- Geographic coverage: {geographic_range}
- Key measurements: {measurements_summary}

Provide an informative response about these ARGO ocean measurements.""")
        ])
        
        # Create chains
        self.parse_chain = self.query_parse_prompt | self.llm | self.json_parser
        self.response_chain = self.response_prompt | self.llm | self.str_parser
    
    def parse_query(self, user_query: str) -> Dict[str, Any]:
        """Parse user query into structured format"""
        try:
            # Add format instructions
            formatted_prompt = self.query_parse_prompt.format_messages(
                query=user_query,
                format_instructions=self.json_parser.get_format_instructions()
            )
            
            result = self.parse_chain.invoke({"query": user_query})
            return result
            
        except Exception as e:
            print(f"Error parsing query: {e}")
            # Return default structure
            return {
                "intent": "filter_profiles",
                "filters": {},
                "visualization": "table", 
                "k": 50
            }
    
    def execute_structured_query(self, query_obj: Dict[str, Any]) -> pd.DataFrame:
        """Execute structured query against ARGO data"""
        if self.full_df is None or self.metadata_df is None:
            print("No data loaded")
            return pd.DataFrame()
        
        try:
            # Start with metadata for efficiency
            meta = self.metadata_df.copy()
            filters = query_obj.get("filters", {})
            
            # Apply geographic filters
            if filters.get("lat_range"):
                min_lat, max_lat = filters["lat_range"]
                meta = meta[(meta['latitude'] >= min_lat) & (meta['latitude'] <= max_lat)]
            
            if filters.get("lon_range"):
                min_lon, max_lon = filters["lon_range"]
                meta = meta[(meta['longitude'] >= min_lon) & (meta['longitude'] <= max_lon)]
            
            # Apply time filters
            if filters.get("date_range"):
                start_date, end_date = filters["date_range"]
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                meta = meta[(meta['time'] >= start_date) & (meta['time'] <= end_date)]
            
            # Apply float ID filter
            if filters.get("float_ids"):
                meta = meta[meta['float_id'].isin(filters["float_ids"])]
            
            # Limit results
            k = min(query_obj.get("k", 50), 200)  # Cap at 200 for performance
            selected_meta = meta.head(k)
            
            if selected_meta.empty:
                return pd.DataFrame()
            
            # Get full data for selected profiles
            profile_keys = selected_meta[['float_id', 'cycle']].to_records(index=False)
            full_data = self.full_df.set_index(['float_id', 'cycle'])
            
            selected_profiles = []
            for float_id, cycle in profile_keys:
                try:
                    profile_data = full_data.loc[(float_id, cycle)]
                    if isinstance(profile_data, pd.Series):
                        profile_data = profile_data.to_frame().T
                    selected_profiles.append(profile_data.reset_index())
                except KeyError:
                    continue
            
            if not selected_profiles:
                return pd.DataFrame()
            
            result_df = pd.concat(selected_profiles, ignore_index=True)
            
            # Apply depth/pressure filters
            if filters.get("depth_range") and 'pressure' in result_df.columns:
                min_depth, max_depth = filters["depth_range"]
                # Approximate depth from pressure (1 dbar ≈ 1 meter)
                result_df = result_df[
                    (result_df['pressure'] >= min_depth) & 
                    (result_df['pressure'] <= max_depth)
                ]
            
            # Filter variables if specified
            base_cols = ['float_id', 'cycle', 'latitude', 'longitude', 'time', 'pressure']
            if filters.get("variables"):
                var_cols = [v for v in filters["variables"] if v in result_df.columns]
                result_df = result_df[base_cols + var_cols]
            
            return result_df.dropna()
            
        except Exception as e:
            print(f"Error executing query: {e}")
            return pd.DataFrame()
    
    def generate_response(self, user_query: str, context: str, results_df: pd.DataFrame) -> str:
        """Generate natural language response using LangChain"""
        try:
            if results_df.empty:
                return "No ARGO profiles were found matching your criteria. Try adjusting the search parameters or expanding the geographic/temporal range."
            
            # Summarize results
            num_profiles = len(results_df.groupby(['float_id', 'cycle']))
            
            if 'time' in results_df.columns:
                time_range = f"{results_df['time'].min().date()} to {results_df['time'].max().date()}"
            else:
                time_range = "Unknown"
            
            lat_range = f"{results_df['latitude'].min():.2f}° to {results_df['latitude'].max():.2f}°N"
            lon_range = f"{results_df['longitude'].min():.2f}° to {results_df['longitude'].max():.2f}°E"
            geographic_range = f"Latitude: {lat_range}, Longitude: {lon_range}"
            
            # Measurement summaries
            measurements = []
            if 'temperature' in results_df.columns:
                temp_stats = results_df['temperature'].describe()
                measurements.append(f"Temperature: {temp_stats['mean']:.1f}°C (±{temp_stats['std']:.1f}°C)")
            
            if 'salinity' in results_df.columns:
                sal_stats = results_df['salinity'].describe()
                measurements.append(f"Salinity: {sal_stats['mean']:.2f} PSU (±{sal_stats['std']:.2f} PSU)")
            
            if 'pressure' in results_df.columns:
                pres_stats = results_df['pressure'].describe()
                measurements.append(f"Depth: 0-{pres_stats['max']:.0f}m")
            
            measurements_summary = "; ".join(measurements) if measurements else "No measurements available"
            
            # Generate response
            response = self.response_chain.invoke({
                "query": user_query,
                "context": context,
                "num_profiles": num_profiles,
                "date_range": time_range,
                "geographic_range": geographic_range,
                "measurements_summary": measurements_summary
            })
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Found {len(results_df)} measurements from ARGO floats. Analysis completed successfully."

class FloatChatRAG:
    """Main RAG system combining vector retrieval and LangChain processing"""
    
    def __init__(self, processed_folder: str, vector_store, use_openai: bool = True, openai_api_key: str = None):
        self.vector_store = vector_store
        self.query_processor = ArgoQueryProcessor(
            processed_folder, 
            use_openai=use_openai, 
            openai_api_key=openai_api_key
        )
    
    def query(self, user_query: str, top_k: int = 10) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
        """Complete RAG pipeline for ARGO data queries"""
        try:
            # Step 1: Vector retrieval for context
            retrieved_docs = self.vector_store.retrieve(user_query, top_k=top_k)
            context = "\n".join([doc['summary'] for doc in retrieved_docs[:5]])  # Use top 5 for context
            
            # Step 2: Parse query with LangChain
            structured_query = self.query_processor.parse_query(user_query)
            
            # Step 3: Execute structured query
            results_df = self.query_processor.execute_structured_query(structured_query)
            
            # Step 4: Generate response
            response = self.query_processor.generate_response(user_query, context, results_df)
            
            return structured_query, results_df, response
            
        except Exception as e:
            print(f"Error in RAG query: {e}")
            return {}, pd.DataFrame(), f"Error processing query: {str(e)}"

# Usage example
def setup_floatchat_rag(processed_folder: str, openai_api_key: str = None):
    """Setup complete FloatChat RAG system"""
    from floatchat_pipeline import ArgoVectorStore  # Import from previous artifact
    
    # Load vector store
    vector_store = ArgoVectorStore(processed_folder)
    if not vector_store.load_index():
        print("Vector index not found. Please run the data processing pipeline first.")
        return None
    
    # Create RAG system
    rag_system = FloatChatRAG(
        processed_folder=processed_folder,
        vector_store=vector_store,
        use_openai=bool(openai_api_key),
        openai_api_key=openai_api_key
    )
    
    return rag_system

# Test function
def test_rag_system():
    """Test the RAG system with sample queries"""
    
    # Setup (you'll need to provide your OpenAI API key)
    rag = setup_floatchat_rag("./argo_data/processed", openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    if rag is None:
        print("Could not initialize RAG system")
        return
    
    # Test queries
    test_queries = [
        "Show me salinity profiles near the equator",
        "Find temperature measurements in the Indian Ocean during 2019",
        "What are the deepest measurements available?",
        "Compare salinity between different regions"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        structured_query, results_df, response = rag.query(query)
        
        print(f"Structured Query: {structured_query}")
        print(f"Results: {len(results_df)} measurements")
        print(f"Response: {response}")

if __name__ == "__main__":
    test_rag_system()