import requests
import os
import xarray as xr
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import glob
from sentence_transformers import SentenceTransformer
import faiss
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class ArgoDataProcessor:
    """Complete pipeline for processing ARGO NetCDF data"""
    
    def __init__(self, base_folder: str = "./argo_data"):
        self.base_folder = base_folder
        self.data_folder = os.path.join(base_folder, "raw")
        self.processed_folder = os.path.join(base_folder, "processed")
        os.makedirs(self.data_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        
    def download_argo_files(self, url: str, max_files: int = 50):
        """Download ARGO NetCDF files from NOAA server"""
        print(f"Fetching file list from: {url}")
        
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            files = [node.get("href") for node in soup.find_all("a") 
                    if node.get("href") and node.get("href").endswith(".nc")]
            
            # Limit files for demo
            files = files[:max_files]
            print(f"Found {len(files)} NetCDF files to download")
            
            def download_file(f):
                try:
                    file_path = os.path.join(self.data_folder, f)
                    if os.path.exists(file_path):
                        print(f"Skipping existing file: {f}")
                        return
                        
                    r_file = requests.get(url + f, timeout=60)
                    r_file.raise_for_status()
                    with open(file_path, "wb") as file:
                        file.write(r_file.content)
                    print(f"Downloaded: {f}")
                except Exception as e:
                    print(f"Error downloading {f}: {e}")
            
            # Download in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                executor.map(download_file, files)
                
            print("Download completed!")
            return len(files)
            
        except Exception as e:
            print(f"Error in download process: {e}")
            return 0
    
    def process_netcdf_files(self) -> pd.DataFrame:
        """Process all NetCDF files and convert to structured DataFrame"""
        all_files = glob.glob(os.path.join(self.data_folder, "*.nc"))
        
        if not all_files:
            print("No NetCDF files found. Please download data first.")
            return pd.DataFrame()
            
        print(f"Processing {len(all_files)} NetCDF files...")
        all_dfs = []
        
        for file_path in all_files:
            try:
                df = self._process_single_nc(file_path)
                if not df.empty:
                    all_dfs.append(df)
                    print(f"Processed: {os.path.basename(file_path)} - {len(df)} records")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            
            # Clean and standardize data
            combined_df = self._clean_dataframe(combined_df)
            
            # Save processed data
            output_file = os.path.join(self.processed_folder, "argo_combined_data.parquet")
            combined_df.to_parquet(output_file, index=False)
            print(f"Saved combined data to: {output_file}")
            print(f"Total records: {len(combined_df)}")
            
            return combined_df
        else:
            print("No data was successfully processed.")
            return pd.DataFrame()
    
    def _process_single_nc(self, file_path: str) -> pd.DataFrame:
        """Process a single NetCDF file"""
        ds = xr.open_dataset(file_path)
        dfs = []
        
        if 'n_levels' not in ds.sizes or 'n_prof' not in ds.sizes:
            print(f"Skipping file with unexpected structure: {file_path}")
            return pd.DataFrame()
        
        n_levels = ds.sizes['n_levels']
        n_prof = ds.sizes['n_prof']
        
        for p in range(n_prof):
            try:
                # Extract profile metadata
                lat = float(ds['latitude'].values[p]) if 'latitude' in ds else np.nan
                lon = float(ds['longitude'].values[p]) if 'longitude' in ds else np.nan
                
                # Handle time data
                if 'juld' in ds:
                    time_val = ds['juld'].values[p]
                    if pd.notna(time_val):
                        juld = pd.Timestamp(time_val)
                    else:
                        juld = pd.NaT
                else:
                    juld = pd.NaT
                
                # Platform information
                if 'platform_number' in ds:
                    float_id = str(ds['platform_number'].values[p]).strip()
                else:
                    float_id = f"unknown_{os.path.basename(file_path)}"
                
                cycle = int(ds['cycle_number'].values[p]) if 'cycle_number' in ds else 0
                
                # Profile measurements
                pres = ds['pres'].values[p, :].flatten() if 'pres' in ds else np.full(n_levels, np.nan)
                temp = ds['temp'].values[p, :].flatten() if 'temp' in ds else np.full(n_levels, np.nan)
                psal = ds['psal'].values[p, :].flatten() if 'psal' in ds else np.full(n_levels, np.nan)
                
                # Create DataFrame for this profile
                df = pd.DataFrame({
                    "float_id": [float_id] * n_levels,
                    "cycle": [cycle] * n_levels,
                    "latitude": [lat] * n_levels,
                    "longitude": [lon] * n_levels,
                    "time": [juld] * n_levels,
                    "pressure": pres,
                    "temperature": temp,
                    "salinity": psal,
                    "file_source": [os.path.basename(file_path)] * n_levels
                })
                
                dfs.append(df)
                
            except Exception as e:
                print(f"Error processing profile {p} in {file_path}: {e}")
                continue
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the DataFrame"""
        # Clean float_id
        df['float_id'] = df['float_id'].astype(str).str.strip()
        df['float_id'] = df['float_id'].str.replace("b'", "").str.replace("'", "")
        
        # Remove invalid data
        df = df.dropna(subset=['latitude', 'longitude'])
        df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
        
        # Convert data types
        numeric_cols = ['pressure', 'temperature', 'salinity']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def create_profile_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create metadata summary for each profile"""
        print("Creating profile metadata...")
        
        def make_profile_doc(group):
            """Create metadata document for a profile group"""
            n_levels = len(group)
            lat = float(group['latitude'].iloc[0])
            lon = float(group['longitude'].iloc[0])
            time = group['time'].iloc[0]
            
            # Calculate ranges with null handling
            def safe_range(series, name):
                valid_data = series.dropna()
                if len(valid_data) > 0:
                    return float(valid_data.min()), float(valid_data.max())
                return None, None
            
            p_min, p_max = safe_range(group['pressure'], 'pressure')
            t_min, t_max = safe_range(group['temperature'], 'temperature')
            s_min, s_max = safe_range(group['salinity'], 'salinity')
            
            # Create summary text
            time_str = time.strftime('%Y-%m-%d') if pd.notna(time) else 'unknown'
            summary = (
                f"Float {group['float_id'].iloc[0]} cycle {group['cycle'].iloc[0]} "
                f"at ({lat:.3f}, {lon:.3f}) on {time_str} with {n_levels} levels. "
                f"Pressure: {p_min}-{p_max} dbar, Temperature: {t_min}-{t_max}°C, "
                f"Salinity: {s_min}-{s_max} PSU"
            )
            
            return {
                "float_id": group['float_id'].iloc[0],
                "cycle": int(group['cycle'].iloc[0]),
                "latitude": lat,
                "longitude": lon,
                "time": pd.to_datetime(time),
                "n_levels": int(n_levels),
                "p_min": p_min, "p_max": p_max,
                "t_min": t_min, "t_max": t_max,
                "s_min": s_min, "s_max": s_max,
                "summary": summary,
                "file_source": group['file_source'].iloc[0]
            }
        
        # Group by profile and create metadata
        grouped = df.groupby(['float_id', 'cycle'])
        metadata_list = []
        
        for (float_id, cycle), group in grouped:
            try:
                metadata_list.append(make_profile_doc(group))
            except Exception as e:
                print(f"Error creating metadata for {float_id} cycle {cycle}: {e}")
        
        metadata_df = pd.DataFrame(metadata_list)
        
        # Save metadata
        metadata_file = os.path.join(self.processed_folder, "argo_profiles_metadata.parquet")
        metadata_df.to_parquet(metadata_file, index=False)
        print(f"Created metadata for {len(metadata_df)} profiles")
        
        return metadata_df

class ArgoVectorStore:
    """Vector database for ARGO profile metadata"""
    
    def __init__(self, processed_folder: str):
        self.processed_folder = processed_folder
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.metadata_df = None
        
    def build_vector_index(self, metadata_df: pd.DataFrame):
        """Build FAISS vector index from profile metadata"""
        print("Building vector index...")
        
        self.metadata_df = metadata_df
        texts = metadata_df['summary'].astype(str).tolist()
        
        # Generate embeddings in batches
        batch_size = 64
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch_texts, 
                show_progress_bar=True, 
                convert_to_numpy=True
            )
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Save index and metadata
        index_file = os.path.join(self.processed_folder, "faiss_argo_profiles.index")
        metadata_file = os.path.join(self.processed_folder, "argo_profiles_metadata_indexed.parquet")
        
        faiss.write_index(self.index, index_file)
        self.metadata_df.to_parquet(metadata_file, index=False)
        
        print(f"Built vector index with {len(embeddings)} profiles")
        
    def load_index(self):
        """Load existing vector index and metadata"""
        index_file = os.path.join(self.processed_folder, "faiss_argo_profiles.index")
        metadata_file = os.path.join(self.processed_folder, "argo_profiles_metadata_indexed.parquet")
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            self.index = faiss.read_index(index_file)
            self.metadata_df = pd.read_parquet(metadata_file)
            print(f"Loaded vector index with {len(self.metadata_df)} profiles")
            return True
        return False
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """Retrieve relevant profiles using vector search"""
        if self.index is None or self.metadata_df is None:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.metadata_df):
                record = self.metadata_df.iloc[idx].to_dict()
                record['similarity_score'] = float(score)
                results.append(record)
        
        return results

# Usage example and main pipeline
def main():
    """Main processing pipeline"""
    
    # Initialize processor
    processor = ArgoDataProcessor(base_folder="./argo_data")
    
    # Download sample data (Indian Ocean 2019/01)
    url = "https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/indian/2019/01/"
    downloaded = processor.download_argo_files(url, max_files=10)  # Limit for demo
    
    if downloaded > 0:
        # Process NetCDF files
        df = processor.process_netcdf_files()
        
        if not df.empty:
            # Create metadata
            metadata_df = processor.create_profile_metadata(df)
            
            # Build vector index
            vector_store = ArgoVectorStore(processor.processed_folder)
            vector_store.build_vector_index(metadata_df)
            
            print("Pipeline completed successfully!")
            print(f"Processed {len(df)} measurement records")
            print(f"Created {len(metadata_df)} profile summaries")
            
            # Test retrieval
            results = vector_store.retrieve("temperature profiles near equator", top_k=5)
            print(f"\nSample retrieval results: {len(results)} profiles found")
            for i, result in enumerate(results[:2]):
                print(f"{i+1}. {result['summary']} (score: {result['similarity_score']:.3f})")
        
        else:
            print("No data was processed successfully.")
    else:
        print("No files were downloaded.")

if __name__ == "__main__":
    main()