import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules (these would be the previous artifacts)
try:
    from floatchat_pipeline import ArgoDataProcessor, ArgoVectorStore
    from floatchat_langchain import FloatChatRAG, setup_floatchat_rag
except ImportError:
    st.error("Please ensure floatchat_pipeline.py and floatchat_langchain.py are in your Python path")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="FloatChat - ARGO Ocean Data Assistant",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         color: #2c3e50;
#         margin: 1rem 0;
#     }
#     .metric-container {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #1f77b4;
#     }
#     .chat-message {
#         padding: 1rem;
#         margin: 0.5rem 0;
#         border-radius: 0.5rem;
#     }
#     .user-message {
#         background-color: #e3f2fd;
#         border-left: 4px solid #2196f3;
#     }
#     .assistant-message {
#         background-color: #f1f8e9;
#         border-left: 4px solid #4caf50;
#     }
# </style>
# """, unsafe_allow_html=True)

class StreamlitFloatChat:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.base_folder = "./argo_data"
        self.processed_folder = os.path.join(self.base_folder, "processed")
        self.rag_system = None
        self._initialize_session_state()
        
        # Load RAG system from session state if available
        if 'rag_system' in st.session_state and st.session_state.rag_system:
            self.rag_system = st.session_state.rag_system
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
    
    def _auto_load_existing_data(self):
        """Automatically load existing processed data"""
        try:
            # Check if all required files exist
            required_files = [
                "argo_combined_data.parquet",
                "argo_profiles_metadata_indexed.parquet", 
                "faiss_argo_profiles.index"
            ]
            
            all_exist = all(
                os.path.exists(os.path.join(self.processed_folder, f)) 
                for f in required_files
            )
            
            if all_exist:
                # Silently load the RAG system
                rag_system = setup_floatchat_rag(
                    processed_folder=self.processed_folder,
                    openai_api_key=None  # Will use env var if available
                )
                
                if rag_system:
                    st.session_state.rag_system = rag_system
                    st.session_state.data_loaded = True
                    self.rag_system = rag_system
                    
        except Exception as e:
            # Silent fail - user can manually load
            pass
        if 'current_results' not in st.session_state:
            st.session_state.current_results = None
        if 'current_query' not in st.session_state:
            st.session_state.current_query = None
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
            
        # Auto-detect existing data and load if available
        if not st.session_state.data_loaded and os.path.exists(self.processed_folder):
            self._auto_load_existing_data()
    
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🌊 FloatChat</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Conversational Interface for ARGO Ocean Data Discovery</p>', unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with configuration options"""
        st.sidebar.title("⚙️ Configuration")
        
        # API Key input
        openai_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for advanced natural language processing"
        )
        
        # Data management section
        st.sidebar.markdown("### 📊 Data Management")
        
        # Check if data exists
        data_exists = os.path.exists(self.processed_folder)
        
        if data_exists:
            # Check if data is already loaded
            if st.session_state.data_loaded:
                st.sidebar.success("✅ Data loaded and ready")
                st.sidebar.info(f"🔍 Vector index: {len(st.session_state.rag_system.vector_store.metadata_df) if st.session_state.rag_system else 0} profiles")
            else:
                st.sidebar.success("✅ Processed data found")
                
                # Load existing data
                if st.sidebar.button("🔄 Load Data", key="load_data_btn"):
                    self._load_rag_system(openai_key)
        else:
            st.sidebar.warning("⚠️ No processed data found")
            
            # Data download section
            st.sidebar.markdown("### 📥 Download Data")
            
            data_url = st.sidebar.text_input(
                "ARGO Data URL",
                value="https://www.ncei.noaa.gov/data/oceans/argo/gadr/data/indian/2019/01/",
                help="URL to ARGO NetCDF files"
            )
            
            max_files = st.sidebar.slider(
                "Max Files to Download",
                min_value=5,
                max_value=100,
                value=20,
                help="Limit files for demo purposes"
            )
            
            if st.sidebar.button("📥 Download & Process Data"):
                self._download_and_process_data(data_url, max_files, openai_key)
        
        # Advanced options
        st.sidebar.markdown("### 🎛️ Advanced Options")
        
        retrieval_k = st.sidebar.slider(
            "Retrieval Top-K",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of similar profiles to retrieve"
        )
        
        max_results = st.sidebar.slider(
            "Max Results",
            min_value=50,
            max_value=500,
            value=100,
            help="Maximum number of measurement records to return"
        )
        
        return {
            'openai_key': openai_key,
            'retrieval_k': retrieval_k,
            'max_results': max_results
        }
    
    def _download_and_process_data(self, data_url: str, max_files: int, openai_key: str):
        """Download and process ARGO data"""
        with st.spinner("Downloading and processing ARGO data..."):
            try:
                # Initialize processor
                processor = ArgoDataProcessor(base_folder=self.base_folder)
                
                # Download data
                st.info(f"📥 Downloading up to {max_files} NetCDF files...")
                downloaded = processor.download_argo_files(data_url, max_files=max_files)
                
                if downloaded > 0:
                    st.success(f"✅ Downloaded {downloaded} files")
                    
                    # Process NetCDF files
                    st.info("🔄 Processing NetCDF files...")
                    df = processor.process_netcdf_files()
                    
                    if not df.empty:
                        st.success(f"✅ Processed {len(df)} measurement records")
                        
                        # Create metadata
                        st.info("📋 Creating profile metadata...")
                        metadata_df = processor.create_profile_metadata(df)
                        st.success(f"✅ Created metadata for {len(metadata_df)} profiles")
                        
                        # Build vector index
                        st.info("🔍 Building vector search index...")
                        vector_store = ArgoVectorStore(processor.processed_folder)
                        vector_store.build_vector_index(metadata_df)
                        st.success("✅ Vector index created")
                        
                        # Initialize RAG system
                        self._load_rag_system(openai_key)
                        
                        st.success("🎉 Data processing completed successfully!")
                        st.balloons()
                    else:
                        st.error("❌ No data was processed successfully")
                else:
                    st.error("❌ No files were downloaded")
                    
            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")
    
    def _load_rag_system(self, openai_key: str):
        """Load the RAG system"""
        try:
            with st.spinner("Loading RAG system..."):
                self.rag_system = setup_floatchat_rag(
                    processed_folder=self.processed_folder,
                    openai_api_key=openai_key if openai_key else None
                )
                
                if self.rag_system:
                    st.session_state.data_loaded = True
                    st.session_state.rag_system = self.rag_system  # Store in session state
                    st.success("✅ RAG system loaded successfully!")
                    st.rerun()  # Force refresh to update UI
                else:
                    st.error("❌ Could not load RAG system")
                    st.session_state.data_loaded = False
                    
        except Exception as e:
            st.error(f"❌ Error loading RAG system: {str(e)}")
            st.session_state.data_loaded = False
    
    def render_chat_interface(self, config):
        """Render the main chat interface"""
        st.markdown('<h2 class="sub-header">💬 Chat with Your Ocean Data</h2>', unsafe_allow_html=True)
        
        # More robust data loading check
        data_ready = (
            st.session_state.data_loaded and 
            st.session_state.rag_system is not None and
            self.rag_system is not None
        )
        
        if not data_ready:
            st.warning("⚠️ Please load or process data first using the sidebar.")
            
            # Debug information
            with st.expander("🔧 Debug Information"):
                st.write(f"data_loaded: {st.session_state.data_loaded}")
                st.write(f"session rag_system: {st.session_state.rag_system is not None}")
                st.write(f"instance rag_system: {self.rag_system is not None}")
                st.write(f"processed folder exists: {os.path.exists(self.processed_folder)}")
                
                if os.path.exists(self.processed_folder):
                    files = os.listdir(self.processed_folder)
                    st.write(f"Files in processed folder: {files}")
                    
                # Manual load button
                if st.button("🔄 Force Load Data", key="force_load"):
                    openai_key = st.text_input("Enter OpenAI Key (optional)", type="password", key="debug_openai")
                    self._load_rag_system(openai_key)
            return
        
        # Sample queries
        st.markdown("### 💡 Sample Queries")
        sample_queries = [
            "Show me salinity profiles near the equator",
            "Find temperature measurements deeper than 500 meters",
            "What floats were active in the Arabian Sea?",
            "Compare salinity between different regions",
            "Show me the warmest water temperatures found"
        ]
        
        cols = st.columns(len(sample_queries))
        for i, query in enumerate(sample_queries):
            if cols[i].button(f"💬 {query[:20]}...", key=f"sample_{i}"):
                self._process_query(query, config)
        
        st.markdown("---")
        
        # Chat input
        user_query = st.text_input(
            "Ask about ARGO ocean data:",
            placeholder="e.g., Show me temperature profiles in the Indian Ocean during March 2019",
            key="chat_input"
        )
        
        col1, col2 = st.columns([1, 4])
        
        if col1.button("🚀 Submit Query", disabled=not user_query):
            self._process_query(user_query, config)
        
        if col2.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.current_results = None
            st.rerun()
        
        # Display chat history
        self._display_chat_history()
    
    def _process_query(self, query: str, config):
        """Process user query using RAG system"""
        # Ensure we have the RAG system
        if not self.rag_system and st.session_state.rag_system:
            self.rag_system = st.session_state.rag_system
            
        if not self.rag_system:
            st.error("❌ RAG system not loaded. Please load data first.")
            return
        
        with st.spinner("🔍 Analyzing your query..."):
            try:
                # Execute RAG query
                structured_query, results_df, response = self.rag_system.query(
                    query, 
                    top_k=config['retrieval_k']
                )
                
                # Store in session state
                st.session_state.current_results = {
                    'query': query,
                    'structured_query': structured_query,
                    'results_df': results_df,
                    'response': response
                }
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'user': query,
                    'assistant': response,
                    'results_count': len(results_df),
                    'timestamp': datetime.now()
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                # Show detailed error for debugging
                with st.expander("🔧 Error Details"):
                    st.code(str(e))
                    import traceback
                    st.code(traceback.format_exc())
    
    def _display_chat_history(self):
        """Display chat conversation history"""
        if st.session_state.chat_history:
            st.markdown("### 📝 Conversation History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    # User message
                    st.markdown(
                        f'<div class="chat-message user-message">'
                        f'<strong>You:</strong> {chat["user"]}'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                    
                    # Assistant message
                    st.markdown(
                        f'<div class="chat-message assistant-message">'
                        f'<strong>FloatChat:</strong> {chat["assistant"]}<br>'
                        f'<small>Found {chat["results_count"]} measurements • {chat["timestamp"].strftime("%H:%M:%S")}</small>'
                        f'</div>', 
                        unsafe_allow_html=True
                    )
                    
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("---")
    
    def render_visualization_panel(self):
        """Render data visualization panel"""
        if not st.session_state.current_results:
            st.info("💡 Submit a query to see visualizations")
            return
        
        results = st.session_state.current_results
        results_df = results['results_df']
        
        if results_df.empty:
            st.warning("No data to visualize")
            return
        
        st.markdown('<h2 class="sub-header">📊 Data Visualization</h2>', unsafe_allow_html=True)
        
        # Data summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Measurements", 
                f"{len(results_df):,}",
                help="Number of individual measurements"
            )
        
        with col2:
            n_profiles = len(results_df.groupby(['float_id', 'cycle']))
            st.metric(
                "Profiles", 
                f"{n_profiles:,}",
                help="Number of distinct float profiles"
            )
        
        with col3:
            n_floats = results_df['float_id'].nunique()
            st.metric(
                "Unique Floats", 
                f"{n_floats}",
                help="Number of different ARGO floats"
            )
        
        with col4:
            if 'time' in results_df.columns:
                time_span = (results_df['time'].max() - results_df['time'].min()).days
                st.metric(
                    "Time Span", 
                    f"{time_span} days",
                    help="Temporal coverage of data"
                )
        
        st.markdown("---")
        
        # Visualization tabs
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "🌍 Geographic Map", 
            "📈 Profile Plots", 
            "📊 Statistical Analysis",
            "📋 Data Table"
        ])
        
        with viz_tab1:
            self._render_geographic_map(results_df)
        
        with viz_tab2:
            self._render_profile_plots(results_df)
        
        with viz_tab3:
            self._render_statistical_analysis(results_df)
        
        with viz_tab4:
            self._render_data_table(results_df)
    
    def _render_geographic_map(self, df):
        """Render geographic map of float locations"""
        st.markdown("#### 🌍 Float Locations")
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            st.warning("No geographic data available")
            return
        
        # Create map data
        map_data = df.groupby(['float_id', 'cycle']).agg({
            'latitude': 'first',
            'longitude': 'first',
            'time': 'first',
            'pressure': 'count'  # Number of measurements per profile
        }).reset_index()
        
        map_data = map_data.rename(columns={'pressure': 'measurement_count'})
        
        # Interactive map with Folium
        center_lat = map_data['latitude'].mean()
        center_lon = map_data['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Add markers
        for _, row in map_data.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                popup=f"Float {row['float_id']} Cycle {row['cycle']}<br>"
                      f"Measurements: {row['measurement_count']}<br>"
                      f"Time: {row['time'].strftime('%Y-%m-%d') if pd.notna(row['time']) else 'Unknown'}",
                tooltip=f"Float {row['float_id']}",
                fillColor='blue',
                color='darkblue',
                weight=2,
                fillOpacity=0.7
            ).add_to(m)
        
        # Display map
        map_data_display = st_folium(m, width=700, height=500)
        
        # Summary statistics
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Geographic Coverage:**")
            st.write(f"• Latitude: {df['latitude'].min():.2f}° to {df['latitude'].max():.2f}°")
            st.write(f"• Longitude: {df['longitude'].min():.2f}° to {df['longitude'].max():.2f}°")
        
        with col2:
            st.write("**Profile Distribution:**")
            st.write(f"• Total Profiles: {len(map_data)}")
            st.write(f"• Avg Measurements/Profile: {map_data['measurement_count'].mean():.1f}")
    
    def _render_profile_plots(self, df):
        """Render ocean profile plots"""
        st.markdown("#### 📈 Ocean Profiles")
        
        available_vars = [col for col in ['temperature', 'salinity'] if col in df.columns]
        
        if not available_vars or 'pressure' not in df.columns:
            st.warning("Insufficient data for profile plots")
            return
        
        # Variable selection
        selected_var = st.selectbox("Select Variable", available_vars, key="profile_var")
        
        # Limit profiles for plotting
        max_profiles = 10
        profile_groups = list(df.groupby(['float_id', 'cycle']))[:max_profiles]
        
        if len(profile_groups) > max_profiles:
            st.info(f"Showing first {max_profiles} profiles out of {len(profile_groups)} total")
        
        # Create profile plot
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, ((float_id, cycle), group) in enumerate(profile_groups):
            # Sort by pressure
            group_sorted = group.sort_values('pressure')
            
            # Remove NaN values
            valid_data = group_sorted.dropna(subset=[selected_var, 'pressure'])
            
            if len(valid_data) > 0:
                fig.add_trace(go.Scatter(
                    x=valid_data[selected_var],
                    y=valid_data['pressure'],
                    mode='lines+markers',
                    name=f'Float {float_id} C{cycle}',
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=True
                ))
        
        # Customize layout
        fig.update_layout(
            title=f'{selected_var.title()} Profiles',
            xaxis_title=f'{selected_var.title()} {"(°C)" if selected_var == "temperature" else "(PSU)"}',
            yaxis_title='Pressure (dbar)',
            yaxis=dict(autorange='reversed'),  # Depth increases downward
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.markdown("**Profile Statistics:**")
        profile_stats = df.groupby(['float_id', 'cycle'])[selected_var].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        st.dataframe(profile_stats.head())
    
    def _render_statistical_analysis(self, df):
        """Render statistical analysis of the data"""
        st.markdown("#### 📊 Statistical Analysis")
        
        numeric_cols = [col for col in ['temperature', 'salinity', 'pressure'] if col in df.columns]
        
        if not numeric_cols:
            st.warning("No numeric data available for analysis")
            return
        
        # Summary statistics
        st.markdown("**Summary Statistics:**")
        summary_stats = df[numeric_cols].describe().round(2)
        st.dataframe(summary_stats)
        
        # Distribution plots
        st.markdown("**Data Distributions:**")
        
        selected_var = st.selectbox("Select Variable for Distribution", numeric_cols, key="dist_var")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                df.dropna(subset=[selected_var]), 
                x=selected_var,
                title=f'{selected_var.title()} Distribution',
                nbins=50
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot by float
            if len(df['float_id'].unique()) <= 20:  # Only show if reasonable number of floats
                fig_box = px.box(
                    df.dropna(subset=[selected_var]), 
                    y=selected_var,
                    x='float_id',
                    title=f'{selected_var.title()} by Float'
                )
                fig_box.update_layout(height=400, xaxis_tickangle=45)
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                # Overall box plot
                fig_box = px.box(
                    df.dropna(subset=[selected_var]), 
                    y=selected_var,
                    title=f'{selected_var.title()} Distribution'
                )
                fig_box.update_layout(height=400)
                st.plotly_chart(fig_box, use_container_width=True)
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            st.markdown("**Correlation Matrix:**")
            corr_matrix = df[numeric_cols].corr().round(2)
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                title="Variable Correlations"
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    def _render_data_table(self, df):
        """Render interactive data table"""
        st.markdown("#### 📋 Data Table")
        
        # Display options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_rows = st.selectbox("Rows to Display", [100, 500, 1000, "All"], index=0)
        
        with col2:
            sort_column = st.selectbox("Sort by", df.columns.tolist())
        
        with col3:
            sort_order = st.selectbox("Sort Order", ["Ascending", "Descending"])
        
        # Apply sorting
        ascending = sort_order == "Ascending"
        df_display = df.sort_values(sort_column, ascending=ascending)
        
        # Apply row limit
        if show_rows != "All":
            df_display = df_display.head(int(show_rows))
        
        # Display table
        st.dataframe(
            df_display, 
            use_container_width=True,
            height=400
        )
        
        # Download options
        st.markdown("**Download Data:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_display.to_csv(index=False)
            st.download_button(
                "📥 Download as CSV",
                csv,
                "argo_data.csv",
                "text/csv"
            )
        
        with col2:
            # Note: For Parquet download in Streamlit, you'd need to use BytesIO
            st.info("💡 Full dataset available in processed folder")
    
    def run(self):
        """Main application runner"""
        self.render_header()
        
        # Sidebar configuration
        config = self.render_sidebar()
        
        # Main content area
        if st.session_state.data_loaded:
            # Create two columns for chat and visualization
            chat_col, viz_col = st.columns([1, 1])
            
            with chat_col:
                self.render_chat_interface(config)
            
            with viz_col:
                self.render_visualization_panel()
        else:
            # Welcome screen
            st.markdown("""
            ### Welcome to FloatChat! 🌊
            
            FloatChat is an AI-powered conversational interface for exploring ARGO ocean float data. 
            
            **Getting Started:**
            1. **Configure API Key** (optional): Add your OpenAI API key in the sidebar for advanced natural language processing
            2. **Load Data**: Either load existing processed data or download new ARGO data
            3. **Start Chatting**: Ask questions about ocean data in natural language
            
            **What you can do:**
            - 🔍 Search for specific ocean measurements
            - 📊 Visualize temperature and salinity profiles  
            - 🗺️ Explore geographic distribution of floats
            - 📈 Analyze statistical patterns in the data
            - 💬 Ask questions in natural language
            
            **Sample Questions:**
            - "Show me salinity profiles near the equator"
            - "What's the temperature range in the Indian Ocean?"
            - "Find the deepest measurements available"
            - "Compare different regions or time periods"
            """)

# Main application entry point
def main():
    app = StreamlitFloatChat()
    app.run()

if __name__ == "__main__":
    main()