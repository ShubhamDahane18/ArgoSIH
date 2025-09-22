import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import numpy as np
from sqlalchemy import create_engine

# Page config
st.set_page_config(
    page_title="FloatChat - ARGO Ocean Data Explorer", 
    page_icon="🌊",
    layout="wide"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Database connection
@st.cache_resource
def init_database():
    engine = create_engine("postgresql+psycopg2://postgres:Sonali%4018@localhost:5432/argo_db")
    return engine

# Load data with caching
@st.cache_data
def load_data():
    try:
        return pd.read_parquet("indian_ocean_enhanced.parquet")
    except:
        st.error("Data file not found. Please run the data processing notebook first.")
        return None

# Natural Language to SQL converter
class NLToSQL:
    def __init__(self, df):
        self.df = df
        self.schema_info = self._get_schema_info()
    
    def _get_schema_info(self):
        return {
            'columns': list(self.df.columns),
            'date_range': f"{self.df['date'].min()} to {self.df['date'].max()}",
            'lat_range': f"{self.df['latitude'].min()} to {self.df['latitude'].max()}",
            'lon_range': f"{self.df['longitude'].min()} to {self.df['longitude'].max()}",
            'regions': list(self.df['region'].unique()),
            'seasons': list(self.df['season'].unique()),
            'institutions': list(self.df['institution'].unique()[:10])
        }
    
    def parse_query(self, query):
        """Simple rule-based NL to filter converter"""
        query_lower = query.lower()
        filters = {}
        
        # Parse time filters
        if 'march' in query_lower:
            filters['month'] = 3
        elif 'summer' in query_lower:
            filters['season'] = 'Summer'
        elif 'winter' in query_lower:
            filters['season'] = 'Winter'
        
        # Parse location filters
        if 'equator' in query_lower or 'equatorial' in query_lower:
            filters['lat_range'] = (-5, 5)
        elif 'northern' in query_lower:
            filters['region'] = 'Northern Indian Ocean'
        elif 'southern' in query_lower:
            filters['region'] = 'Southern Indian Ocean'
        elif 'tropical' in query_lower:
            filters['region'] = 'Tropical Indian Ocean'
        
        # Parse year filters
        for year in range(2015, 2026):
            if str(year) in query_lower:
                filters['year'] = year
                break
        
        return filters
    
    def execute_query(self, query):
        """Execute the parsed query on dataframe"""
        filters = self.parse_query(query)
        result_df = self.df.copy()
        
        for key, value in filters.items():
            if key == 'lat_range':
                result_df = result_df[
                    (result_df['latitude'] >= value[0]) & 
                    (result_df['latitude'] <= value[1])
                ]
            elif key in result_df.columns:
                result_df = result_df[result_df[key] == value]
        
        return result_df, filters

# Chat interface
def chat_interface():
    st.subheader("🤖 FloatChat - Ask about ARGO data")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    nl_sql = NLToSQL(df)
    
    # Chat input
    user_query = st.chat_input("Ask me about ARGO floats data (e.g., 'Show me data near equator in March 2023')")
    
    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Process query
        result_df, filters = nl_sql.execute_query(user_query)
        
        # Generate response
        response = f"Found {len(result_df):,} profiles"
        if filters:
            response += f" with filters: {filters}"
        
        st.session_state.chat_history.append({"role": "assistant", "content": response, "data": result_df})
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if "data" in message and len(message["data"]) > 0:
                    # Show map visualization
                    fig = px.scatter_mapbox(
                        message["data"].sample(min(1000, len(message["data"]))),
                        lat="latitude", lon="longitude",
                        color="year",
                        hover_data=["wmo", "institution"],
                        mapbox_style="open-street-map",
                        zoom=2,
                        title=f"ARGO Float Locations ({len(message['data']):,} profiles)"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

# Main dashboard
def main_dashboard():
    df = load_data()
    if df is None:
        return
    
    # Sidebar filters
    st.sidebar.header("📍 Data Filters")
    
    # Year filter
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(2020, int(df['year'].max())),
        step=1
    )
    
    # Region filter
    regions = st.sidebar.multiselect(
        "Regions",
        options=df['region'].unique(),
        default=df['region'].unique()
    )
    
    # Institution filter
    top_institutions = df['institution'].value_counts().head(10).index.tolist()
    institutions = st.sidebar.multiselect(
        "Institutions",
        options=top_institutions,
        default=top_institutions[:3]
    )
    
    # Apply filters
    filtered_df = df[
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1]) &
        (df['region'].isin(regions)) &
        (df['institution'].isin(institutions))
    ]
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Profiles", f"{len(filtered_df):,}")
    with col2:
        st.metric("Active Floats", f"{filtered_df['wmo'].nunique():,}")
    with col3:
        st.metric("Institutions", f"{filtered_df['institution'].nunique()}")
    with col4:
        st.metric("Countries", f"{len(filtered_df['dac'].unique())}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Temporal distribution
        temporal_data = filtered_df.groupby('year').size().reset_index(name='profiles')
        fig1 = px.line(temporal_data, x='year', y='profiles', 
                      title="Profiles per Year",
                      markers=True)
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Regional distribution
        region_data = filtered_df['region'].value_counts().reset_index()
        fig3 = px.pie(region_data, names='region', values='count',
                     title="Regional Distribution")
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Geographic distribution (sample for performance)
        sample_df = filtered_df.sample(min(5000, len(filtered_df)))
        fig2 = px.scatter_mapbox(
            sample_df,
            lat="latitude", lon="longitude",
            color="region",
            hover_data=["wmo", "year", "institution"],
            mapbox_style="open-street-map",
            zoom=2,
            title=f"ARGO Float Locations (Sample of {len(sample_df):,})"
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Institutional contribution
        inst_data = filtered_df['institution'].value_counts().head(8).reset_index()
        fig4 = px.bar(inst_data, x='count', y='institution',
                     title="Top Institutions by Profile Count",
                     orientation='h')
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Seasonal analysis
    st.subheader("🌦️ Seasonal Analysis")
    seasonal_region = pd.crosstab(filtered_df['season'], filtered_df['region'])
    fig5 = px.imshow(seasonal_region.values,
                    x=seasonal_region.columns,
                    y=seasonal_region.index,
                    color_continuous_scale="Blues",
                    title="Profile Distribution by Season and Region")
    fig5.update_layout(height=300)
    st.plotly_chart(fig5, use_container_width=True)

# Data explorer
def data_explorer():
    df = load_data()
    if df is None:
        return
    
    st.subheader("🔍 Data Explorer")
    
    # Sample data viewer
    st.write("Sample Data:")
    st.dataframe(df.head(100))
    
    # Data summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Summary**")
        st.write(f"- Total Profiles: {len(df):,}")
        st.write(f"- Unique Floats: {df['wmo'].nunique():,}")
        st.write(f"- Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        st.write(f"- Geographic Coverage: {df['latitude'].min():.1f}°N to {df['latitude'].max():.1f}°N")
        
    with col2:
        st.write("**Missing Values**")
        missing_data = df.isnull().sum()
        for col, missing in missing_data.items():
            if missing > 0:
                st.write(f"- {col}: {missing:,} ({missing/len(df)*100:.2f}%)")
        if missing_data.sum() == 0:
            st.write("✅ No missing values!")

# Main application
def main():
    st.title("🌊 FloatChat - ARGO Ocean Data Explorer")
    st.markdown("*AI-Powered Conversational Interface for Ocean Data Discovery*")
    
    # Navigation
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Chat Interface", "🔍 Data Explorer"])
    
    with tab1:
        main_dashboard()
    
    with tab2:
        chat_interface()
    
    with tab3:
        data_explorer()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**FloatChat v1.0**")
    st.sidebar.markdown("Built for SIH 2025")
    st.sidebar.markdown("ARGO Float Data Analysis")

if __name__ == "__main__":
    main()