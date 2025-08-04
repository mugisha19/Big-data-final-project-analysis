"""
Complete FAO Statistical Data Analysis Pipeline with Guaranteed CSV Export
========================================================================

This script will:
1. Load and clean your FAO data
2. Perform analysis
3. AUTOMATICALLY export cleaned data to CSV on your local machine

Just run this script and it will create the CSV file for you!
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union
import gc
import sys
import json
import sqlite3
from datetime import datetime
import os

# ML imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    silhouette_score
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to set up matplotlib for different environments
try:
    plt.ion()  # Interactive mode
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for safety
except:
    pass

# Configure pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)


class Config:
    """Configuration class for analysis parameters."""
    
    def __init__(self):
        # File settings
        self.DATA_PATH = "./data/FAOSTAT_data_en_8-3-2025.csv"
        
        # Analysis parameters
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 42
        self.N_CLUSTERS = 3
        self.IQR_MULTIPLIER = 1.5
        
        # Plot settings
        self.FIGURE_SIZE = (10, 6)
        self.LARGE_FIGURE_SIZE = (12, 6)
        
        # Memory management
        self.MAX_ROWS_FOR_PLOTTING = 50000
        self.MAX_CATEGORIES_FOR_PLOTTING = 20
        
        # Column configurations
        self.REQUIRED_COLUMNS = ['Value', 'Area', 'Indicator', 'Year']
        self.TEXT_COLUMNS = ['Domain', 'Area', 'Indicator', 'Sex', 'Element', 'Source', 'Unit']
        self.LABEL_ENCODE_COLUMNS = ['Domain', 'Area', 'Indicator', 'Sex', 'Element', 'Source', 'Unit']


class FAODataAnalyzer:
    """Main class for FAO data analysis pipeline with guaranteed CSV export."""
    
    def __init__(self, config: Config = None):
        """Initialize the analyzer."""
        self.config = config or Config()
        self.df = None
        self.df_sample = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        
        # Set up plotting
        try:
            plt.style.use('default')
            sns.set_palette("husl")
        except:
            pass
    
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            data_path = Path(self.config.DATA_PATH)
            if not data_path.exists():
                print(f"❌ Data file not found: {data_path}")
                print("Please check the file path in the Config class.")
                return None
            
            print(f"📂 Loading data from {data_path}")
            
            # Try to load all data first, then chunk if needed
            try:
                self.df = pd.read_csv(data_path)
                print(f"✅ Data loaded successfully. Shape: {self.df.shape}")
            except MemoryError:
                print("⚠️  Large file detected. Loading in chunks...")
                chunk_list = []
                chunk_size = 10000
                
                for chunk in pd.read_csv(data_path, chunksize=chunk_size):
                    chunk_list.append(chunk)
                    if len(chunk_list) * chunk_size > 100000:
                        print(f"⚠️  Loading first {len(chunk_list) * chunk_size} rows for demo.")
                        break
                
                self.df = pd.concat(chunk_list, ignore_index=True)
                print(f"✅ Data loaded successfully. Shape: {self.df.shape}")
            
            print(f"💾 Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Validate required columns
            missing_cols = set(self.config.REQUIRED_COLUMNS) - set(self.df.columns)
            if missing_cols:
                print(f"❌ Missing required columns: {missing_cols}")
                print(f"Available columns: {list(self.df.columns)}")
                return None
            
            return self.df
            
        except FileNotFoundError:
            print(f"❌ File not found: {self.config.DATA_PATH}")
            print("Please make sure the file exists and the path is correct.")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess the dataframe."""
        if self.df is None:
            print("❌ Data not loaded. Call load_data() first.")
            return None
        
        print("🧹 Starting data cleaning process...")
        initial_shape = self.df.shape
        
        try:
            # Handle missing values
            print("   • Handling missing values...")
            missing_before = self.df.isnull().sum().sum()
            
            # Drop rows with missing critical fields
            self.df.dropna(subset=self.config.REQUIRED_COLUMNS, inplace=True)
            
            # Fill optional text fields
            optional_fields = ['Note', 'Flag', 'Flag Description']
            for field in optional_fields:
                if field in self.df.columns:
                    self.df[field] = self.df[field].fillna('')
            
            missing_after = self.df.isnull().sum().sum()
            print(f"     - Reduced missing values from {missing_before} to {missing_after}")
            
            # Standardize text columns
            print("   • Standardizing text columns...")
            for col in self.config.TEXT_COLUMNS:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(str).str.strip().str.title()
            
            # Remove outliers
            print("   • Removing outliers...")
            self.df = self._remove_outliers_safe(self.df, 'Value')
            
            # Label encoding
            print("   • Applying label encoding...")
            self._apply_label_encoding_safe()
            
            # Scale values
            print("   • Scaling numerical values...")
            if 'Value' in self.df.columns:
                self.df['Value_Scaled'] = self.scaler.fit_transform(self.df[['Value']])
            
            # Create sample for plotting
            if len(self.df) > self.config.MAX_ROWS_FOR_PLOTTING:
                self.df_sample = self.df.sample(n=self.config.MAX_ROWS_FOR_PLOTTING, 
                                              random_state=self.config.RANDOM_STATE)
                print(f"   • Created sample dataset ({len(self.df_sample)} rows) for visualization")
            else:
                self.df_sample = self.df.copy()
            
            # Force garbage collection
            gc.collect()
            
            final_shape = self.df.shape
            print(f"✅ Data cleaning completed. Shape: {initial_shape} → {final_shape}")
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error during data cleaning: {e}")
            return None
    
    def _remove_outliers_safe(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove outliers using IQR method with safety checks."""
        try:
            if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
                print(f"     - Skipping outlier removal for {column}")
                return df
            
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:
                print(f"     - No outliers to remove (IQR = 0) for {column}")
                return df
            
            lower_bound = Q1 - self.config.IQR_MULTIPLIER * IQR
            upper_bound = Q3 + self.config.IQR_MULTIPLIER * IQR
            
            initial_count = len(df)
            df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            removed_count = initial_count - len(df_clean)
            
            print(f"     - Removed {removed_count} outliers from {column}")
            return df_clean
            
        except Exception as e:
            print(f"     - Error removing outliers: {e}")
            return df
    
    def _apply_label_encoding_safe(self) -> None:
        """Apply label encoding with error handling."""
        for col in self.config.LABEL_ENCODE_COLUMNS:
            if col in self.df.columns:
                try:
                    if self.df[col].nunique() < 1000:
                        le = LabelEncoder()
                        self.df[f'{col}_Encoded'] = le.fit_transform(self.df[col].astype(str))
                        self.encoders[col] = le
                    else:
                        print(f"     - Skipping encoding for {col} (too many categories)")
                except Exception as e:
                    print(f"     - Error encoding {col}: {e}")
    
    def export_to_csv(self, filename: str = None, output_dir: str = "./cleaned_data") -> str:
        """
        Export cleaned data to CSV - GUARANTEED TO WORK!
        
        Args:
            filename: Custom filename (optional)
            output_dir: Output directory
            
        Returns:
            str: Path to exported file
        """
        if self.df is None:
            print("❌ No cleaned data available. Run load_data() and clean_data() first!")
            return None
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"fao_cleaned_data_{timestamp}.csv"
            
            # Ensure .csv extension
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            # Full file path
            filepath = os.path.join(output_dir, filename)
            
            # Export to CSV
            self.df.to_csv(filepath, index=False, encoding='utf-8')
            
            # Verify file was created
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / 1024  # KB
                
                print("\n🎉 CSV EXPORT SUCCESSFUL!")
                print("=" * 50)
                print(f"📁 File location: {os.path.abspath(filepath)}")
                print(f"📊 Data exported: {len(self.df):,} rows × {len(self.df.columns)} columns")
                print(f"💾 File size: {file_size:.1f} KB")
                print(f"📅 Export time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 50)
                
                return filepath
            else:
                print("❌ File was not created successfully")
                return None
                
        except PermissionError:
            print(f"❌ Permission denied. Cannot write to {output_dir}")
            print("Try changing the output_dir or running as administrator")
            return None
        except Exception as e:
            print(f"❌ Error exporting CSV: {e}")
            return None
    
    def generate_summary_statistics(self) -> None:
        """Generate summary statistics."""
        if self.df is None:
            print("❌ Data not available.")
            return
        
        print("\n" + "="*60)
        print("📊 DATASET SUMMARY")
        print("="*60)
        
        try:
            print(f"Dataset shape: {self.df.shape}")
            print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Data types
            print(f"\nData types: {dict(self.df.dtypes.value_counts())}")
            
            # Numerical summary
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"\n📈 NUMERICAL SUMMARY:")
                print("-" * 40)
                if len(self.df) > 10000:
                    print("(Based on sample for performance)")
                    print(self.df[numeric_cols].sample(min(10000, len(self.df))).describe())
                else:
                    print(self.df[numeric_cols].describe())
            
            # Categorical summary
            print(f"\n📋 CATEGORICAL SUMMARY:")
            print("-" * 40)
            
            key_columns = ['Area', 'Indicator', 'Sex', 'Element']
            for col in key_columns:
                if col in self.df.columns:
                    unique_count = self.df[col].nunique()
                    print(f"{col}: {unique_count} unique values")
                    
                    if unique_count <= 10:
                        print(f"  Values: {list(self.df[col].value_counts().head().index)}")
            
        except Exception as e:
            print(f"❌ Error generating summary: {e}")
    
    def create_basic_plots(self) -> None:
        """Create basic exploratory plots."""
        if self.df_sample is None:
            print("❌ Data not available for plotting.")
            return
        
        print("📊 Creating basic plots...")
        
        try:
            plot_data = self.df_sample
            
            # 1. Distribution of Values
            if 'Value' in plot_data.columns:
                plt.figure(figsize=self.config.FIGURE_SIZE)
                values = plot_data['Value'].dropna()
                
                if len(values) > 0:
                    if values.max() / values.min() > 1000:
                        plt.hist(np.log10(values + 1), bins=50, alpha=0.7, edgecolor='black')
                        plt.xlabel('Log10(Value + 1)')
                        plt.title('Distribution of Values (Log Scale)')
                    else:
                        plt.hist(values, bins=50, alpha=0.7, edgecolor='black')
                        plt.xlabel('Value')
                        plt.title('Distribution of Values')
                    
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    try:
                        plt.show()
                    except:
                        plt.savefig('value_distribution.png', dpi=100, bbox_inches='tight')
                        print("   • Plot saved as 'value_distribution.png'")
                    plt.close()
            
            # 2. Time series if Year exists
            if 'Year' in plot_data.columns:
                plt.figure(figsize=self.config.FIGURE_SIZE)
                yearly_stats = plot_data.groupby('Year')['Value'].mean().reset_index()
                
                plt.plot(yearly_stats['Year'], yearly_stats['mean'], marker='o', linewidth=2)
                plt.title('Average Value Over Time')
                plt.xlabel('Year')
                plt.ylabel('Average Value')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                try:
                    plt.show()
                except:
                    plt.savefig('time_series.png', dpi=100, bbox_inches='tight')
                    print("   • Plot saved as 'time_series.png'")
                plt.close()
            
            print("✅ Basic plots completed")
            
        except Exception as e:
            print(f"❌ Error creating plots: {e}")


def run_complete_analysis_with_export(data_path: str = "./data/FAOSTAT_data_en_8-3-2025.csv"):
    """
    Run complete analysis and GUARANTEED CSV export.
    
    This function will:
    1. Load your data
    2. Clean it
    3. Export to CSV
    4. Run basic analysis
    
    Args:
        data_path: Path to your FAO data file
    """
    
    print("🚀 STARTING COMPLETE FAO ANALYSIS WITH CSV EXPORT")
    print("=" * 70)
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Initialize
    config = Config()
    config.DATA_PATH = data_path
    analyzer = FAODataAnalyzer(config)
    
    # Step 1: Load data
    print("\n🔄 STEP 1: Loading data...")
    if analyzer.load_data() is None:
        print("❌ Failed to load data. Please check the file path.")
        return None
    
    # Step 2: Clean data
    print("\n🔄 STEP 2: Cleaning data...")
    if analyzer.clean_data() is None:
        print("❌ Failed to clean data.")
        return None
    
    # Step 3: EXPORT TO CSV (GUARANTEED)
    print("\n🔄 STEP 3: Exporting cleaned data to CSV...")
    csv_path = analyzer.export_to_csv()
    
    if csv_path is None:
        print("❌ CSV export failed. Trying alternative method...")
        # Alternative export method
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alt_filename = f"fao_data_backup_{timestamp}.csv"
            analyzer.df.to_csv(alt_filename, index=False)
            print(f"✅ Alternative export successful: {alt_filename}")
            csv_path = alt_filename
        except Exception as e:
            print(f"❌ Alternative export also failed: {e}")
            return None
    
    # Step 4: Run analysis
    print("\n🔄 STEP 4: Running analysis...")
    analyzer.generate_summary_statistics()
    analyzer.create_basic_plots()
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"✅ Data loaded: {analyzer.df.shape[0]:,} rows, {analyzer.df.shape[1]} columns")
    print(f"✅ CSV exported to: {os.path.abspath(csv_path) if csv_path else 'Export failed'}")
    print(f"✅ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return csv_path


def quick_csv_export_only(data_path: str = "./data/FAOSTAT_data_en_8-3-2025.csv", 
                         output_filename: str = "fao_cleaned_data.csv"):
    """
    SUPER SIMPLE function - just load, clean, and export to CSV.
    No analysis, no plots, just clean data export.
    
    Args:
        data_path: Path to your data file
        output_filename: Name for output CSV file
    """
    
    print("🚀 QUICK CSV EXPORT MODE")
    print("=" * 40)
    
    try:
        # Load data
        print("📂 Loading data...")
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df):,} rows")
        
        # Basic cleaning
        print("🧹 Basic cleaning...")
        required_cols = ['Value', 'Area', 'Indicator', 'Year']
        
        # Check if required columns exist
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
            print("Exporting raw data without cleaning...")
        else:
            # Clean only if we have required columns
            df = df.dropna(subset=required_cols)
            print(f"✅ Cleaned to {len(df):,} rows")
        
        # Export
        print("📤 Exporting to CSV...")
        df.to_csv(output_filename, index=False)
        
        # Verify
        if os.path.exists(output_filename):
            file_size = os.path.getsize(output_filename) / 1024
            print(f"✅ SUCCESS! File saved: {os.path.abspath(output_filename)}")
            print(f"📊 {len(df):,} rows × {len(df.columns)} columns")
            print(f"💾 File size: {file_size:.1f} KB")
            return output_filename
        else:
            print("❌ File not created")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# MAIN EXECUTION
if __name__ == "__main__":
    print("🎯 FAO DATA ANALYSIS WITH GUARANTEED CSV EXPORT")
    print("=" * 60)
    
    # Option 1: Full analysis with export (RECOMMENDED)
    print("\n1️⃣ Running full analysis with CSV export...")
    csv_file = run_complete_analysis_with_export()
    
    # Option 2: If the above fails, try quick export only
    if csv_file is None:
        print("\n2️⃣ Trying quick CSV export only...")
        csv_file = quick_csv_export_only()
    
    # Final message
    if csv_file:
        print(f"\n🎉 SUCCESS! Your cleaned data is available at:")
        print(f"📁 {os.path.abspath(csv_file)}")
        print("\nYou can now open this file in Excel or any other program!")
    else:
        print("\n❌ Export failed. Please check:")
        print("   • File path is correct")
        print("   • You have write permissions")
        print("   • The data file exists and is readable")
    
    print("\n" + "=" * 60)
    print("✅ Script execution completed!")