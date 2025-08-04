# Simple CSV Export Function for Your Local Machine
# Add this to your existing FAO analysis script

import os
from datetime import datetime

def export_to_local_csv(analyzer, output_folder="./cleaned_data_exports"):
    """
    Export cleaned data to CSV on your local machine.
    
    Args:
        analyzer: Your FAODataAnalyzer instance
        output_folder: Folder to save the CSV file
    """
    
    if analyzer.df is None:
        print("❌ No cleaned data available. Run the analysis first!")
        return None
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fao_cleaned_data_{timestamp}.csv"
    filepath = os.path.join(output_folder, filename)
    
    try:
        # Export to CSV
        analyzer.df.to_csv(filepath, index=False, encoding='utf-8')
        
        # Get file info
        file_size = os.path.getsize(filepath) / 1024  # Size in KB
        
        print("✅ CSV Export Successful!")
        print(f"📁 File saved to: {os.path.abspath(filepath)}")
        print(f"📊 Data exported: {len(analyzer.df)} rows, {len(analyzer.df.columns)} columns")
        print(f"💾 File size: {file_size:.1f} KB")
        
        return filepath
        
    except Exception as e:
        print(f"❌ Error exporting CSV: {e}")
        return None

# Usage example:
# After running your analysis
def run_analysis_and_export():
    """Complete workflow: analyze data and export to CSV"""
    
    print("🚀 Starting FAO Data Analysis and Export...")
    print("="*60)
    
    # 1. Run your analysis
    analyzer = FAODataAnalyzer()
    
    # Load and clean data
    if analyzer.load_data() is not None:
        if analyzer.clean_data() is not None:
            
            # 2. Export cleaned data to CSV
            csv_path = export_to_local_csv(analyzer)
            
            # 3. Optional: Also run the full analysis
            print("\n📊 Running full analysis...")
            analyzer.generate_summary_statistics()
            analyzer.create_exploratory_plots()
            clustering_results = analyzer.perform_clustering_analysis()
            model_metrics = analyzer.build_regression_model()
            analyzer.print_results_summary(clustering_results, model_metrics)
            
            print(f"\n🎉 Complete! Your cleaned data is saved as CSV at:")
            print(f"📁 {os.path.abspath(csv_path) if csv_path else 'Export failed'}")
            
            return csv_path
        else:
            print("❌ Data cleaning failed")
    else:
        print("❌ Data loading failed")
    
    return None

# Quick export function (minimal version)
def quick_csv_export(df, filename="cleaned_fao_data.csv"):
    """
    Super simple CSV export function
    
    Args:
        df: Your cleaned DataFrame
        filename: Name for the CSV file
    """
    try:
        df.to_csv(filename, index=False)
        print(f"✅ CSV saved: {filename}")
        print(f"📊 Exported {len(df)} rows and {len(df.columns)} columns")
        return filename
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return None

# If you want to export immediately after cleaning
def add_export_to_analyzer():
    """Add export method directly to your analyzer"""
    
    def export_csv(self, filename=None):
        """Export cleaned data to CSV"""
        if self.df is None:
            print("❌ No data to export")
            return None
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fao_cleaned_{timestamp}.csv"
        
        try:
            self.df.to_csv(filename, index=False)
            print(f"✅ Exported to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return None
    
    # Add this method to your FAODataAnalyzer class
    FAODataAnalyzer.export_csv = export_csv

# Example usage in your script:
if __name__ == "__main__":
    
    # Method 1: Run analysis and export
    csv_file = run_analysis_and_export()
    
    # Method 2: Quick export after you have a DataFrame
    # quick_csv_export(your_dataframe, "my_cleaned_data.csv")
    
    # Method 3: Add export method to analyzer
    # add_export_to_analyzer()
    # analyzer = FAODataAnalyzer()
    # analyzer.load_data()
    # analyzer.clean_data()
    # analyzer.export_csv("my_file.csv")