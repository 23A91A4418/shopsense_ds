import os
import sqlalchemy
import pandas as pd
from fpdf import FPDF
from shopsense.database import execute_query
from shopsense.eda import compute_univariate_stats, churn_distribution_summary, compute_monthly_revenue

class ShopSenseEDAReport(FPDF):
    def header(self):
        # Top banner background
        self.set_fill_color(26, 54, 93)  # Deep Navy Blue
        self.rect(0, 0, 210, 20, 'F')
        
        # Banner Title
        self.set_font("helvetica", "B", 12)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 5)
        self.cell(0, 10, "SHOPSENSE ANALYTICS  |  Customer Intelligence Pipeline", align="L")
        
        # Banner Date/Subtitle
        self.set_font("helvetica", "I", 9)
        self.set_xy(10, 5)
        self.cell(0, 10, "Exploratory Data Analysis Report", align="R")
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(113, 128, 150)  # Slate gray
        # Page number
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")

    def section_header(self, label):
        self.ln(5)
        self.set_font("helvetica", "B", 14)
        self.set_text_color(26, 54, 93)
        self.cell(0, 10, label, ln=True)
        # Horizontal line below header
        self.set_draw_color(43, 108, 176)  # Mid Blue
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def add_key_value_row(self, key, value, background=False):
        if background:
            self.set_fill_color(247, 250, 252)  # Soft gray
            fill = True
        else:
            fill = False
        
        self.set_font("helvetica", "B", 10)
        self.set_text_color(45, 55, 72)  # Charcoal
        self.cell(80, 8, f"  {key}", border=0, fill=fill)
        self.set_font("helvetica", "", 10)
        self.cell(100, 8, f"{value}", border=0, fill=fill, ln=True)

def generate_pdf_report():
    print("Connecting to database...")
    engine = sqlalchemy.create_engine('postgresql://postgres:admin123@localhost:5432/postgres')
    
    # Query database
    customers_df = execute_query("SELECT * FROM shopsense.customers", engine)
    transactions_df = execute_query("SELECT * FROM shopsense.transactions", engine)
    
    if len(customers_df) == 0 or len(transactions_df) == 0:
        raise ValueError("Database tables are empty. Please ensure the pipeline data is ingested first.")
    
    print(f"Loaded {len(customers_df)} customers and {len(transactions_df)} transactions.")
    
    # Compute stats
    churn_metrics = churn_distribution_summary(customers_df)
    uni_stats = compute_univariate_stats(customers_df, ["age"])
    monthly_rev = compute_monthly_revenue(transactions_df)
    
    # Initialize PDF
    pdf = ShopSenseEDAReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_margins(10, 20, 10)
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Block
    pdf.ln(5)
    pdf.set_font("helvetica", "B", 20)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 12, "Executive Customer Intelligence Report", align="C", ln=True)
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(113, 128, 150)
    pdf.cell(0, 6, "Automated Generation of E-Commerce Transactional & Behavioral EDA", align="C", ln=True)
    pdf.ln(8)
    
    # Executive Summary Section
    pdf.section_header("1. Executive Summary")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(45, 55, 72)
    summary_text = (
        "ShopSense Analytics has processed three years of historical customer transactional "
        "and clickstream event data. This automated diagnostic report highlights our primary "
        "customer traits, overall revenue growth patterns, and key retention metrics. These inputs "
        "serve as the structural baseline for our churn prediction, forecasting, and clustering pipelines."
    )
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(5)
    
    # Key Metrics Table
    pdf.add_key_value_row("Total Customer Cohort", f"{churn_metrics['total_customers']:,}", background=True)
    pdf.add_key_value_row("Historical Transaction Count", f"{len(transactions_df):,}", background=False)
    pdf.add_key_value_row("Average Customer Age", f"{uni_stats.loc['age', 'mean']:.1f} years (Median: {uni_stats.loc['age', 'median']:.1f})", background=True)
    pdf.add_key_value_row("Observed Customer Churn Rate", f"{churn_metrics['churn_rate'] * 100:.2f}% ({churn_metrics['churned_count']:,} churned)", background=False)
    pdf.add_key_value_row("Total Return-Adjusted Revenue", f"INR {monthly_rev['return_adjusted_revenue'].sum():,.2f}", background=True)
    
    pdf.ln(10)
    
    # Customer Churn Analysis Section
    pdf.section_header("2. Churn & Demographic Insights")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(45, 55, 72)
    churn_analysis_intro = (
        "Demographic distributions show clear trends when stratified by our customer churn labels. "
        "Below are the churn rates segmented by key categorical acquisition features and premium statuses:"
    )
    pdf.multi_cell(0, 6, churn_analysis_intro)
    pdf.ln(4)
    
    # Premium vs Non-Premium Churn
    premium_churn = churn_metrics["churn_by_premium"].get(True, 0.0) * 100
    non_premium_churn = churn_metrics["churn_by_premium"].get(False, 0.0) * 100
    pdf.add_key_value_row("Premium Member Churn Rate", f"{premium_churn:.2f}%", background=True)
    pdf.add_key_value_row("Non-Premium Member Churn Rate", f"{non_premium_churn:.2f}%", background=False)
    
    # Churn by channel
    pdf.ln(4)
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 8, "Churn Rate by Acquisition Channel:", ln=True)
    
    bg_toggle = True
    for channel, rate in churn_metrics["churn_by_channel"].items():
        pdf.add_key_value_row(channel.capitalize(), f"{rate * 100:.2f}%", background=bg_toggle)
        bg_toggle = not bg_toggle
        
    # Second Page
    pdf.add_page()
    
    # Revenue Trend Section
    pdf.section_header("3. Monthly Revenue Trend")
    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(45, 55, 72)
    revenue_desc = (
        "The graph below depicts the return-adjusted monthly revenue trend across the observed "
        "historical timeframe. Significant seasonal multipliers are visible in the fourth quarter "
        "(October through December), driving a 1.5x-2x increase in monthly sales volumes."
    )
    pdf.multi_cell(0, 6, revenue_desc)
    pdf.ln(5)
    
    # Embed the Plot
    plot_path = "reports/revenue_trend.png"
    if os.path.exists(plot_path):
        # Centers image: page width 210, margins 10 left/right -> available width 190.
        # Draw image at width 160, centered -> left position = (210 - 160)/2 = 25.
        pdf.image(plot_path, x=25, y=pdf.get_y(), w=160)
    else:
        pdf.set_font("helvetica", "I", 10)
        pdf.set_text_color(220, 50, 50)
        pdf.cell(0, 10, "Error: Monthly revenue trend image ('reports/revenue_trend.png') not found.", ln=True)
        
    # Output the PDF
    os.makedirs("reports", exist_ok=True)
    output_pdf_path = "reports/eda_report.pdf"
    pdf.output(output_pdf_path)
    print(f"Report compiled successfully at: {output_pdf_path}")

if __name__ == "__main__":
    generate_pdf_report()
