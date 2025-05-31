import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a new component for company comparison
def company_comparison_component(df):
    st.header("Company Comparison")
    
    # Select companies to compare
    companies = sorted(df['Company'].unique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        company1 = st.selectbox("Select First Company", companies, index=0)
    
    with col2:
        # Default to second company in the list, or first if only one company
        default_idx = min(1, len(companies)-1)
        company2 = st.selectbox("Select Second Company", companies, index=default_idx)
    
    # Select period type
    period_type = st.selectbox("Select Period Type", ["Annual", "Quarterly"], index=0)
    
    # Filter data for selected companies and period type
    company1_data = df[(df['Company'] == company1) & (df['Period_Type'] == period_type)]
    company2_data = df[(df['Company'] == company2) & (df['Period_Type'] == period_type)]
    
    if company1_data.empty or company2_data.empty:
        st.warning(f"No {period_type.lower()} data available for one or both companies.")
        return
    
    # Get the most recent data for each company
    company1_recent = company1_data.sort_values('Date', ascending=False).iloc[0]
    company2_recent = company2_data.sort_values('Date', ascending=False).iloc[0]
    
    # Display comparison
    st.subheader(f"Comparing {company1} vs {company2} ({period_type} Data)")
    
    # Financial metrics comparison
    metrics = ['ROE', 'Net Profit Margin', 'ROA', 'Gross Margin', 
              'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']
    
    # Create tabs for different comparison views
    tab1, tab2, tab3 = st.tabs(["Side-by-Side Metrics", "Radar Comparison", "Historical Trends"])
    
    with tab1:
        # Side-by-side metrics comparison
        for i in range(0, len(metrics), 3):
            cols = st.columns(3)
            for j in range(3):
                if i+j < len(metrics):
                    metric = metrics[i+j]
                    if metric in company1_recent and metric in company2_recent:
                        with cols[j]:
                            st.subheader(metric)
                            
                            # Format values based on metric type
                            if metric in ['ROE', 'Net Profit Margin', 'ROA', 'Gross Margin']:
                                val1 = f"{company1_recent[metric]*100:.2f}%"
                                val2 = f"{company2_recent[metric]*100:.2f}%"
                            else:
                                val1 = f"{company1_recent[metric]:.2f}"
                                val2 = f"{company2_recent[metric]:.2f}"
                            
                            # Calculate difference and determine which is better
                            diff = company1_recent[metric] - company2_recent[metric]
                            
                            # Determine which is better (higher is better for most metrics except debt ratios)
                            if metric in ['Debt-to-Equity', 'Debt-to-Assets']:
                                better = "lower" if diff < 0 else "higher"
                                color = "green" if diff < 0 else "red"
                            else:
                                better = "higher" if diff > 0 else "lower"
                                color = "green" if diff > 0 else "red"
                            
                            # Display comparison
                            st.write(f"**{company1}**: {val1}")
                            st.write(f"**{company2}**: {val2}")
                            
                            # Show difference with color
                            if abs(diff) > 0.001:  # Only show meaningful differences
                                if metric in ['ROE', 'Net Profit Margin', 'ROA', 'Gross Margin']:
                                    diff_text = f"{abs(diff)*100:.2f}% {better}"
                                else:
                                    diff_text = f"{abs(diff):.2f} {better}"
                                
                                st.markdown(f"<span style='color:{color}'>{company1} is {diff_text}</span>", unsafe_allow_html=True)
    
    with tab2:
        # Radar chart comparison
        radar_metrics = [m for m in metrics if m in company1_recent and m in company2_recent]
        
        if radar_metrics:
            # Normalize values for radar chart (0-1 scale)
            max_values = {}
            min_values = {}
            
            for metric in radar_metrics:
                all_values = df[metric].dropna()
                max_values[metric] = all_values.max()
                min_values[metric] = all_values.min()
            
            # Normalize company values
            company1_norm = {}
            company2_norm = {}
            
            for metric in radar_metrics:
                # Avoid division by zero
                if max_values[metric] == min_values[metric]:
                    company1_norm[metric] = 0.5
                    company2_norm[metric] = 0.5
                else:
                    # For debt ratios, invert normalization (lower is better)
                    if metric in ['Debt-to-Equity', 'Debt-to-Assets']:
                        company1_norm[metric] = 1 - (company1_recent[metric] - min_values[metric]) / (max_values[metric] - min_values[metric])
                        company2_norm[metric] = 1 - (company2_recent[metric] - min_values[metric]) / (max_values[metric] - min_values[metric])
                    else:
                        company1_norm[metric] = (company1_recent[metric] - min_values[metric]) / (max_values[metric] - min_values[metric])
                        company2_norm[metric] = (company2_recent[metric] - min_values[metric]) / (max_values[metric] - min_values[metric])
            
            # Create radar chart
            fig = go.Figure()
            
            # Add traces for each company
            fig.add_trace(go.Scatterpolar(
                r=[company1_norm[m] for m in radar_metrics],
                theta=radar_metrics,
                fill='toself',
                name=company1
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=[company2_norm[m] for m in radar_metrics],
                theta=radar_metrics,
                fill='toself',
                name=company2
            ))
            
            # Update layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title=f"Financial Performance Comparison ({period_type})",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.info("This radar chart shows normalized performance across key metrics. Higher values (further from center) are better for all metrics. For debt ratios, the values are inverted so that lower debt appears as higher performance.")
        else:
            st.warning("Insufficient data for radar chart comparison.")
    
    with tab3:
        # Historical trends comparison
        st.subheader("Historical Performance Comparison")
        
        # Select metric for historical comparison
        selected_metric = st.selectbox("Select Metric for Historical Comparison", metrics)
        
        # Filter data for historical comparison
        company1_history = company1_data.sort_values('Date')
        company2_history = company2_data.sort_values('Date')
        
        # Create historical comparison chart
        fig = go.Figure()
        
        # Add traces for each company
        if selected_metric in company1_history.columns:
            fig.add_trace(go.Scatter(
                x=company1_history['Period'],
                y=company1_history[selected_metric],
                mode='lines+markers',
                name=company1
            ))
        
        if selected_metric in company2_history.columns:
            fig.add_trace(go.Scatter(
                x=company2_history['Period'],
                y=company2_history[selected_metric],
                mode='lines+markers',
                name=company2
            ))
        
        # Format y-axis based on metric type
        if selected_metric in ['ROE', 'Net Profit Margin', 'ROA', 'Gross Margin']:
            fig.update_layout(yaxis_tickformat='.1%')
        
        # Update layout
        fig.update_layout(
            title=f"{selected_metric} Comparison Over Time",
            xaxis_title="Period",
            yaxis_title=selected_metric,
            legend_title="Company",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights
        if selected_metric in company1_history.columns and selected_metric in company2_history.columns:
            # Calculate trends
            company1_start = company1_history[selected_metric].iloc[0]
            company1_end = company1_history[selected_metric].iloc[-1]
            company1_change = company1_end - company1_start
            
            company2_start = company2_history[selected_metric].iloc[0]
            company2_end = company2_history[selected_metric].iloc[-1]
            company2_change = company2_end - company2_start
            
            # Determine which company has better trend
            if selected_metric in ['Debt-to-Equity', 'Debt-to-Assets']:
                company1_trend = "improved" if company1_change < 0 else "worsened"
                company2_trend = "improved" if company2_change < 0 else "worsened"
            else:
                company1_trend = "improved" if company1_change > 0 else "worsened"
                company2_trend = "improved" if company2_change > 0 else "worsened"
            
            # Format change values
            if selected_metric in ['ROE', 'Net Profit Margin', 'ROA', 'Gross Margin']:
                company1_change_str = f"{abs(company1_change)*100:.2f}%"
                company2_change_str = f"{abs(company2_change)*100:.2f}%"
            else:
                company1_change_str = f"{abs(company1_change):.2f}"
                company2_change_str = f"{abs(company2_change):.2f}"
            
            # Display insights
            st.write(f"**{company1}** {company1_trend} by {company1_change_str} over this period.")
            st.write(f"**{company2}** {company2_trend} by {company2_change_str} over this period.")
            
            # Determine overall winner
            if selected_metric in ['Debt-to-Equity', 'Debt-to-Assets']:
                better_trend = company1 if company1_change < company2_change else company2
            else:
                better_trend = company1 if company1_change > company2_change else company2
            
            st.write(f"**{better_trend}** showed better trend performance in {selected_metric}.")

# Create a new component for industry analysis
def industry_analysis_component(df):
    st.header("Industry Analysis")
    
    # Create industry averages
    st.subheader("Industry Benchmarking")
    
    # Select period type
    period_type = st.selectbox("Select Period Type for Industry Analysis", ["Annual", "Quarterly"], index=0)
    
    # Filter data for selected period type
    industry_data = df[df['Period_Type'] == period_type]
    
    if industry_data.empty:
        st.warning(f"No {period_type.lower()} data available for industry analysis.")
        return
    
    # Get the most recent year
    if 'Year' in industry_data.columns:
        recent_years = sorted(industry_data['Year'].unique(), reverse=True)
        selected_year = st.selectbox("Select Year for Analysis", recent_years)
        
        # Filter data for selected year
        year_data = industry_data[industry_data['Year'] == selected_year]
    else:
        st.warning("Year information not available for industry analysis.")
        return
    
    # Calculate industry averages
    companies = sorted(year_data['Company'].unique())
    metrics = ['ROE', 'Net Profit Margin', 'ROA', 'Gross Margin', 
              'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']
    
    # Create tabs for different industry views
    tab1, tab2 = st.tabs(["Industry Benchmarks", "Company Rankings"])
    
    with tab1:
        # Calculate industry averages
        industry_avg = {}
        for metric in metrics:
            if metric in year_data.columns:
                industry_avg[metric] = year_data[metric].mean()
        
        # Display industry averages and company comparisons
        st.subheader(f"Industry Benchmarks for {selected_year} ({period_type})")
        
        # Create benchmark chart
        benchmark_data = []
        
        for company in companies:
            company_data = year_data[year_data['Company'] == company]
            if not company_data.empty:
                company_row = {'Company': company}
                
                for metric in metrics:
                    if metric in company_data.columns and metric in industry_avg:
                        company_row[metric] = company_data[metric].iloc[0]
                        company_row[f"{metric}_vs_Industry"] = company_data[metric].iloc[0] - industry_avg[metric]
                
                benchmark_data.append(company_row)
        
        if benchmark_data:
            benchmark_df = pd.DataFrame(benchmark_data)
            
            # Display benchmark chart for each metric
            for metric in metrics:
                if metric in benchmark_df.columns:
                    # Create a horizontal bar chart
                    fig = go.Figure()
                    
                    # Add industry average line
                    fig.add_shape(
                        type="line",
                        x0=industry_avg[metric],
                        x1=industry_avg[metric],
                        y0=-0.5,
                        y1=len(companies) - 0.5,
                        line=dict(color="red", width=2, dash="dash"),
                    )
                    
                    # Add bars for each company
                    fig.add_trace(go.Bar(
                        y=benchmark_df['Company'],
                        x=benchmark_df[metric],
                        orientation='h',
                        marker_color='lightblue',
                        name=metric
                    ))
                    
                    # Format x-axis based on metric type
                    if metric in ['ROE', 'Net Profit Margin', 'ROA', 'Gross Margin']:
                        fig.update_layout(xaxis_tickformat='.1%')
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{metric} by Company vs Industry Average ({industry_avg[metric]:.4f})",
                        xaxis_title=metric,
                        yaxis_title="Company",
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add insights
                    above_avg = benchmark_df[benchmark_df[metric] > industry_avg[metric]]['Company'].tolist()
                    below_avg = benchmark_df[benchmark_df[metric] < industry_avg[metric]]['Company'].tolist()
                    
                    if metric in ['Debt-to-Equity', 'Debt-to-Assets']:
                        # For debt ratios, lower is better
                        st.write(f"**Companies with better (lower) {metric} than industry average:** {', '.join(below_avg)}")
                        st.write(f"**Companies with worse (higher) {metric} than industry average:** {', '.join(above_avg)}")
                    else:
                        # For other metrics, higher is better
                        st.write(f"**Companies with better (higher) {metric} than industry average:** {', '.join(above_avg)}")
                        st.write(f"**Companies with worse (lower) {metric} than industry average:** {', '.join(below_avg)}")
    
    with tab2:
        # Company rankings
        st.subheader(f"Company Rankings for {selected_year} ({period_type})")
        
        # Select metric for ranking
        ranking_metric = st.selectbox("Select Metric for Ranking", metrics)
        
        if ranking_metric in year_data.columns:
            # Create ranking dataframe
            ranking_data = []
            
            for company in companies:
                company_data = year_data[year_data['Company'] == company]
                if not company_data.empty and ranking_metric in company_data.columns:
                    ranking_data.append({
                        'Company': company,
                        ranking_metric: company_data[ranking_metric].iloc[0]
                    })
            
            if ranking_data:
                ranking_df = pd.DataFrame(ranking_data)
                
                # Sort by metric (ascending for debt ratios, descending for others)
                if ranking_metric in ['Debt-to-Equity', 'Debt-to-Assets']:
                    ranking_df = ranking_df.sort_values(ranking_metric, ascending=True)
                    best_company = ranking_df.iloc[0]['Company']
                    worst_company = ranking_df.iloc[-1]['Company']
                else:
                    ranking_df = ranking_df.sort_values(ranking_metric, ascending=False)
                    best_company = ranking_df.iloc[0]['Company']
                    worst_company = ranking_df.iloc[-1]['Company']
                
                # Add rank column
                ranking_df['Rank'] = range(1, len(ranking_df) + 1)
                
                # Reorder columns
                ranking_df = ranking_df[['Rank', 'Company', ranking_metric]]
                
                # Format metric values
                if ranking_metric in ['ROE', 'Net Profit Margin', 'ROA', 'Gross Margin']:
                    ranking_df[ranking_metric] = ranking_df[ranking_metric].apply(lambda x: f"{x*100:.2f}%")
                else:
                    ranking_df[ranking_metric] = ranking_df[ranking_metric].apply(lambda x: f"{x:.2f}")
                
                # Display ranking table
                st.dataframe(ranking_df, use_container_width=True)
                
                # Display insights
                st.write(f"**{best_company}** ranks #1 in {ranking_metric}.")
                st.write(f"**{worst_company}** ranks last in {ranking_metric}.")
                
                # Create ranking visualization
                fig = px.bar(
                    ranking_df,
                    x='Company',
                    y=ranking_metric,
                    color='Rank',
                    color_continuous_scale='Viridis',
                    title=f"Company Rankings by {ranking_metric}"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for {ranking_metric} ranking.")
        else:
            st.warning(f"{ranking_metric} data not available for ranking.")
