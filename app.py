import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set the page configuration
st.set_page_config(
    page_title="Profit Percent Change Heatmap",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data with caching to improve performance
@st.cache_data
def load_data():
    file_name = 'data/processed/simulations/simulations_jan31.parquet'
    df_profit_all = pd.read_parquet(file_name)
    df_profit_all['Quantile'] = df_profit_all['Quantile'].astype(str)
    return df_profit_all

df_profit_all = load_data()

# --- Define global variables ---
v_min = df_profit_all['Mean Profit Percent Change'].min()
v_max = df_profit_all['Mean Profit Percent Change'].max()

# --- Define helper functions ---
def format_prob(prob, decimal_places=5):
    formatted = f"{prob:.{decimal_places}f}".rstrip('0').rstrip('.')
    return formatted if formatted else '0'

def create_heatmap_figure(df_filtered, year_text, market_demand):
    # Get sorted unique V6 Percent and Initial Probability values
    unique_v6_percent = sorted(df_filtered['V6 Percent'].unique())
    unique_init_prob = sorted(df_filtered['Initial Probability'].unique())

    # Format initial probabilities
    unique_init_prob_labels = [format_prob(prob) for prob in unique_init_prob]

    # Define the grid size
    n_rows = len(unique_v6_percent)
    n_cols = len(unique_init_prob)

    # Create subplot titles as empty strings
    subplot_titles = [''] * (n_rows * n_cols)

    # Create the figure with subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
        shared_xaxes=True,
        shared_yaxes=True,
        specs=[[{'type': 'heatmap'} for _ in range(n_cols)] for _ in range(n_rows)]
    )

    # Add heatmaps to subplots
    for _, row_data in df_filtered[['V6 Percent', 'Initial Probability']].drop_duplicates().iterrows():
        v6_value = row_data['V6 Percent']
        init_prob_value = row_data['Initial Probability']

        # Filter the dataframe for the current combination
        df_subset = df_filtered[
            (df_filtered['V6 Percent'] == v6_value) &
            (df_filtered['Initial Probability'] == init_prob_value)
        ]

        # Pivot the data for heatmap
        pivot_profit = df_subset.pivot_table(
            values='Mean Profit Percent Change',
            index='Quantile',
            columns='Sprays in May',
            aggfunc="mean"
        ).sort_index(axis=1)

        # Get x and y axes labels
        x_labels = pivot_profit.columns.tolist()
        y_labels = pivot_profit.index.tolist()

        # Determine subplot position
        v6_index = unique_v6_percent.index(v6_value)
        init_prob_index = unique_init_prob.index(init_prob_value)
        row = v6_index + 1
        col = init_prob_index + 1

        # Add heatmap to the subplot
        fig.add_trace(
            go.Heatmap(
                z=pivot_profit.values,
                x=x_labels,
                y=y_labels,
                colorscale='Viridis',
                zmin=v_min,
                zmax=v_max,
                coloraxis='coloraxis',
                showscale=False,
                xgap=0,
                ygap=0,
                hoverongaps=False
            ),
            row=row,
            col=col
        )

    # Define Dracula color palette
    dracula_bg = '#282a36'
    dracula_font = '#f8f8f2'

    # Update layout
    fig.update_layout(
        template=None,
        plot_bgcolor=dracula_bg,
        paper_bgcolor=dracula_bg,
        font=dict(
            color=dracula_font,
            size=10
        ),
        title=dict(
            text=f'Relative Change in Profit given {market_demand.capitalize()} Market Demand in {year_text}',
            font=dict(
                color=dracula_font,
                size=18
            ),
            x=0.5,
            y=0.96,
            xanchor='center',
            yanchor='top'
        ),
        margin=dict(l=80, r=180, t=100, b=80),
        coloraxis=dict(
            colorscale='Viridis',
            cmin=v_min,
            cmax=v_max,
            colorbar=dict(
                title='Profit % Change',
                titleside='right',
                tickfont=dict(size=10),
                len=1.05,
                y=0.5,
                x=1.08,
                thickness=10
            )
        )
    )

    # Add "Initial Probability of Disease" label at the top center
    fig.add_annotation(
        dict(
            text="Initial Probability of Disease",
            x=0.5,
            y=1.15,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=12,
                color=dracula_font
            ),
            align='center',
            xanchor='center'
        )
    )

    # Add initial probability labels
    for idx, init_prob_label in enumerate(unique_init_prob_labels):
        x_pos = (idx + 1 - 0.5) / n_cols
        fig.add_annotation(
            dict(
                text=f"p₀ = {init_prob_label}",
                x=x_pos,
                y=1.06,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(
                    size=8,
                    color=dracula_font
                ),
                align='center',
                xanchor='center'
            )
        )

    # Add row labels (% V6) at the right of each row
    for idx, v6_value in enumerate(unique_v6_percent):
        y_pos = 1 - ((idx + 1 - 0.5) / n_rows)
        fig.add_annotation(
            dict(
                text=f"{int(v6_value * 100)}% V6",
                x=1.08,
                y=y_pos,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(
                    size=8,
                    color=dracula_font
                ),
                align='center',
                yanchor='middle'
            )
        )

    # Add shared x-axis title
    fig.add_annotation(
        dict(
            text="Number of Sprays in May",
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=12,
                color=dracula_font
            ),
            align='center'
        )
    )

    # Add shared y-axis title
    fig.add_annotation(
        dict(
            text="Percentile of Dispersal-Centrality",
            x=-0.08,
            y=0.5,
            textangle=-90,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(
                size=12,
                color=dracula_font
            ),
            align='center'
        )
    )

    # Update hover template
    fig.update_traces(
        hovertemplate=
        'Sprays in May: %{x}<br>' +
        'Percentile: %{y}<br>' +
        'Profit % Change: %{z:.2f}%<extra></extra>'
    )

    return fig

def create_boxplot_figure(df_boxplot):
    # Get number of unique sprays to define colors
    sprays = sorted(df_boxplot['Sprays in May'].unique())
    n_sprays = len(sprays)

    # Generate 'vlag' color palette
    vlag_colors = sns.color_palette('vlag', n_colors=n_sprays).as_hex()

    # Define Dracula color palette
    dracula_bg = '#282a36'
    dracula_font = '#f8f8f2'

    # Create and display the boxplot
    box_fig = go.Figure()

    # Create boxplot traces
    for i, spray in enumerate(sprays):
        df_spray = df_boxplot[df_boxplot['Sprays in May'] == spray]
        box_fig.add_trace(
            go.Box(
                y=df_spray['Mean Profit Percent Change'],
                name=str(spray),
                marker_color=vlag_colors[i],
                boxmean='sd',  # Show mean and standard deviation
                line=dict(color=vlag_colors[i]),
                fillcolor=vlag_colors[i],
                marker=dict(size=3),  # Reduce outlier marker size
                boxpoints='suspectedoutliers',  # Show only suspected outliers
                showlegend=False
            )
        )

    # Update layout with y-axis range matching v_min and v_max
    box_fig.update_layout(
        template=None,
        plot_bgcolor=dracula_bg,
        paper_bgcolor=dracula_bg,
        font=dict(
            color=dracula_font,
            size=10
        ),
        title=dict(
            text='Mean Profit % Change vs. Sprays in May',
            font=dict(
                color=dracula_font,
                size=14
            ),
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title='Sprays in May',
        yaxis_title='Mean Profit % Change',
        xaxis=dict(
            tickmode='linear',
            showgrid=False,
            zeroline=False,
            showline=True,
            mirror=True,
            linecolor=dracula_font,
            tickfont=dict(color=dracula_font)
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            mirror=True,
            linecolor=dracula_font,
            tickfont=dict(color=dracula_font),
            range=[v_min, v_max]  # Set y-axis range to match v_min and v_max
        )
    )

    return box_fig

def create_disease_incidence_boxplot(df_boxplot):
    # Define Dracula color palette
    dracula_bg = '#282a36'
    dracula_font = '#f8f8f2'
    
    # Create figure
    disease_fig = go.Figure()
    
    # Define months and their corresponding columns
    months = {
        'May': 'Disease Incidence May',
        'June': 'Disease Incidence June',
        'July': 'Disease Incidence July'
    }
    
    # Define colors for each month
    colors = ['#8be9fd', '#50fa7b', '#ff79c6']  # Dracula theme colors
    
    # Add traces for each month
    for (month, column), color in zip(months.items(), colors):
        sprays = sorted(df_boxplot['Sprays in May'].unique())
        
        for spray in sprays:
            df_spray = df_boxplot[df_boxplot['Sprays in May'] == spray]
            
            disease_fig.add_trace(
                go.Box(
                    y=df_spray[column],
                    name=str(spray),
                    legendgroup=month,
                    legendgrouptitle_text=month,
                    marker_color=color,
                    boxmean=True,  # Show mean
                    line=dict(color=color),
                    fillcolor=color,
                    marker=dict(size=3),
                    boxpoints='suspectedoutliers',
                    offsetgroup=month,
                    showlegend=True if spray == sprays[0] else False  # Show legend only for first spray of each month
                )
            )
    
    # Update layout
    disease_fig.update_layout(
        template=None,
        plot_bgcolor=dracula_bg,
        paper_bgcolor=dracula_bg,
        font=dict(
            color=dracula_font,
            size=10
        ),
        title=dict(
            text='Disease Incidence by Month vs. Sprays in May',
            font=dict(
                color=dracula_font,
                size=14
            ),
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title='Sprays in May',
        yaxis_title='Disease Incidence',
        xaxis=dict(
            tickmode='linear',
            showgrid=False,
            zeroline=False,
            showline=True,
            mirror=True,
            linecolor=dracula_font,
            tickfont=dict(color=dracula_font)
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            mirror=True,
            linecolor=dracula_font,
            tickfont=dict(color=dracula_font),
            range=[0, 1]  # Disease incidence is between 0 and 1
        ),
        boxmode='group',  # Group boxes by month
        legend=dict(
            title='',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return disease_fig

def create_metric_boxplot(df_boxplot, metrics, title, y_label, color_palette='vlag', y_range=None):
    """
    Create a boxplot for specified metrics.
    
    Args:
        df_boxplot: DataFrame containing the data
        metrics: List of column names to plot
        title: Plot title
        y_label: Y-axis label
        color_palette: Color palette to use
        y_range: Optional y-axis range
    """
    # Define Dracula color palette
    dracula_bg = '#282a36'
    dracula_font = '#f8f8f2'
    
    # Create figure
    fig = go.Figure()
    
    # Get colors for each metric
    if color_palette == 'dracula':
        colors = ['#8be9fd', '#50fa7b', '#ff79c6', '#bd93f9']  # Dracula theme colors
    else:
        colors = sns.color_palette(color_palette, n_colors=len(metrics)).as_hex()
    
    # Add traces for each metric
    sprays = sorted(df_boxplot['Sprays in May'].unique())
    
    for metric, color in zip(metrics, colors):
        for spray in sprays:
            df_spray = df_boxplot[df_boxplot['Sprays in May'] == spray]
            
            fig.add_trace(
                go.Box(
                    y=df_spray[metric],
                    name=str(spray),
                    legendgroup=metric,
                    legendgrouptitle_text=metric.replace('Mean ', '').replace('Disease Incidence ', ''),
                    marker_color=color,
                    boxmean=True,
                    line=dict(color=color),
                    fillcolor=color,
                    marker=dict(size=3),
                    boxpoints='suspectedoutliers',
                    offsetgroup=metric,
                    showlegend=True if spray == sprays[0] else False
                )
            )
    
    # Update layout
    fig.update_layout(
        template=None,
        plot_bgcolor=dracula_bg,
        paper_bgcolor=dracula_bg,
        font=dict(
            color=dracula_font,
            size=10
        ),
        title=dict(
            text=title,
            font=dict(
                color=dracula_font,
                size=14
            ),
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title='Sprays in May',
        yaxis_title=y_label,
        xaxis=dict(
            tickmode='linear',
            showgrid=False,
            zeroline=False,
            showline=True,
            mirror=True,
            linecolor=dracula_font,
            tickfont=dict(color=dracula_font)
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            mirror=True,
            linecolor=dracula_font,
            tickfont=dict(color=dracula_font),
            range=y_range
        ),
        boxmode='group',
        legend=dict(
            title='',
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig

# fungicide cost plot
def create_fungicide_period_boxplot(df_boxplot):
    # Define Dracula color palette
    dracula_bg = '#282a36'
    dracula_font = '#f8f8f2'
    
    # Define the periods and their corresponding columns
    period_columns = {
        'Early Season': 'Mean Fungicide Cost Early Season',
        'May': 'Mean Fungicide Cost May',
        'June': 'Mean Fungicide Cost June',
        'July': 'Mean Fungicide Cost July',
        'Late Season': 'Mean Fungicide Cost Late Season'
    }
    
    # Create figure
    fig = go.Figure()
    
    # Add a box plot for each period
    colors = sns.color_palette('viridis', n_colors=len(period_columns)).as_hex()
    
    for (period, column), color in zip(period_columns.items(), colors):
        fig.add_trace(
            go.Box(
                y=df_boxplot[column],
                name=period,
                marker_color=color,
                boxmean=True,  # Show mean
                line=dict(color=color),
                fillcolor=color,
                marker=dict(size=3),
                boxpoints='suspectedoutliers'
            )
        )
    
    # Update layout
    fig.update_layout(
        template=None,
        plot_bgcolor=dracula_bg,
        paper_bgcolor=dracula_bg,
        font=dict(
            color=dracula_font,
            size=10
        ),
        title=dict(
            text='Fungicide Costs by Period',
            font=dict(
                color=dracula_font,
                size=14
            ),
            x=0.5,
            y=0.95,
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title='Period',
        yaxis_title='Fungicide Cost ($)',
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            mirror=True,
            linecolor=dracula_font,
            tickfont=dict(color=dracula_font)
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            mirror=True,
            linecolor=dracula_font,
            tickfont=dict(color=dracula_font)
        ),
        showlegend=False
    )
    
    return fig


# --- Create filters at the top ---
st.markdown("### Selection Parameters")

# Create six columns for all filters
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    # Year dropdown
    years_options = [2014, 2015, 2016, 2017, 'All']
    year = st.selectbox('Select Year', years_options, index=4)

with col2:
    # Market Demand dropdown
    market_demand_options = ['low', 'moderate', 'high']
    market_demand = st.selectbox(
        'Select Market Demand',
        options=market_demand_options,
        index=0
    )

# Filter data based on selections
df_filtered = df_profit_all.copy()
if year != 'All':
    df_filtered = df_filtered[df_filtered['Year'] == int(year)]
df_filtered = df_filtered[df_filtered['Market Demand'] == market_demand]

# Calculate ranges for metrics from the full filtered dataset
price_range = [df_filtered['Mean Price'].min(), df_filtered['Mean Price'].max()]
yield_range = [df_filtered['Mean Yield'].min(), df_filtered['Mean Yield'].max()]
cone_color_range = [df_filtered['Mean Cone Color'].min(), df_filtered['Mean Cone Color'].max()]

# Get unique values for the boxplot filters
unique_init_prob = sorted(df_filtered['Initial Probability'].unique())
unique_v6_percent = sorted(df_filtered['V6 Percent'].unique())
unique_quantiles = sorted(df_filtered['Quantile'].unique())

with col3:
    # Initial Probability dropdown
    init_prob_options = [format_prob(prob) for prob in unique_init_prob]
    init_prob_selected_label = st.selectbox(
        'Initial Probability of Disease (p₀)',
        options=init_prob_options
    )
    init_prob_selected = unique_init_prob[init_prob_options.index(init_prob_selected_label)]

with col4:
    # V6 Percent dropdown
    v6_percent_options = [f"{int(v6 * 100)}%" for v6 in unique_v6_percent]
    v6_percent_selected_label = st.selectbox(
        'V6 Percent',
        options=v6_percent_options
    )
    v6_percent_selected = unique_v6_percent[v6_percent_options.index(v6_percent_selected_label)]

with col5:
    # Quantile dropdown
    quantile_selected = st.selectbox(
        'Quantile',
        options=unique_quantiles
    )
    
with col6:
    # Number of Leaves dropdown
    unique_num_leaves = sorted(df_filtered['Number of Leaves'].unique())
    num_leaves_options = [str(leaf) for leaf in unique_num_leaves]
    num_leaves_selected = st.selectbox(
        'Number of Leaves',
        options=num_leaves_options
    )
    # Convert selected option back to appropriate type (e.g., int)
    try:
        num_leaves_selected = int(num_leaves_selected)
    except ValueError:
        num_leaves_selected = float(num_leaves_selected)
    
    # Apply the Number of Leaves filter
    df_filtered = df_filtered[df_filtered['Number of Leaves'] == num_leaves_selected]


# Filter data for boxplot
df_boxplot = df_filtered[
    (df_filtered['Initial Probability'] == init_prob_selected) &
    (df_filtered['V6 Percent'] == v6_percent_selected) &
    (df_filtered['Quantile'] == quantile_selected)
]

# Prepare year text for the title
year_text = 'All Years' if year == 'All' else str(year)

# Create the figures
heatmap_fig = create_heatmap_figure(df_filtered, year_text, market_demand)
boxplot_fig = create_boxplot_figure(df_boxplot)

# Display plots
col_left, col_right = st.columns([3, 1])

with col_left:
    st.plotly_chart(heatmap_fig, use_container_width=True)

# Create two columns for the smaller plots
col_right_top, col_right_bottom = st.columns([1, 1])

with col_right_top:
    st.plotly_chart(boxplot_fig, use_container_width=True)

with col_right_bottom:
    # Create and display the disease incidence boxplot
    disease_boxplot_fig = create_disease_incidence_boxplot(df_boxplot)
    st.plotly_chart(disease_boxplot_fig, use_container_width=True)

# Create columns for additional plots
col_metrics1, col_metrics2 = st.columns(2)

with col_metrics1:
    # Disease and Cone Incidence boxplot
    disease_metrics = ['Disease Incidence May', 'Disease Incidence June', 
                      'Disease Incidence July', 'Mean Cone Incidence']
    disease_cone_fig = create_metric_boxplot(
        df_boxplot,
        disease_metrics,
        'Disease and Cone Incidence vs. Sprays in May',
        'Incidence',
        'dracula',
        [0, 1]
    )
    st.plotly_chart(disease_cone_fig, use_container_width=True)

    # Price boxplot
    price_fig = create_metric_boxplot(
        df_boxplot,
        ['Mean Price'],
        'Price vs. Sprays in May',
        'Price ($)',
        'viridis'
    )
    st.plotly_chart(price_fig, use_container_width=True)

with col_metrics2:
    # Yield boxplot
    yield_fig = create_metric_boxplot(
        df_boxplot,
        ['Mean Yield'],
        'Yield vs. Sprays in May',
        'Yield',
        'viridis'
    )
    st.plotly_chart(yield_fig, use_container_width=True)

    # Cone Color boxplot
    cone_color_fig = create_metric_boxplot(
        df_boxplot,
        ['Mean Cone Color'],
        'Cone Color vs. Sprays in May',
        'Cone Color',
        'viridis'
    )
    st.plotly_chart(cone_color_fig, use_container_width=True)
    
with col_metrics2:  # or wherever you want to place the plot
    fungicide_period_fig = create_fungicide_period_boxplot(df_boxplot)
    st.plotly_chart(fungicide_period_fig, use_container_width=True)