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
    file_name = 'data/processed/simulations/simulations_dec6.parquet'
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
            range=[v_min, v_max]
        )
    )

    return box_fig

def create_disease_boxplot(df_filtered):
    # Define Dracula color palette
    dracula_bg = '#282a36'
    dracula_font = '#f8f8f2'
    
    # Create the figure
    fig = go.Figure()
    
    # Define colors for each month
    month_colors = {'May': '#ff5555', 'June': '#50fa7b', 'July': '#8be9fd'}
    months = ['May', 'June', 'July']
    
    # Add traces for each month
    for month in months:
        column_name = f'Disease Incidence {month}'
        
        fig.add_trace(
            go.Box(
                y=df_filtered[column_name],
                x=df_filtered['Sprays in May'],
                name=f'{month}',
                marker_color=month_colors[month],
                boxmean='sd',
                marker=dict(size=3),
                boxpoints='suspectedoutliers',
                legendgroup=month
            )
        )
    
    fig.update_layout(
        template=None,
        plot_bgcolor=dracula_bg,
        paper_bgcolor=dracula_bg,
        font=dict(
            color=dracula_font,
            size=10
        ),
        title=dict(
            text='Disease Incidence vs. Sprays in May',
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
            range=[0, 1]
        ),
        showlegend=True,
        legend=dict(
            title='Month',
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        )
    )
    
    return fig

# --- Create filters at the top ---
st.markdown("### Selection Parameters")

# Create row for main filters
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    years_options = [2014, 2015, 2016, 2017, 'All']
    year = st.selectbox('Select Year', years_options, index=4)

with col2:
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

# Get unique values for the boxplot filters
unique_init_prob = sorted(df_filtered['Initial Probability'].unique())
unique_v6_percent = sorted(df_filtered['V6 Percent'].unique())
unique_quantiles = sorted(df_filtered['Quantile'].unique())

with col3:
    init_prob_options = [format_prob(prob) for prob in unique_init_prob]
    init_prob_selected_label = st.selectbox(
        'Initial Probability of Disease (p₀)',
        options=init_prob_options
    )
    init_prob_selected = unique_init_prob[init_prob_options.index(init_prob_selected_label)]

with col4:
    v6_percent_options = [f"{int(v6 * 100)}%" for v6 in unique_v6_percent]
    v6_percent_selected_label = st.selectbox(
        'V6 Percent',
        options=v6_percent_options
    )
    v6_percent_selected = unique_v6_percent[v6_percent_options.index(v6_percent_selected_label)]

with col5:
    quantile_selected = st.selectbox(
        'Quantile',
        options=unique_quantiles
    )

# Filter data for profit boxplot
df_boxplot = df_filtered[
    (df_filtered['Initial Probability'] == init_prob_selected) &
    (df_filtered['V6 Percent'] == v6_percent_selected) &
    (df_filtered['Quantile'] == quantile_selected)
]

# Prepare year text for the title
year_text = 'All Years' if year == 'All' else str(year)

# Create the figures
heatmap_fig = create_heatmap_figure(df_filtered, year_text, market_demand)
profit_boxplot_fig = create_boxplot_figure(df_boxplot)
disease_boxplot_fig = create_disease_boxplot(df_filtered)

# Display plots
st.markdown("")  # Add some spacing
col_left, col_right = st.columns([3, 1])

with col_left:
    st.plotly_chart(heatmap_fig, use_container_width=True)

with col_right:
    st.plotly_chart(profit_boxplot_fig, use_container_width=True)
    st.plotly_chart(disease_boxplot_fig, use_container_width=True)