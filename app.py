import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(page_title="Penguins Dataset Analysis", layout="wide")

# Title
st.title("🐧 Penguins Dataset - IDA & EDA")
st.markdown("**Initial Data Analysis and Exploratory Data Analysis**")

# Load dataset
@st.cache_data
def load_data():
    return sns.load_dataset('penguins')

df = load_data()

# Sidebar
st.sidebar.header("Navigation")
analysis_type = st.sidebar.radio(
    "Select Analysis:",
    ["Dataset Overview", "Class Imbalance", "Missing Values", "Outliers", 
     "Correlation Analysis", "Distribution Plots", "Scatter Plots"]
)

# Dataset Overview
if analysis_type == "Dataset Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    st.subheader("First 10 Rows")
    st.dataframe(df.head(10))
    
    st.subheader("Dataset Info")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Types:**")
        st.dataframe(pd.DataFrame({
            'Column': df.dtypes.index,
            'Type': df.dtypes.values
        }))
    
    with col2:
        st.write("**Statistical Summary:**")
        st.dataframe(df.describe())

# Class Imbalance
elif analysis_type == "Class Imbalance":
    st.header("⚖️ Class Imbalance Analysis")
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Selectbox to choose categorical variable
    selected_cat = st.selectbox("Select Categorical Variable:", categorical_cols)
    
    st.subheader(f"{selected_cat.title()} Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cat_counts = df[selected_cat].value_counts()
        st.dataframe(pd.DataFrame({
            selected_cat.title(): cat_counts.index,
            'Count': cat_counts.values,
            'Percentage': (cat_counts.values / len(df) * 100).round(2)
        }))
    
    with col2:
        # Interactive bar chart
        fig = px.bar(
            x=cat_counts.index,
            y=cat_counts.values,
            labels={'x': selected_cat.title(), 'y': 'Count'},
            title=f'{selected_cat.title()} Distribution',
            color=cat_counts.index,
            text=cat_counts.values
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Pie chart
    st.subheader(f"{selected_cat.title()} Proportion")
    fig = px.pie(
        values=cat_counts.values,
        names=cat_counts.index,
        title=f'{selected_cat.title()} Proportion',
        hole=0.3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

# Missing Values
elif analysis_type == "Missing Values":
    st.header("🔍 Missing Values Analysis")
    
    # Missing values summary
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Values Summary")
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': missing_pct.values
        })
        st.dataframe(missing_df)
    
    with col2:
        st.subheader("Missing Values Bar Chart")
        missing_filtered = missing[missing > 0]
        if len(missing_filtered) > 0:
            fig = px.bar(
                x=missing_filtered.index,
                y=missing_filtered.values,
                labels={'x': 'Column', 'y': 'Missing Count'},
                title='Missing Values by Column',
                text=missing_filtered.values,
                color=missing_filtered.values,
                color_continuous_scale='Reds'
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No missing values found!")
    
    # Heatmap
    st.subheader("Missing Values Heatmap")
    missing_matrix = df.isnull().astype(int)
    
    fig = px.imshow(
        missing_matrix.T,
        labels=dict(x="Row Index", y="Column", color="Missing"),
        y=df.columns,
        color_continuous_scale='Viridis',
        aspect='auto',
        title='Missing Values Heatmap (Yellow = Missing)'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Outliers
elif analysis_type == "Outliers":
    st.header("📈 Outlier Detection")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    st.subheader("Box Plots for Numerical Features")
    
    # Create interactive box plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=numeric_cols
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for i, col in enumerate(numeric_cols):
        if i < len(positions):
            row, col_pos = positions[i]
            fig.add_trace(
                go.Box(y=df[col], name=col, marker_color='#3498DB'),
                row=row, col=col_pos
            )
    
    fig.update_layout(height=700, showlegend=False, title_text="Box Plots for Outlier Detection")
    st.plotly_chart(fig, use_container_width=True)
    
    # Outlier statistics
    st.subheader("Outlier Statistics (IQR Method)")
    
    outlier_data = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_data.append({
            'Feature': col,
            'Outlier Count': len(outliers),
            'Percentage': round(len(outliers) / len(df) * 100, 2)
        })
    
    st.dataframe(pd.DataFrame(outlier_data))

# Correlation Analysis
elif analysis_type == "Correlation Analysis":
    st.header("🔗 Correlation Analysis")
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    st.subheader("Correlation Matrix")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(numeric_df.corr().round(3))
    
    with col2:
        corr_matrix = numeric_df.corr()
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            labels=dict(color="Correlation"),
            title='Correlation Heatmap',
            aspect='auto'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Pairwise correlation
    st.subheader("Strong Correlations (|r| > 0.5)")
    corr_matrix = numeric_df.corr()
    strong_corr = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                strong_corr.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': round(corr_matrix.iloc[i, j], 3)
                })
    
    if strong_corr:
        st.dataframe(pd.DataFrame(strong_corr))
    else:
        st.info("No strong correlations found (threshold: |r| > 0.5)")

# Distribution Plots
elif analysis_type == "Distribution Plots":
    st.header("📊 Distribution Plots")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    st.subheader("Histograms for Numerical Features")
    
    # Create interactive histograms
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=numeric_cols
    )
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for i, col in enumerate(numeric_cols):
        if i < len(positions):
            row, col_pos = positions[i]
            fig.add_trace(
                go.Histogram(x=df[col], name=col, marker_color='#9B59B6', opacity=0.7),
                row=row, col=col_pos
            )
    
    fig.update_layout(height=700, showlegend=False, title_text="Distribution of Numerical Features")
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution by species
    st.subheader("Distributions by Species")
    
    selected_feature = st.selectbox("Select a numerical feature:", numeric_cols)
    
    fig = px.histogram(
        df,
        x=selected_feature,
        color='species',
        marginal='box',
        title=f'Distribution of {selected_feature} by Species',
        labels={selected_feature: selected_feature},
        barmode='overlay',
        opacity=0.7
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Violin plot
    st.subheader("Violin Plot by Species")
    fig = px.violin(
        df,
        y=selected_feature,
        x='species',
        color='species',
        box=True,
        points='all',
        title=f'{selected_feature} Distribution by Species (Violin Plot)'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# Scatter Plots
elif analysis_type == "Scatter Plots":
    st.header("🎯 Scatter Plots")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis:", numeric_cols, index=0)
    with col2:
        y_axis = st.selectbox("Select Y-axis:", numeric_cols, index=1)
    
    color_by = st.selectbox("Color by:", ['species', 'sex', 'island', 'None'])
    
    # Interactive scatter plot
    if color_by == 'None':
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            title=f'{y_axis} vs {x_axis}',
            opacity=0.7,
            hover_data=df.columns
        )
    else:
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            title=f'{y_axis} vs {x_axis}',
            opacity=0.7,
            hover_data=df.columns
        )
    
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # 3D Scatter plot
    st.subheader("3D Scatter Plot")
    col1, col2, col3 = st.columns(3)
    with col1:
        x_3d = st.selectbox("Select X-axis (3D):", numeric_cols, index=0, key='x3d')
    with col2:
        y_3d = st.selectbox("Select Y-axis (3D):", numeric_cols, index=1, key='y3d')
    with col3:
        z_3d = st.selectbox("Select Z-axis (3D):", numeric_cols, index=2, key='z3d')
    
    color_3d = st.selectbox("Color by (3D):", ['species', 'sex', 'island'], key='color3d')
    
    fig = px.scatter_3d(
        df,
        x=x_3d,
        y=y_3d,
        z=z_3d,
        color=color_3d,
        title=f'3D Scatter Plot: {x_3d} vs {y_3d} vs {z_3d}',
        opacity=0.7,
        hover_data=df.columns
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter matrix
    st.subheader("Scatter Matrix")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    color_matrix = st.selectbox("Color by (Scatter Matrix):", categorical_cols + ['None'], key='color_matrix')
    
    if st.button("Generate Scatter Matrix"):
        with st.spinner("Generating scatter matrix..."):
            # Create figure with subplots
            n_vars = len(numeric_cols)
            fig = make_subplots(
                rows=n_vars, 
                cols=n_vars,
                subplot_titles=[col if i == 0 else '' for i, col in enumerate(numeric_cols * n_vars)],
                vertical_spacing=0.02,
                horizontal_spacing=0.02
            )
            
            # Get unique categories and colors
            if color_matrix != 'None':
                categories = df[color_matrix].dropna().unique()
                colors_map = px.colors.qualitative.Plotly[:len(categories)]
                color_discrete_map = {cat: colors_map[i] for i, cat in enumerate(categories)}
            
            # Create scatter plots and KDE plots
            for i, col_y in enumerate(numeric_cols):
                for j, col_x in enumerate(numeric_cols):
                    row = i + 1
                    col = j + 1
                    
                    # Diagonal: KDE plots
                    if i == j:
                        if color_matrix == 'None':
                            # Single KDE
                            data_clean = df[col_x].dropna()
                            from scipy import stats
                            density = stats.gaussian_kde(data_clean)
                            xs = np.linspace(data_clean.min(), data_clean.max(), 200)
                            ys = density(xs)
                            
                            fig.add_trace(
                                go.Scatter(x=xs, y=ys, mode='lines', 
                                          line=dict(color='#636EFA', width=2),
                                          fill='tozeroy', fillcolor='rgba(99, 110, 250, 0.3)',
                                          showlegend=False),
                                row=row, col=col
                            )
                        else:
                            # Multiple KDEs by category
                            from scipy import stats
                            for idx, cat in enumerate(categories):
                                data_cat = df[df[color_matrix] == cat][col_x].dropna()
                                if len(data_cat) > 1:
                                    density = stats.gaussian_kde(data_cat)
                                    xs = np.linspace(data_cat.min(), data_cat.max(), 200)
                                    ys = density(xs)
                                    
                                    show_legend = (i == 0)  # Only show legend for first row
                                    fig.add_trace(
                                        go.Scatter(x=xs, y=ys, mode='lines',
                                                  name=str(cat),
                                                  line=dict(color=color_discrete_map[cat], width=2),
                                                  fill='tozeroy', 
                                                  fillcolor=color_discrete_map[cat].replace('rgb', 'rgba').replace(')', ', 0.3)'),
                                                  showlegend=show_legend,
                                                  legendgroup=str(cat)),
                                        row=row, col=col
                                    )
                    
                    # Off-diagonal: scatter plots
                    else:
                        if color_matrix == 'None':
                            fig.add_trace(
                                go.Scatter(x=df[col_x], y=df[col_y], mode='markers',
                                          marker=dict(size=4, opacity=0.6),
                                          showlegend=False),
                                row=row, col=col
                            )
                        else:
                            for idx, cat in enumerate(categories):
                                data_cat = df[df[color_matrix] == cat]
                                show_legend = (i == 0 and j == 1)  # Only show legend once
                                fig.add_trace(
                                    go.Scatter(x=data_cat[col_x], y=data_cat[col_y], 
                                              mode='markers',
                                              name=str(cat),
                                              marker=dict(size=4, opacity=0.6, 
                                                        color=color_discrete_map[cat]),
                                              showlegend=show_legend,
                                              legendgroup=str(cat)),
                                    row=row, col=col
                                )
                    
                    # Update axes labels
                    fig.update_xaxes(title_text=col_x if i == n_vars - 1 else "", row=row, col=col)
                    fig.update_yaxes(title_text=col_y if j == 0 else "", row=row, col=col)
            
            title_text = 'Scatter Matrix with KDE on Diagonal'
            if color_matrix != 'None':
                title_text += f' (colored by {color_matrix})'
            
            fig.update_layout(
                height=800, 
                width=800,
                title_text=title_text,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit, Seaborn, and Plotly")
