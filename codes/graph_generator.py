import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def generate_histogram(df, title: str):
    """
    Generates a histogram from a single-column DataFrame or Series and returns the plotly figure.
    X-axis markers are set at intervals of 10.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if df.shape[1] != 1:
        raise ValueError("DataFrame must have exactly one column.")

    column_name = df.columns[0]
    data = df.iloc[:, 0].dropna().astype(float)

    fig = px.histogram(
        data_frame=data, 
        x=column_name, 
        nbins=10, 
        title=title
    )
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=10)

    return fig


def generate_scatterplot_with_regression(df: pd.DataFrame, title: str):
    """
    Generates a scatter plot with a regression line (manually added) from a two-column DataFrame.
    """
    if df.shape[1] != 2 or not all(df.dtypes.apply(np.issubdtype, args=(np.number,))):
        raise ValueError("DataFrame must contain exactly two numerical columns.")

    x_col, y_col = df.columns
    x = df[x_col].dropna()
    y = df[y_col].dropna()

    # Fit regression line (1st degree polynomial)
    coeffs = np.polyfit(x, y, deg=1)
    slope, intercept = coeffs

    # Create scatter plot
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Data Points'
    ))

    # Regression line
    x_range = np.linspace(x.min(), x.max(), 100)
    y_fit = slope * x_range + intercept
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_fit,
        mode='lines',
        name='Regression Line'
    ))

    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col)

    return fig


def generate_stripplots(df: pd.DataFrame):
    """
    Generates a single strip plot for all numerical columns in the given DataFrame,
    placing them next to each other using a shared axis.
    """
    numerical_columns = df.select_dtypes(include=['number']).columns

    if len(numerical_columns) == 0:
        print("No numerical columns found in the DataFrame.")
        return None

    fig = px.strip(df, x=numerical_columns, orientation='h')

    return fig
