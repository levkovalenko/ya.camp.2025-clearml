import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def class_distribution(data: pd.DataFrame, target_col: str) -> go.Figure:
    polarity_distribution = data.groupby(target_col, as_index=False).agg(
        count=pd.NamedAgg(target_col, "count")
    )
    return px.histogram(polarity_distribution, x=target_col, y="count")
