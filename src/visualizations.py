"""Plotly chart generators for the Streamlit dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def sales_over_time(df: pd.DataFrame) -> go.Figure:
    monthly = df.groupby(df["Date"].dt.to_period("M"))["Sales"].sum().reset_index()
    monthly["Date"] = monthly["Date"].astype(str)
    fig = px.line(
        monthly, x="Date", y="Sales",
        title="Monthly Sales Over Time",
        labels={"Sales": "Total Sales ($)", "Date": "Month"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def sales_by_product(df: pd.DataFrame) -> go.Figure:
    prod = df.groupby("Product")["Sales"].sum().reset_index().sort_values("Sales", ascending=False)
    fig = px.bar(
        prod, x="Product", y="Sales",
        title="Total Sales by Product",
        color="Product",
        labels={"Sales": "Total Sales ($)"},
    )
    return fig


def sales_by_region(df: pd.DataFrame) -> go.Figure:
    reg = df.groupby("Region")["Sales"].sum().reset_index()
    fig = px.pie(
        reg, names="Region", values="Sales",
        title="Sales Distribution by Region",
    )
    return fig


def satisfaction_by_product(df: pd.DataFrame) -> go.Figure:
    sat = df.groupby("Product")["Customer_Satisfaction"].mean().reset_index()
    fig = px.bar(
        sat, x="Product", y="Customer_Satisfaction",
        title="Average Customer Satisfaction by Product",
        color="Product",
        labels={"Customer_Satisfaction": "Avg Satisfaction (1-5)"},
        range_y=[0, 5],
    )
    fig.add_hline(y=sat["Customer_Satisfaction"].mean(), line_dash="dash",
                  annotation_text="Overall avg", annotation_position="top left")
    return fig


def sales_by_age_gender(df: pd.DataFrame) -> go.Figure:
    grp = (
        df.groupby(["AgeGroup", "Customer_Gender"])["Sales"]
        .sum()
        .reset_index()
        .sort_values("AgeGroup")
    )
    fig = px.bar(
        grp, x="AgeGroup", y="Sales", color="Customer_Gender",
        barmode="group",
        title="Sales by Age Group and Gender",
        labels={"Sales": "Total Sales ($)", "AgeGroup": "Age Group"},
    )
    return fig


def quarterly_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(
        index="Quarter", columns="Year", values="Sales", aggfunc="sum"
    )
    fig = px.imshow(
        pivot,
        title="Quarterly Sales Heatmap (by Year)",
        labels={"x": "Year", "y": "Quarter", "color": "Sales ($)"},
        aspect="auto",
        color_continuous_scale="Blues",
    )
    return fig


def satisfaction_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="Customer_Satisfaction", nbins=20,
        title="Customer Satisfaction Score Distribution",
        labels={"Customer_Satisfaction": "Satisfaction Score"},
        color_discrete_sequence=["#636EFA"],
    )
    return fig
