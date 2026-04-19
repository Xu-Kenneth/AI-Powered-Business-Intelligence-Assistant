"""Loads sales_data.csv and computes summary statistics for the knowledge base."""

import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.csv")

AGE_BINS = [18, 25, 35, 45, 55, 70]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Quarter"] = df["Date"].dt.quarter
    df["AgeGroup"] = pd.cut(
        df["Customer_Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
    ).astype(str)
    return df


def compute_summary(df: pd.DataFrame) -> dict:
    summary = {}

    summary["total_sales"] = round(df["Sales"].sum(), 2)
    summary["total_orders"] = len(df)
    summary["avg_order_value"] = round(df["Sales"].mean(), 2)
    summary["median_order_value"] = round(df["Sales"].median(), 2)
    summary["std_order_value"] = round(df["Sales"].std(), 2)
    summary["date_range"] = f"{df['Date'].min().date()} to {df['Date'].max().date()}"
    summary["avg_satisfaction"] = round(df["Customer_Satisfaction"].mean(), 2)
    summary["median_satisfaction"] = round(df["Customer_Satisfaction"].median(), 2)

    summary["sales_by_year"] = (
        df.groupby("Year")["Sales"].sum().round(2).to_dict()
    )
    summary["sales_by_quarter"] = (
        df.groupby(["Year", "Quarter"])["Sales"].sum().round(2).to_dict()
    )
    summary["avg_monthly_sales"] = (
        df.groupby("Month")["Sales"].mean().round(2).to_dict()
    )

    summary["sales_by_product"] = (
        df.groupby("Product")["Sales"].sum().round(2).to_dict()
    )
    summary["orders_by_product"] = (
        df.groupby("Product").size().to_dict()
    )
    summary["avg_satisfaction_by_product"] = (
        df.groupby("Product")["Customer_Satisfaction"].mean().round(2).to_dict()
    )

    summary["sales_by_region"] = (
        df.groupby("Region")["Sales"].sum().round(2).to_dict()
    )
    summary["orders_by_region"] = (
        df.groupby("Region").size().to_dict()
    )
    summary["avg_satisfaction_by_region"] = (
        df.groupby("Region")["Customer_Satisfaction"].mean().round(2).to_dict()
    )

    summary["sales_by_age_group"] = (
        df.groupby("AgeGroup")["Sales"].sum().round(2).to_dict()
    )
    summary["orders_by_age_group"] = (
        df.groupby("AgeGroup").size().to_dict()
    )
    summary["sales_by_gender"] = (
        df.groupby("Customer_Gender")["Sales"].sum().round(2).to_dict()
    )
    summary["orders_by_gender"] = (
        df.groupby("Customer_Gender").size().to_dict()
    )
    summary["avg_satisfaction_by_gender"] = (
        df.groupby("Customer_Gender")["Customer_Satisfaction"].mean().round(2).to_dict()
    )

    return summary
