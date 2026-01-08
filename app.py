import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(
    page_title="Washington D.C. Bike Rentals Dashboard",
    page_icon="🚴",
    layout="wide"
)

st.title("Washington D.C. Bike Rentals Dashboard 🚴")
st.markdown(
    """
    Interactive dashboard summarizing the analysis of the Washington D.C. bike rental data  
    (Assignments I & II: EDA and visualizations).
    """
)

# ------------------------------------------------
# Data loading and preprocessing
# ------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour

    season_map = {1: "spring", 2: "summer", 3: "fall", 4: "winter"}
    df["season_name"] = df["season"].map(season_map)

    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    df["day_name"] = df["dayofweek"].map(dow_map)

    def get_day_period(h):
        if 0 <= h < 6:
            return "night"
        elif 6 <= h < 12:
            return "morning"
        elif 12 <= h < 18:
            return "afternoon"
        else:
            return "evening"

    df["day_period"] = df["hour"].apply(get_day_period)
    return df

df = load_data()
df["total"] = df["count"]
numeric_cols = ["temp", "atemp", "humidity", "windspeed", "casual", "registered", "count"]

# ------------------------------------------------
# Sidebar filters
# ------------------------------------------------
st.sidebar.header("Filters")

year_options = sorted(df["year"].unique())
selected_years = st.sidebar.multiselect(
    "Select year(s)", options=year_options, default=year_options
)

season_options = df["season_name"].unique().tolist()
selected_seasons = st.sidebar.multiselect(
    "Select season(s)", options=season_options, default=season_options
)

workingday_filter = st.sidebar.radio(
    "Working day?",
    options=["All", "Working days", "Non-working days"],
    index=0
)

filtered = df[df["year"].isin(selected_years)]
filtered = filtered[filtered["season_name"].isin(selected_seasons)]

if workingday_filter == "Working days":
    filtered = filtered[filtered["workingday"] == 1]
elif workingday_filter == "Non-working days":
    filtered = filtered[filtered["workingday"] == 0]

st.sidebar.markdown("---")
st.sidebar.markdown("Use the filters to explore how rentals change across years, seasons and working days.")

# ------------------------------------------------
# KPIs
# ------------------------------------------------
st.subheader("Key figures (after filters)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total rentals", f"{int(filtered['count'].sum()):,}")
with col2:
    st.metric("Mean hourly rentals", f"{filtered['count'].mean():.1f}")
with col3:
    st.metric("Registered rentals", f"{int(filtered['registered'].sum()):,}")
with col4:
    st.metric("Casual rentals", f"{int(filtered['casual'].sum()):,}")

# ------------------------------------------------
# Plot 1: Rentals by month
# ------------------------------------------------
st.markdown("### Rentals by month")

metric_choice = st.selectbox(
    "Metric to show",
    options=["count", "casual", "registered"],
    index=0
)

monthly = (
    filtered.groupby(["year", "month"], as_index=False)[metric_choice].mean()
)
fig1 = px.line(
    monthly,
    x="month",
    y=metric_choice,
    color="year",
    markers=True,
    labels={"month": "Month", metric_choice: f"Mean hourly {metric_choice}"},
    title=f"Mean hourly {metric_choice} by month and year"
)
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------------
# Plot 2: Hourly pattern by working day
# ------------------------------------------------
st.markdown("### Hourly pattern by working day")

hourly = (
    filtered.groupby(["hour", "workingday"], as_index=False)["count"].mean()
)
hourly["workingday_label"] = hourly["workingday"].map({0: "Non-working", 1: "Working"})

fig2 = px.line(
    hourly,
    x="hour",
    y="count",
    color="workingday_label",
    markers=True,
    labels={"count": "Mean hourly rentals", "hour": "Hour of day"},
    title="Mean hourly rentals by working vs non-working days"
)
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------
# Plot 3: Rentals vs chosen variable
# ------------------------------------------------
st.markdown("### Effect of weather and time-related factors")

x_axis_choice = st.radio(
    "X-axis variable",
    options=["temp", "atemp", "humidity", "windspeed", "hour", "dayofweek"],
    index=0,
    horizontal=True
)

fig3 = px.scatter(
    filtered.sample(min(len(filtered), 5000), random_state=0),
    x=x_axis_choice,
    y="count",
    color="season_name",
    opacity=0.6,
    trendline="lowess",
    labels={"count": "Hourly rentals", x_axis_choice: x_axis_choice, "season_name": "Season"},
    title=f"Hourly rentals vs {x_axis_choice} (colored by season)"
)
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------
# Plot 4: Day period vs working day
# ------------------------------------------------
st.markdown("### Rentals by period of day")

day_period_stats = (
    filtered.groupby(["day_period", "workingday"], as_index=False)["count"].mean()
)
day_period_stats["workingday_label"] = day_period_stats["workingday"].map(
    {0: "Non-working", 1: "Working"}
)

fig4 = px.bar(
    day_period_stats,
    x="day_period",
    y="count",
    color="workingday_label",
    barmode="group",
    labels={"day_period": "Period of day", "count": "Mean hourly rentals"},
    title="Mean hourly rentals by period of day and working day"
)
st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------
# Plot 5: Correlation heatmap
# ------------------------------------------------
st.markdown("### Correlation between numeric variables")

corr = filtered[numeric_cols].corr()

fig5, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation matrix of numeric variables")
st.pyplot(fig5)

# ------------------------------------------------
# Raw data
# ------------------------------------------------
with st.expander("Show raw data (after filters)"):
    st.dataframe(filtered.head(200))
