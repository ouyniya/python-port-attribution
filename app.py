import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# === UI HEADER ===
# st.title("Equity Mutual Fund Performance Attribution")

st.markdown(
    f"""<p style='font-size: 1rem; font-weight: normal;'>nysdev.com | 2025</p>
            <div style='
                display: flex;
                flex-direction: column;
                line-height: 0.8;
                align-items: start;
                text-align: left;
            '>
                <p style='font-size: 4rem; font-weight: bold;'>Equity Mutual Fund</p>
                <p style='font-size: 3rem; font-weight: semi-bold;'>Performance Attribution</p>
            </div>
            <hr />
            """,
    unsafe_allow_html=True,
)

# === Sample Files Section ===
with st.chat_message("assistant"):
    st.write(
        "Welcome! Here's a breakdown of how this app helps you understand portfolio attribution."
    )
    st.markdown(
        """
    - **Holdings-based attribution**
    - **Time-series performance analysis**
    - **Brinson Attribution (1986)**: Allocation, Selection, Interaction
    """
    )
    st.info("📥 Download example data to get started.")
    if st.toggle("Click to show example data"):
        for label, path in {
            "Prices CSV": "data/prices.csv",
            "Fund Weights CSV": "data/fund_weights.csv",
            "Benchmark Weights CSV": "data/benchmark_weights.csv",
            "Stock-Sector Map CSV": "data/stock_sector_map.csv",
        }.items():
            with open(path, "rb") as f:
                st.download_button(
                    f"Download {label}", data=f, file_name=path.split("/")[-1]
                )

with st.chat_message("assistant"):
    fund_name = st.text_input("Fund Name")

    # Upload files
    st.write("Upload files to get start!")

    # === Upload Section ===
    st.subheader("📤 Upload your data")
    prices_file = st.file_uploader("📈 Prices CSV", type="csv")
    fund_file = st.file_uploader("📁 Fund Weights CSV", type="csv")
    benchmark_file = st.file_uploader("🏦 Benchmark Weights CSV", type="csv")
    sector_file = st.file_uploader("🗂️ Stock-Sector Map CSV", type="csv")

    prices_file = "data/prices.csv"
    fund_file = "data/fund_weights.csv"
    benchmark_file = "data/benchmark_weights.csv"
    sector_file = "data/stock_sector_map.csv"


@st.cache_data
def load_csv(file, index_col=0):
    try:
        df = pd.read_csv(file, index_col=index_col, parse_dates=True)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def validate_data(prices, fund_weights, benchmark_weights, sector_map):
    errors = []

    if not all(prices.columns == fund_weights.columns):
        errors.append("❌ Columns in prices and fund weights must match.")

    if not all(prices.columns == benchmark_weights.columns):
        errors.append("❌ Columns in prices and benchmark weights must match.")

    if "Stock" not in sector_map.columns or "Sector" not in sector_map.columns:
        errors.append("❌ Sector map must contain 'Stock' and 'Sector' columns.")

    return errors


# === Analysis Section ===


# Aggregate Daily Returns & Weights to Sector Level
def calc_agg_daily_return_weight_to_sector_level(
    sector_map, prices, fund_weights, benchmark_weights
):

    returns = prices.pct_change().fillna(0)
    stock_to_sector = sector_map.set_index("Stock")["Sector"]

    # Prepare sector-level data structures
    fund_sector_returns = {}
    benchmark_sector_returns = {}
    fund_sector_weights = {}
    benchmark_sector_weights = {}

    dates = returns.index

    for i in range(len(dates)):
        date = dates[i]

        ret = returns.loc[date]
        fw = fund_weights.loc[date]
        bw = benchmark_weights.loc[date]

        # Combine into DataFrame
        df = pd.DataFrame(
            {"return": ret, "fw": fw, "bw": bw, "sector": stock_to_sector}
        )

        # Group by sector
        grouped = df.groupby("sector")

        fund_sector_returns[date] = grouped.apply(
            lambda x: np.sum(x["fw"] * x["return"])
        )
        benchmark_sector_returns[date] = grouped.apply(
            lambda x: np.sum(x["bw"] * x["return"])
        )
        fund_sector_weights[date] = grouped["fw"].sum()
        benchmark_sector_weights[date] = grouped["bw"].sum()

    return (
        fund_sector_returns,
        benchmark_sector_returns,
        fund_sector_weights,
        benchmark_sector_weights,
    )


def calc_daily_return_and_excess_return(prices, fund_weights, benchmark_weights):

    returns = prices.pct_change().fillna(0)

    r_port = (fund_weights * returns).sum(axis=1)
    r_bm = (benchmark_weights * returns).sum(axis=1)

    r_port.name = "port_return"
    r_bm.name = "bm_returns"

    performance_result = pd.concat([r_port, r_bm], axis=1)
    performance_result["cum_r_port"] = (1 + r_port).cumprod() - 1
    performance_result["cum_r_bm"] = (1 + r_bm).cumprod() - 1
    performance_result["excess_return"] = r_port - r_bm

    return performance_result


def calc_brinson_attribution_1986(prices, fund_weights, benchmark_weights, sector_map):
    """
    This function calculates brinson attribution (1986) for a specific period
    :param prices (pandas.DataFrame):
    :param fund_weights (pandas.DataFrame):
    :param benchmark_weights (pandas.DataFrame):
    :param sector_map (pandas.DataFrame):
    """
    # shift weight
    fund_weights = fund_weights.shift(1, axis=0)
    fund_weights = fund_weights.fillna(method="bfill")  # fill with next row

    benchmark_weights = benchmark_weights.shift(1, axis=0)
    benchmark_weights = benchmark_weights.fillna(method="bfill")  # fill with next row

    # get performance return
    performance_result = calc_daily_return_and_excess_return(
        prices, fund_weights, benchmark_weights
    )

    # get attribution result
    (
        fund_sector_returns,
        benchmark_sector_returns,
        fund_sector_weights,
        benchmark_sector_weights,
    ) = calc_agg_daily_return_weight_to_sector_level(
        sector_map, prices, fund_weights, benchmark_weights
    )

    fund_sector_returns = pd.DataFrame(fund_sector_returns)
    benchmark_sector_returns = pd.DataFrame(benchmark_sector_returns)
    fund_sector_weights = pd.DataFrame(fund_sector_weights)
    benchmark_sector_weights = pd.DataFrame(benchmark_sector_weights)

    allocation_df = pd.DataFrame()
    selection_df = pd.DataFrame()
    interaction_df = pd.DataFrame()

    dates = sorted(fund_sector_returns.keys())  # ให้แน่ใจว่าเรียงลำดับวัน

    for i in range(1, len(dates)):
        date = dates[i]

        rp = fund_sector_returns[date]
        rb = benchmark_sector_returns[date]
        wp = fund_sector_weights[date]
        wb = benchmark_sector_weights[date]

        # total return
        rp = rp / wp
        rb = rb / wb

        # คำนวณราย sector
        allocation = (wp - wb) * rb
        selection = wb * (rp - rb)
        interaction = (wp - wb) * (rp - rb)

        # เพิ่มแต่ละ column เป็น date
        allocation_df[date] = allocation
        selection_df[date] = selection
        interaction_df[date] = interaction

    attrib_summary = pd.DataFrame(
        {
            "Allocation": allocation_df.sum(),
            "Selection": selection_df.sum(),
            "Interaction": interaction_df.sum(),
        }
    )

    attrib_summary["Total effect"] = (
        attrib_summary["Allocation"]
        + attrib_summary["Selection"]
        + attrib_summary["Interaction"]
    )

    # คำนวณ attribution แบบ sector-level (รวมทุกวัน)
    sector_allocation = allocation_df.sum(axis=1)
    sector_selection = selection_df.sum(axis=1)
    sector_interaction = interaction_df.sum(axis=1)
    sector_total = sector_allocation + sector_selection + sector_interaction

    sector_summary = pd.DataFrame(
        {
            "Allocation": sector_allocation,
            "Selection": sector_selection,
            "Interaction": sector_interaction,
            "Total": sector_total,
        }
    )

    sector_summary = sector_summary.applymap(lambda x: x * 100 if pd.notnull(x) else "")
    sector_summary = sector_summary.sort_values(by=["Total"], ascending=False)

    # total effect
    total_allocation_effect = attrib_summary.iloc[:, 0].sum()
    total_selection_effect = attrib_summary.iloc[:, 1].sum()
    total_interaction_effect = attrib_summary.iloc[:, 2].sum()
    total_attribution_effect = attrib_summary.iloc[:, 3].sum()

    # as of
    # Get latest values
    latest_date = fund_sector_returns.columns[-1]
    from_date = fund_sector_returns.columns[1]

    # performance table
    cum_port = performance_result["cum_r_port"].iloc[-1]
    cum_bench = performance_result["cum_r_bm"].iloc[-1]
    diff = cum_port - cum_bench

    st.divider()

    # ---- KPI STYLE ----
    def kpi_card(label, value, suffix="%", is_positive=True):
        arrow = "" if is_positive else "⚠️"
        color = "green" if is_positive else "red"
        return f"<div style='font-size:18px;'><b>{label}:</b> <span style='color:{color}'>{value:.2%} {arrow}</span></div>"

    # ---- LAYOUT ----
    st.header(f"{fund_name}: Performance Attribution Report")
    st.caption(f"Period: from {from_date.date()} to {latest_date.date()}")

    st.subheader("📊 Executive Summary")

    # --- KPI Cards ---
    st.markdown(
        kpi_card("Fund Return", cum_port, is_positive=(cum_port >= cum_bench)),
        unsafe_allow_html=True,
    )
    st.markdown(
        kpi_card("Benchmark Return", cum_bench, is_positive=True),
        unsafe_allow_html=True,
    )
    st.markdown(
        kpi_card("Alpha (Active Return)", diff, is_positive=(diff >= 0)),
        unsafe_allow_html=True,
    )

    # --- Bullet Points Summary ---
    performance_text = "outperformed" if diff >= 0 else "underperformed"
    color = "green" if diff >= 0 else "red"

    best_sector = sector_summary.index[0]
    worst_sector = sector_summary.index[-1]


    best_driver = sector_summary.loc[best_sector][["Allocation", "Selection"]].idxmax()
    worst_driver = sector_summary.loc[worst_sector][
        ["Allocation", "Selection"]
    ].idxmin()


    st.markdown("")
    st.markdown("")
    st.markdown("##### 🔍 Summary Points")
    st.markdown(
        f":gray-badge[:material/edit:] Fund **{performance_text}** the benchmark by :{color}[{diff:.2%}]"
    )
    st.markdown(
        f":green-badge[:material/check:] The **best contributing sector** is **{best_sector}**, mainly due to **{best_driver} effect**.  "
    )
    st.markdown(
        f":red-badge[:material/close:] The **worst contributing sector** is **{worst_sector}**, mainly due to **{worst_driver} effect**."
    )


    st.divider()

    st.subheader("📊 Performance Overview")

    col1, col2 = st.columns(2, border=True)

    with col1:
        st.metric(
            label="Fund vs Benchmark", value=f"{cum_port:.2%}", delta=f"{diff:.2%}"
        )

    with col2:
        color = "green" if diff >= 0 else "red"
        multi = f"""**Portfolio return:** {cum_port:.2%}  
        **Benchmark return:** {cum_bench:.2%}  
        **Excess return:** :{color}[{diff:.2%}]"""
        st.markdown(multi)

    # graph cum return
    chart_data = pd.DataFrame(
        performance_result, columns=["cum_r_port", "cum_r_bm", "excess_return"]
    )

    # Multiply values by 100 to convert to percent
    chart_data_pct = chart_data * 100

    fig = go.Figure()

    # Portfolio line
    fig.add_trace(
        go.Scatter(
            x=chart_data_pct.index,
            y=chart_data_pct["cum_r_port"],
            mode="lines+text",
            name="Portfolio",
            line=dict(color="blue"),
            text=[
                f"{y:.2f}%" if i == len(chart_data_pct) - 1 else ""
                for i, y in enumerate(chart_data_pct["cum_r_port"])
            ],
            textposition="top right",
        )
    )

    # Benchmark line
    fig.add_trace(
        go.Scatter(
            x=chart_data_pct.index,
            y=chart_data_pct["cum_r_bm"],
            mode="lines+text",
            name="Benchmark",
            line=dict(color="gray", dash="dot"),
            text=[
                f"{y:.2f}%" if i == len(chart_data_pct) - 1 else ""
                for i, y in enumerate(chart_data_pct["cum_r_bm"])
            ],
            textposition="bottom right",
        )
    )

    fig.update_layout(
        title="Cumulative Returns Over Time (%)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        template="plotly_white",
        legend=dict(x=0, y=1),
        height=500,
        yaxis_tickformat=".2f",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # แสดง Top 5 Sector ที่ช่วย portfolio และ Top 5 ที่ถ่วง performance
    st.subheader("📊 Sector Attribution Summary")

    # total_allocation_effect = attrib_summary.iloc[:,0].sum()
    # total_selection_effect = attrib_summary.iloc[:,1].sum()
    # total_interaction_effect = attrib_summary.iloc[:,2].sum()
    # total_attribution_effect = attrib_summary.iloc[:,3].sum()

    col1, col2, col3, col4 = st.columns(4, border=True)

    with col1:
        color = "green" if total_allocation_effect >= 0 else "red"
        st.markdown(
            f"<div style='font-size:14px; margin-bottom: 1rem;'>Allocation Effect <span style='font-size:36px; line-height: 1.25; color:{color}'>{total_allocation_effect:.2%}</span></div>",
            unsafe_allow_html=True,
        )

    with col2:
        color = "green" if total_selection_effect >= 0 else "red"
        st.markdown(
            f"<div style='font-size:14px; margin-bottom: 1rem;'>Selection Effect <span style='font-size:36px; line-height: 1.25; color:{color}'>{total_selection_effect:.2%}</span></div>",
            unsafe_allow_html=True,
        )

    with col3:
        color = "green" if total_interaction_effect >= 0 else "red"
        st.markdown(
            f"<div style='font-size:14px; margin-bottom: 1rem;'>Interaction Effect <span style='font-size:36px; line-height: 1.25; color:{color}'>{total_interaction_effect:.2%}</span></div>",
            unsafe_allow_html=True,
        )

    with col4:
        color = "green" if total_attribution_effect >= 0 else "red"
        st.markdown(
            f"<div style='font-size:14px; margin-bottom: 1rem;'>Total Effect <span style='font-size:36px; line-height: 1.25; color:{color}'>{total_attribution_effect:.2%}</span></div>",
            unsafe_allow_html=True,
        )

    # สรุปข้อความแบบ Insight
    st.markdown("**📝 Insight Summary**")

    best_sector = sector_summary.index[-1]
    worst_sector = sector_summary.index[0]

    best_driver = sector_summary.loc[best_sector][["Allocation", "Selection"]].idxmax()
    worst_driver = sector_summary.loc[worst_sector][
        ["Allocation", "Selection"]
    ].idxmin()

    st.markdown(
        f"""
    - ✅ The **best contributing sector** is **{best_sector}**, mainly due to **{best_driver} effect**.
    - ❌ The **worst contributing sector** is **{worst_sector}**, mainly due to **{worst_driver} effect**.
    """
    )

    # {"✅ Positive contribution" if total_pct > 0 else "⚠️ Negative contribution"} to performance.

    # total attribution
    st.markdown("")
    st.markdown("")
    st.markdown("**Attribution Effect (%)**")
    chart_data = pd.DataFrame(
    {
        "sector": sector_summary.index,
        "Total Effect (%)": sector_summary['Total'],
        "sector":  sector_summary.index,
    }
    )

    st.bar_chart(chart_data, x="sector", y="Total Effect (%)", color="sector", horizontal=True)  

    # allocated effect
    fig2 = go.Figure()

    # Portfolio bars
    fig2.add_trace(
        go.Bar(
            x=sector_summary.index,
            y=sector_summary["Allocation"],
            name="Allocation",
        )
    )

    fig2.add_trace(
        go.Bar(
            x=sector_summary.index,
            y=sector_summary["Selection"],
            name="Selection",

        )
    )

    fig2.add_trace(
        go.Bar(
            x=sector_summary.index,
            y=sector_summary["Interaction"],
            name="Interaction",

        )
    )

    fig2.add_trace(
        go.Bar(
            x=sector_summary.index,
            y=sector_summary["Total"],
            name="Total",
        )
    )

    fig2.update_layout(
        title="Attribution Effect Allocation (%)",
        xaxis_title="Date",
        yaxis_title="Effect (%)",
        template="plotly_dark",
        legend=dict(x=1, y=1),
        height=500,
        yaxis_tickformat=".2f",
        barmode="group"  # <--- Stacked bars
    )

    st.plotly_chart(fig2, use_container_width=True, key="Attribution_effect")




    st.write("Top 5 Best Performing Sectors (Total Attribution):")
    st.dataframe(sector_summary.head(5).style.background_gradient(cmap="Greens"))

    sector_summary = sector_summary.sort_values(by=["Total"], ascending=True)

    st.write("Bottom 5 Worst Performing Sectors (Total Attribution):")
    st.dataframe(sector_summary.head(5).style.background_gradient(cmap="Reds"))


if prices_file and fund_file and benchmark_file:
    prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
    fund_weights = pd.read_csv(fund_file, index_col=0, parse_dates=True)
    benchmark_weights = pd.read_csv(benchmark_file, index_col=0, parse_dates=True)
    sector_map = pd.read_csv(sector_file)

    if len(prices) == len(fund_weights) and len(prices) == len(benchmark_weights):
        # if st.button("Submit"):
        calc_brinson_attribution_1986(
            prices, fund_weights, benchmark_weights, sector_map
        )
