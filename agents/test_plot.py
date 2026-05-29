import yfinance as yf
import plotly.graph_objects as go


TICKER = "NVDA"

stock = yf.Ticker(TICKER)

# Get intraday data
data = yf.download(
    TICKER,
    period="1d",
    interval="5m",
    auto_adjust=True,
    prepost=False,
    progress=False,
)

# Remove empty rows
data = data.dropna()

# Handle yfinance multi-index columns if present
if isinstance(data.columns, type(data.columns)) and hasattr(data.columns, "levels"):
    if len(data.columns.levels) > 1:
        data.columns = data.columns.get_level_values(0)

current_price = float(data["Close"].iloc[-1])
open_price = float(data["Open"].iloc[0])
day_low = float(data["Low"].min())
day_high = float(data["High"].max())
volume = int(data["Volume"].sum())

change = current_price - open_price
change_pct = (change / open_price) * 100

line_color = "#ff3b30" if change < 0 else "#00c853"

title_text = (
    f"<span style='font-size:18px;color:#a0a0a0'>NVIDIA Corp ({TICKER})</span><br>"
    f"<span style='font-size:46px;color:white'>${current_price:,.2f}</span><br>"
    f"<span style='font-size:18px;color:{line_color}'>"
    f"{change:+.2f} ({change_pct:+.2f}%) · Today"
    f"</span>"
)

fig = go.Figure()

# Price line with filled area
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Close"],
        mode="lines",
        line=dict(color=line_color, width=3),
        fill="tozeroy",
        fillcolor="rgba(255, 59, 48, 0.18)" if change < 0 else "rgba(0, 200, 83, 0.18)",
        hovertemplate="%{x}<br>Price: $%{y:.2f}<extra></extra>",
    )
)

fig.update_layout(
    template="plotly_dark",
    title=dict(
        text=title_text,
        x=0.02,
        y=0.95,
        xanchor="left",
        yanchor="top",
    ),
    width=1100,
    height=700,
    paper_bgcolor="#1f1f1f",
    plot_bgcolor="#1f1f1f",
    margin=dict(l=50, r=50, t=170, b=110),
    hovermode="x unified",
    showlegend=False,
    xaxis=dict(
        title="",
        showgrid=False,
        showline=True,
        linecolor="#444",
        tickfont=dict(color="#aaaaaa", size=13),
    ),
    yaxis=dict(
        title="",
        side="right",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.12)",
        zeroline=False,
        tickprefix="$",
        tickfont=dict(color="#aaaaaa", size=13),
    ),
)

# Bottom stats, similar to the card
stats_text = (
    f"Open<br><b>${open_price:,.2f}</b><br><br>"
    f"Day Low<br><b>${day_low:,.2f}</b><br><br>"
    f"Day High<br><b>${day_high:,.2f}</b>"
)

stats_text_2 = (
    f"Volume<br><b>{volume / 1_000_000:.1f}M</b><br><br>"
    f"Last Price<br><b>${current_price:,.2f}</b><br><br>"
    f"Change<br><b>{change:+.2f}</b>"
)

fig.add_annotation(
    text=stats_text,
    xref="paper",
    yref="paper",
    x=0.02,
    y=-0.25,
    showarrow=False,
    align="left",
    font=dict(size=15, color="#bbbbbb"),
)

fig.add_annotation(
    text=stats_text_2,
    xref="paper",
    yref="paper",
    x=0.38,
    y=-0.25,
    showarrow=False,
    align="left",
    font=dict(size=15, color="#bbbbbb"),
)

fig.show()