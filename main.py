import io
import os
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def std_cols(df):
    """CSV 열 이름을 표준 이름으로 통일 (KeyError 방지)"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").
                str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (pd.to_datetime(parts[0], format="%Y%m%d")
                           + pd.to_timedelta(parts[1].astype(int), unit="h"))
        df["time"] = parsed
    return df


def bar_by_id(df, col, title, ylabel, color):
    """개체별 측정값 막대그래프 (개체번호는 원값 그대로 표시)"""
    d = df[["개체번호", col]].dropna().copy()
    if d.empty:
        return None
    d["id_str"] = d["개체번호"].apply(
        lambda v: str(int(v)) if isinstance(v, (int, float)) and float(v).is_integer() else str(v)
    )
    d = d.sort_values("id_str")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["id_str"], y=d[col], name=ylabel,
        marker_color=color, marker_line_color="rgba(0,0,0,0.3)",
        marker_line_width=0.5
    ))
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 15}},
        xaxis_title="개체번호", yaxis_title=ylabel,
        hovermode="x unified",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(tickangle=45)
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def scatter_with_fit(x, y, xtitle, ytitle, r, p, color="coral"):
    """산점도 + np.polyfit 추세선 (trendline 인자 사용 금지)"""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 2:
        return None
    slope, intercept = np.polyfit(d["x"], d["y"], 1)
    xs = np.linspace(d["x"].min(), d["x"].max(), 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["x"], y=d["y"], mode="markers",
        marker=dict(color=color, size=9, opacity=0.75,
                    line=dict(width=0.8, color="rgba(0,0,0,0.4)")),
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>",
        name="측정값"
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=slope * xs + intercept, mode="lines",
        line=dict(color="navy", width=2.5, dash="solid"),
        name=f"추세선 (기울기={slope:.3f})",
        hoverinfo="skip"
    ))
    fig.update_layout(
        title={"text": f"{xtitle} vs {ytitle}<br>Pearson r = {r:.3f}  (p = {p:.4f})",
               "x": 0.5, "xanchor": "center", "font": {"size": 14}},
        xaxis_title=xtitle, yaxis_title=ytitle,
        hovermode="closest",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def env_time_series(env_df, col, title, ylabel, color):
    """환경 데이터 시계열 선 그래프"""
    d = env_df[["time", col]].dropna()
    if d.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["time"], y=d[col], mode="lines",
        line=dict(color=color, width=1.4),
        name=ylabel,
        hovertemplate="%{x|%m/%d %H:%M}<br>" + ylabel + "=%{y:.2f}<extra></extra>"
    ))
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 14}},
        xaxis_title="시간", yaxis_title=ylabel,
        hovermode="x unified",
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig


# ─────────────────────────────────────────────
# 사이드바: CSS · 파일 업로드
# ─────────────────────────────────────────────
st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>""",
    unsafe_allow_html=True
)

st.set_page_config(page_title="극지식물 분석 대시보드", layout="wide")

env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

env_df = pd.DataFrame()
growth_df = pd.DataFrame()
school_name = "업로드된 파일명 기준"

if env_file is not None:
    data = env_file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            env_df = pd.read_csv(io.BytesIO(data), encoding=enc)
            break
        except Exception:
            pass
    if not env_df.empty:
        env_df = std_cols(env_df)
        raw = env_file.name
        school_name = raw.rsplit(".", 1)[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

if growth_file is not None:
    data = growth_file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            growth_df = pd.read_csv(io.BytesIO(data), encoding=enc)
            break
        except Exception:
            pass
    if not growth_df.empty:
        growth_df = std_cols(growth_df)

# ─────────────────────────────────────────────
# 본문
# ─────────────────────────────────────────────
st.title("극지식물 지상부·지하부 길이의 상관관계 분석")
st.caption("학교별 CSV 열 이름이 달라도 자동으로 맞춰 분석합니다.")

if not env_df.empty:
    st.sidebar.success(f"환경 데이터 로드 완료 — {school_name}")
    st.sidebar.caption(f"{len(env_df)}행 · 열: {', '.join(env_df.columns)}")
if not growth_df.empty:
    st.sidebar.success(f"생육 데이터 로드 완료 — {school_name}")
    st.sidebar.caption(f"{len(growth_df)}개체 · 열: {', '.join(growth_df.columns)}")

if env_df.empty and growth_df.empty:
    st.info("📎 사이드바에서 환경 CSV와 생육 CSV를 각각 올려주세요.")
    st.stop()

tab1, tab2, tab3 = st.tabs(
    ["실험 개요", "지상부·지하부 길이 상관관계", "결론 · 기타 분석"]
)

with tab1:
    try:
        st.header("📋 실험 개요")
        st.markdown("""
**가설**  
극지식물은 생존 환경에 적응하기 위해 지상부보다 지하부를 더 길게 뻗는 경향이 있으며,
개체별로 지상부·지하부 길이 사이에 일정한 상관관계가 존재할 것이다.

**실험 목적**  
- 개체별 지상부 길이·지하부 길이의 분포와 차이를 평균·표준편차로 확인한다.  
- 지상부 길이와 지하부 길이 사이의 선형 관계(피어슨 상관계수)를 파악한다.  
- 잎 수, 생중량 등 다른 지표와의 관계를 함께 비교해 생육 수준을 더 잘 나타내는 지표를 찾는다.
        """.strip())

        st.divider()

        st.subheader("🧪 개체별 생육 데이터 요약")

        if not growth_df.empty:
            cols = [c for c in ["개체번호", "잎 수", "지상부 길이", "지하부 길이",
                                 "생중량", "지상부 질량", "지하부 질량"]
                    if c in growth_df.columns]
            available = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                         if c in growth_df.columns]
            if available:
                summary = growth_df[available].describe().round(2).T
                summary.insert(0, "개체 수", len(growth_df))
                summary.insert(1, "결측 수", growth_df[available].isna().sum().values)
                st.dataframe(summary, width="stretch", use_container_width=False)
            else:
                st.info("⚠ 생육 데이터에 표시할 측정 열이 없습니다.")

            st.divider()
            st.subheader("📊 개체별 지상부·지하부 길이 막대그래프")

            col_bar1, col_bar2 = st.columns(2)
            with col_bar1:
                if "지상부 길이" in growth_df.columns:
                    fig1 = bar_by_id(growth_df, "지상부 길이",
                                     "개체별 지상부 길이 (mm)",
                                     "지상부 길이 (mm)", "#2E86AB")
                    if fig1:
                        st.plotly_chart(fig1, width="stretch")
                    else:
                        st.info("⚠ 그래프에 표시할 데이터가 충분하지 않습니다.")
                else:
                    st.info("⚠ '지상부 길이' 열이 없어 그래프를 그리지 못했습니다.")
            with col_bar2:
                if "지하부 길이" in growth_df.columns:
                    fig2 = bar_by_id(growth_df, "지하부 길이",
                                     "개체별 지하부 길이 (mm)",
                                     "지하부 길이 (mm)", "#A23B72")
                    if fig2:
                        st.plotly_chart(fig2, width="stretch")
                    else:
                        st.info("⚠ 그래프에 표시할 데이터가 충분하지 않습니다.")
                else:
                    st.info("⚠ '지하부 길이' 열이 없어 그래프를 그리지 못했습니다.")

            st.divider()
            st.subheader("🌿 생육 상위·하위 개체")

            if "생중량" in growth_df.columns:
                g = growth_df[["개체번호", "생중량"]].dropna().copy()
                if not g.empty:
                    g["id_str"] = g["개체번호"].apply(
                        lambda v: str(int(v))
                        if isinstance(v, (int, float)) and float(v).is_integer()
                        else str(v)
                    )
                    g = g.sort_values("생중량", ascending=False)
                    top3 = g.head(3)
                    bot3 = g.tail(3)
                    st.caption("생중량 상위 3개체")
                    st.dataframe(
                        top3[["id_str", "생중량"]].rename(
                            columns={"id_str": "개체번호", "생중량": "생중량(g)"}),
                        width="stretch", use_container_width=False
                    )
                    st.caption("생중량 하위 3개체")
                    st.dataframe(
                        bot3[["id_str", "생중량"]].rename(
                            columns={"id_str": "개체번호", "생중량": "생중량(g)"}),
                        width="stretch", use_container_width=False
                    )
                else:
                    st.info("⚠ 생중량 데이터가 없습니다.")
            else:
                st.info(
                    "⚠ '생중량' 열이 없습니다. "
                    "(지상부 질량 + 지하부 질량으로 자동 계산되지 않은 학교입니다.)"
                )
        else:
            st.info("📎 생육 결과 CSV를 올려주세요.")

        with st.expander("🌡 환경 데이터 요약 (있을 경우)"):
            if not env_df.empty:
                num_cols = [c for c in ["temperature", "humidity", "ph", "ec",
                                         "water_temp", "co2"]
                            if c in env_df.columns]
                if num_cols:
                    env_summary = env_df[num_cols].describe().round(2).T
                    env_summary.insert(0, "측정 횟수", len(env_df))
                    env_summary.insert(1, "결측 수", env_df[num_cols].isna().sum().values)
                    st.dataframe(env_summary, width="stretch", use_container_width=False)

                    st.divider()
                    st.subheader("📈 환경 시계열")

                    env_cols_plot = {
                        "temperature": ("온도 (°C)", "#E63946"),
                        "humidity": ("습도 (%)", "#489FB5"),
                        "ph": ("pH", "#2A9D8F"),
                        "ec": ("EC (mS/cm)", "#E76F51"),
                    }
                    for c, (lbl, color) in env_cols_plot.items():
                        if c in env_df.columns:
                            fig = env_time_series(env_df, c, f"{lbl} 시계열", lbl, color)
                            if fig:
                                st.plotly_chart(fig, width="stretch")
                            else:
                                st.info(f"⚠ {lbl} 데이터 포인트가 없습니다.")
                else:
                    st.info("⚠ 환경 데이터에 표시 가능한 숫자 열이 없습니다.")
            else:
                st.info("📎 환경 데이터 CSV를 올려주세요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab2:
    try:
        st.header("🔗 지상부·지하부 길이 상관관계")
        st.caption(
            "평균·표준편차·상관계수는 빈 값을 제외하고 계산합니다. "
            "값이 2개 미만이면 해당 항목은 건너뜁니다."
        )

        # ── 환경 요약 ──
        st.subheader("🌡 환경 데이터 요약 (평균·최소·최대)")
        if not env_df.empty:
            env_num = [c for c in ["temperature", "humidity", "ph", "ec",
                                    "water_temp", "co2"]
                       if c in env_df.columns]
            if env_num:
                env_desc = env_df[env_num].describe().T
                summary_tbl = pd.DataFrame({
                    "평균": env_desc["mean"].round(2),
                    "최소": env_desc["min"].round(2),
                    "최대": env_desc["max"].round(2),
                    "표준편차": env_desc["std"].round(2),
                })
                st.dataframe(summary_tbl, width="stretch", use_container_width=False)
            else:
                st.info("⚠ 환경 데이터에 표시 가능한 숫자 열이 없습니다.")
        else:
            st.info("📎 환경 데이터 CSV를 올려주세요.")

        st.divider()

        # ── 통계 카드 ──
        st.subheader("📐 개체별 길이 통계 & 피어슨 상관계수")
        if (not growth_df.empty
                and "지상부 길이" in growth_df.columns
                and "지하부 길이" in growth_df.columns):
            above = growth_df["지상부 길이"].dropna()
            below = growth_df["지하부 길이"].dropna()

            m_above = above.mean() if len(above) >= 2 else np.nan
            sd_above = above.std(ddof=1) if len(above) >= 2 else np.nan
            m_below = below.mean() if len(below) >= 2 else np.nan
            sd_below = below.std(ddof=1) if len(below) >= 2 else np.nan

            both = growth_df[["지상부 길이", "지하부 길이"]].dropna()
            if len(both) >= 2:
                r_val, p_val = stats.pearsonr(both["지상부 길이"], both["지하부 길이"])
            else:
                r_val, p_val = np.nan, np.nan

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "지상부 길이 평균 (mm)",
                f"{m_above:.2f}" if not np.isnan(m_above) else "N/A",
                delta=f"n={len(above)}" if len(above) >= 2 else "데이터 부족"
            )
            c2.metric(
                "지하부 길이 평균 (mm)",
                f"{m_below:.2f}" if not np.isnan(m_below) else "N/A",
                delta=f"n={len(below)}" if len(below) >= 2 else "데이터 부족"
            )
            c3.metric(
                "피어슨 상관계수 r",
                f"{r_val:.3f}"
                if not (isinstance(r_val, float) and np.isnan(r_val))
                else "N/A",
                delta=f"p={p_val:.4f}"
                if not (isinstance(p_val, float) and np.isnan(p_val))
                else "계산 불가",
                help=(
                    "r이 1에 가까울수록 지상부·지하부 길이가 함께 증가/감소하는 "
                    "강한 양의 선형 관계"
                )
            )

            sd_c1, sd_c2 = st.columns(2)
            sd_c1.metric(
                "지상부 길이 표준편차 (mm)",
                f"{sd_above:.2f}" if not np.isnan(sd_above) else "N/A"
            )
            sd_c2.metric(
                "지하부 길이 표준편차 (mm)",
                f"{sd_below:.2f}" if not np.isnan(sd_below) else "N/A"
            )
        else:
            missing = []
            if "지상부 길이" not in growth_df.columns:
                missing.append("지상부 길이")
            if "지하부 길이" not in growth_df.columns:
                missing.append("지하부 길이")
            st.info(
                f"⚠ 다음 열이 없어 통계를 계산하지 못했습니다: "
                f"{', '.join(missing) if missing else '데이터 없음'}"
            )

        st.divider()

        # ── 산점도 + 추세선 ──
        st.subheader("📈 지상부 길이 vs 지하부 길이 (산점도 + 선형 추세선)")
        if (not growth_df.empty
                and "지상부 길이" in growth_df.columns
                and "지하부 길이" in growth_df.columns):
            both = growth_df[["지상부 길이", "지하부 길이"]].dropna()
            if len(both) >= 2:
                r_val, p_val = stats.pearsonr(both["지상부 길이"], both["지하부 길이"])
                fig = scatter_with_fit(
                    both["지상부 길이"], both["지하부 길이"],
                    "지상부 길이 (mm)", "지하부 길이 (mm)",
                    r_val, p_val, color="#2E86AB"
                )
                if fig:
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("⚠ 산점도를 그릴 데이터가 충분하지 않습니다.")
            else:
                st.info(
                    "⚠ 지상부·지하부 길이 쌍이 2개 미만이라 "
                    "산점도를 그리지 못했습니다."
                )
        else:
            st.info("⚠ 지상부 길이 또는 지하부 길이 열이 없습니다.")

        # ── 환경 시계열 참고 ──
        with st.expander("🌡 환경 데이터 시계열 (참고)"):
            if not env_df.empty:
                env_cols_plot2 = {
                    "temperature": ("온도 (°C)", "#E63946"),
                    "humidity": ("습도 (%)", "#489FB5"),
                    "ph": ("pH", "#2A9D8F"),
                    "ec": ("EC (mS/cm)", "#E76F51"),
                    "water_temp": ("수온 (°C)", "#8E44AD"),
                    "co2": ("CO₂ (ppm)", "#16A085"),
                }
                for c, (lbl, color) in env_cols_plot2.items():
                    if c in env_df.columns:
                        fig = env_time_series(env_df, c, f"{lbl} 변화", lbl, color)
                        if fig:
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.info(f"⚠ {lbl} 데이터 포인트가 없습니다.")
            else:
                st.info("📎 환경 데이터 CSV를 올려주세요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab3:
    try:
        st.header("📝 결론 및 기타 분석")

        # ── 상관 요약 ──
        st.subheader("상관관계 분석 요약")
        if (not growth_df.empty
                and "지상부 길이" in growth_df.columns
                and "지하부 길이" in growth_df.columns):
            both = growth_df[["지상부 길이", "지하부 길이"]].dropna()
            if len(both) >= 2:
                r_val, p_val = stats.pearsonr(both["지상부 길이"], both["지하부 길이"])
                direction = (
                    "강한 양의 상관" if r_val > 0.5
                    else ("약한 양의 상관" if r_val > 0
                          else ("강한 음의 상관" if r_val < -0.5
                                else "약한 음의 상관"))
                )
                strength = (
                    "매우 강하다" if abs(r_val) >= 0.8
                    else ("강하다" if abs(r_val) >= 0.6
                          else ("보통" if abs(r_val) >= 0.4
                                else ("약하다" if abs(r_val) >= 0.2
                                      else "매우 약하다")))
                )
                st.markdown(f"""
**지상부 길이 ↔ 지하부 길이**
- 피어슨 상관계수 **r = {r_val:.3f}** (p = {p_val:.4f}, n = {len(both)})
- 방향: {direction}
- 강도: {strength}

**해석 (통계 용어 설명 포함)**  
- **피어슨 상관계수(r)**: 두 변수가 함께 증가하거나 감소하는 선형 관계의 강도를
  -1 ~ 1 사이 값으로 나타냄. 1에 가까우면 강한 양의 상관, -1에 가까우면 강한 음의 상관,
  0에 가까우면 선형 관계가 거의 없음.  
- **p값**: '실제로는 상관이 없는데 우연히 이렇게 나왔을 확률'.
  보통 p < 0.05이면 통계적으로 의미 있는 상관으로 봄.
                """.strip())
            else:
                st.info("⚠ 지상부·지하부 길이 쌍이 부족해 상관분석을 하지 못했습니다.")
        else:
            st.info(
                "⚠ 지상부 길이 또는 지하부 길이 열이 없어 "
                "상관 요약을 표시하지 못했습니다."
            )

        st.divider()

        # ── 추가 분석: 잎 수 ↔ 지하부 길이 ──
        st.subheader("🍃 잎 수 ↔ 지하부 길이")
        if (not growth_df.empty
                and "잎 수" in growth_df.columns
                and "지하부 길이" in growth_df.columns):
            d1 = growth_df[["잎 수", "지하부 길이"]].dropna()
            if len(d1) >= 2:
                r1, p1 = stats.pearsonr(d1["잎 수"], d1["지하부 길이"])
                fig1 = scatter_with_fit(
                    d1["잎 수"], d1["지하부 길이"],
                    "잎 수 (장)", "지하부 길이 (mm)", r1, p1, color="#A23B72"
                )
                if fig1:
                    st.plotly_chart(fig1, width="stretch")
                st.caption(f"r = {r1:.3f}  (p = {p1:.4f}, n = {len(d1)})")
            else:
                st.info("⚠ 잎 수·지하부 길이 쌍이 2개 미만입니다.")
        else:
            st.info("⚠ '잎 수' 또는 '지하부 길이' 열이 없습니다.")

        st.divider()

        # ── 추가 분석: 잎 수 ↔ 생중량 ──
        st.subheader("🍃 잎 수 ↔ 생중량")
        if (not growth_df.empty
                and "잎 수" in growth_df.columns
                and "생중량" in growth_df.columns):
            d2 = growth_df[["잎 수", "생중량"]].dropna()
            if len(d2) >= 2:
                r2, p2 = stats.pearsonr(d2["잎 수"], d2["생중량"])
                fig2 = scatter_with_fit(
                    d2["잎 수"], d2["생중량"],
                    "잎 수 (장)", "생중량 (g)", r2, p2, color="#E9C46A"
                )
                if fig2:
                    st.plotly_chart(fig2, width="stretch")
                st.caption(f"r = {r2:.3f}  (p = {p2:.4f}, n = {len(d2)})")
            else:
                st.info("⚠ 잎 수·생중량 쌍이 2개 미만입니다.")
        else:
            st.info("⚠ '잎 수' 또는 '생중량' 열이 없습니다.")

        st.divider()

        # ── 추가 분석: 생중량 ↔ 지하부 길이 ──
        st.subheader("⚖ 생중량 ↔ 지하부 길이")
        if (not growth_df.empty
                and "생중량" in growth_df.columns
                and "지하부 길이" in growth_df.columns):
            d3 = growth_df[["생중량", "지하부 길이"]].dropna()
            if len(d3) >= 2:
                r3, p3 = stats.pearsonr(d3["생중량"], d3["지하부 길이"])
                fig3 = scatter_with_fit(
                    d3["생중량"], d3["지하부 길이"],
                    "생중량 (g)", "지하부 길이 (mm)", r3, p3, color="#2A9D8F"
                )
                if fig3:
                    st.plotly_chart(fig3, width="stretch")
                st.caption(f"r = {r3:.3f}  (p = {p3:.4f}, n = {len(d3)})")
            else:
                st.info("⚠ 생중량·지하부 길이 쌍이 2개 미만입니다.")
        else:
            st.info("⚠ '생중량' 또는 '지하부 길이' 열이 없습니다.")

        st.divider()

        # ── 지상부 길이 vs 지하부 길이 vs 잎 수 vs 지하부 길이 비교 판단 ──
        st.subheader(
            "🔍 지상부 길이와 지하부 길이 vs 잎 수와 지하부 길이 비교 중 "
            "개체의 생육 수준의 상관관계를 더 뚜렷하게 나타내는 비교 판단"
        )

        if (not growth_df.empty
                and "지상부 길이" in growth_df.columns
                and "잎 수" in growth_df.columns
                and "지하부 길이" in growth_df.columns):
            below = growth_df["지하부 길이"].dropna()
            above = growth_df["지상부 길이"].dropna()
            leaves = growth_df["잎 수"].dropna()

            # 비교 1: 지상부 길이 vs 지하부 길이
            ab = pd.DataFrame({"x": above, "y": below}).dropna()
            if len(ab) >= 2:
                r_ab, p_ab = stats.pearsonr(ab["x"], ab["y"])
            else:
                r_ab, p_ab = np.nan, np.nan

            # 비교 2: 잎 수 vs 지하부 길이
            lv = pd.DataFrame({"x": leaves, "y": below}).dropna()
            if len(lv) >= 2:
                r_lv, p_lv = stats.pearsonr(lv["x"], lv["y"])
            else:
                r_lv, p_lv = np.nan, np.nan

            # 비교표
            comp_rows = [
                {
                    "비교 쌍": "지상부 길이 vs 지하부 길이",
                    "피어슨 r": round(r_ab, 3) if not np.isnan(r_ab) else "계산 불가",
                    "p값": round(p_ab, 4) if not np.isnan(p_ab) else "계산 불가",
                    "n": len(ab),
                },
                {
                    "비교 쌍": "잎 수 vs 지하부 길이",
                    "피어슨 r": round(r_lv, 3) if not np.isnan(r_lv) else "계산 불가",
                    "p값": round(p_lv, 4) if not np.isnan(p_lv) else "계산 불가",
                    "n": len(lv),
                },
            ]
            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, width="stretch", use_container_width=False)

            # 판단 문장
            both_valid = (not np.isnan(r_ab)) and (not np.isnan(r_lv))
            if both_valid:
                stronger_pair = (
                    "지상부 길이 vs 지하부 길이"
                    if abs(r_ab) > abs(r_lv)
                    else ("잎 수 vs 지하부 길이"
                          if abs(r_lv) > abs(r_ab)
                          else "두 비교의 |r| 값이 같음")
                )
                stronger_r = max(abs(r_ab), abs(r_lv))
                st.markdown(f"""
**판단**

- **|r|이 더 큰 비교**: **{stronger_pair}** (|r| = {stronger_r:.3f})
- 지상부 길이와 지하부 길이의 r = {r_ab:.3f}
- 잎 수와 지하부 길이의 r = {r_lv:.3f}

→ |r|이 클수록 두 지표 사이의 선형 관계가 더 뚜렷하다는 뜻이므로,
**현재 데이터에서는 {stronger_pair} 쪽이 개체의 생육 수준 차이를 더 잘 드러내는 비교**로 볼 수 있습니다.

⚠ r만으로 "어느 지표가 더 중요하다"고 단정할 수는 없으며,
p값·n·산점도의 형태(비선형 패턴, 이상치)도 함께 확인해야 합니다.
                """.strip())
            else:
                st.info(
                    "⚠ 두 비교를 모두 계산하기에는 데이터가 충분하지 않습니다. "
                    "(각 비교별 쌍이 2개 미만)"
                )

            # CV(변동성) 보조 비교
            below_cv = (below.std(ddof=1) / below.mean()) if len(below) >= 2 and below.mean() > 0 else np.nan
            above_cv = (above.std(ddof=1) / above.mean()) if len(above) >= 2 and above.mean() > 0 else np.nan
            leaves_cv = (leaves.std(ddof=1) / leaves.mean()) if len(leaves) >= 2 and leaves.mean() > 0 else np.nan

            cv_valid = all(cv is not np.nan for cv in [below_cv, above_cv, leaves_cv])
            if cv_valid:
                cv_rows = [
                    {"지표": "지상부 길이 (CV)", "변동계수(CV)": round(above_cv, 3)},
                    {"지표": "지하부 길이 (CV)", "변동계수(CV)": round(below_cv, 3)},
                    {"지표": "잎 수 (CV)", "변동계수(CV)": round(leaves_cv, 3)},
                ]
                cv_df = pd.DataFrame(cv_rows)
                st.divider()
                st.caption("보조 지표: 변동계수(CV) 비교")
                st.dataframe(cv_df, width="stretch", use_container_width=False)
                cv_sorted = sorted(cv_rows, key=lambda t: t["변동계수(CV)"], reverse=True)
                best_cv_name = cv_sorted[0]["지표"]
                best_cv_val = cv_sorted[0]["변동계수(CV)"]
                st.caption(
                    f"변동계수(CV = 표준편차 ÷ 평균)가 가장 큰 지표는 **{best_cv_name}** "
                    f"= {best_cv_val:.3f}. "
                    "CV가 클수록 개체 간 차이가 크므로 생육 수준을 더 뚜렷하게 구분할 가능성이 높습니다."
                )
                st.caption(
                    "⚠ CV는 '개체 간 차이가 얼마나 큰지'를 보는 보조 지표이며, "
                    "생물학적 의미를 직접 판정하는 기준은 아닙니다."
                )
            else:
                st.info("⚠ CV 계산에 필요한 모든 지표가 갖춰지지 않았습니다.")
        else:
            missing = []
            if "지상부 길이" not in growth_df.columns:
                missing.append("지상부 길이")
            if "잎 수" not in growth_df.columns:
                missing.append("잎 수")
            if "지하부 길이" not in growth_df.columns:
                missing.append("지하부 길이")
            st.info(
                f"⚠ 다음 열이 없어 비교 판단을 표시하지 못했습니다: "
                f"{', '.join(missing) if missing else '데이터 없음'}"
            )

        with st.expander("📌 참고자료: 통계 용어 간단 설명"):
            st.markdown("""
- **평균(mean)**: 모든 값을 더해 개수로 나눈 값. 데이터의 '대표값' 중 하나.  
- **표준편차(standard deviation)**: 값들이 평균에서 얼마나 흩어져 있는지를 나타냄.
  클수록 개체 간 차이가 큼.  
- **변동계수(CV, coefficient of variation)**: 표준편차를 평균으로 나눈 값.
  단위가 다른 지표끼리 '흩어짐 정도'를 비교할 때 사용.  
- **피어슨 상관계수(Pearson r)**: 두 변수가 함께 증가/감소하는 직선 관계의 강도(-1 ~ 1).  
- **p값(p-value)**: '실제로는 관계가 없는데 우연히 이 정도 상관이 나왔을 확률'.
  보통 0.05 미만이면 통계적으로 의미 있다고 봄.
            """.strip())
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ─────────────────────────────────────────────
# 다운로드 버튼
# ─────────────────────────────────────────────
st.divider()
st.caption("📥 분석 결과 다운로드")
if not env_df.empty:
    csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "환경 데이터 다운로드 (utf-8-sig)",
        csv_bytes,
        file_name=f"{school_name}_환경_표준화.csv",
        mime="text/csv"
    )
if not growth_df.empty:
    csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "생육 데이터 다운로드 (utf-8-sig)",
        csv_bytes,
        file_name=f"{school_name}_생육_표준화.csv",
        mime="text/csv"
    )
