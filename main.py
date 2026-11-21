import math
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─────────────────────────────────────────────
# 업로드된 파일의 서명 (이름+크기로 구성 → id 속성 없이도 동작)
# ─────────────────────────────────────────────
def _file_sig(f):
    """파일 변경 감지를 위한 서명. file.id 없이도 동작."""
    if f is None:
        return ""
    return f.name.replace(".csv", "") + "|" + str(f.size)

# ─────────────────────────────────────────────
# CSV 읽기 (인코딩 자동 감지)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv(file):
    """file.getvalue()로 바이트를 꺼낸 뒤 utf-8-sig → cp949 순으로 시도."""
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ─────────────────────────────────────────────
# 열 이름 표준화  (KeyError 방지 — 학교마다 다른 열 이름 대응)
# ─────────────────────────────────────────────
def std_cols(df):
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
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce"
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    # 생중량 열이 없으면 지상부 질량 + 지하부 질량으로 계산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    # 시간 열 파싱 (날짜_시각 형식 자동 처리)
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df

# ─────────────────────────────────────────────
# Streamlit 앱
# ─────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

# 한글 폰트 CSS
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

# ── 사이드바: 파일 업로드 ──────────────────────
env_file    = st.sidebar.file_uploader("🌡️ 환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("🌱 생육 결과 데이터 CSV", type="csv")

st.sidebar.markdown("---")
st.sidebar.subheader("📐 생육 측정 항목 선택")
growth_cols_all = ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
selected_cols = []
for col in growth_cols_all:
    if st.sidebar.checkbox(col, value=True, key=f"chk_{col}"):
        selected_cols.append(col)
if not selected_cols:
    st.sidebar.warning("하나 이상의 측정 항목을 선택해 주세요.")

# ── 데이터 한 번만 읽기 (파일 서명으로 변경 감지, id 속성 사용 안 함) ──
env_sig  = _file_sig(env_file)
grw_sig  = _file_sig(growth_file)

_prev_env = st.session_state.get("_env_sig",  "")
_prev_grw = st.session_state.get("_grw_sig",  "")

if env_file and env_sig != _prev_env:
    with st.spinner("환경 데이터를 읽는 중..."):
        try:
            st.session_state["env_df"]   = std_cols(read_csv(env_file))
            st.session_state["_env_sig"] = env_sig
        except Exception as e:
            st.warning(f"환경 CSV를 읽지 못했어요: {e}")
            st.session_state["env_df"] = pd.DataFrame()
elif not env_file:
    st.session_state["env_df"] = pd.DataFrame()

if growth_file and grw_sig != _prev_grw:
    with st.spinner("생육 결과 데이터를 읽는 중..."):
        try:
            st.session_state["growth_df"]  = std_cols(read_csv(growth_file))
            st.session_state["_grw_sig"]   = grw_sig
        except Exception as e:
            st.warning(f"생육 CSV를 읽지 못했어요: {e}")
            st.session_state["growth_df"] = pd.DataFrame()
elif not growth_file:
    st.session_state["growth_df"] = pd.DataFrame()

env_df    = st.session_state.get("env_df",    pd.DataFrame())
growth_df = st.session_state.get("growth_df", pd.DataFrame())

# 학교명: 업로드한 파일명에서 추출 (하드코딩 금지)
env_school    = env_file.name.replace(".csv", "").strip() if env_file else "업로드된 환경 CSV"
growth_school = growth_file.name.replace(".csv", "").strip() if growth_file else "업로드된 생육 CSV"

ts = st.tabs([
    "📖 실험 개요",
    "🌡️ 환경 데이터",
    "📊 생육 결과",
    "📥 데이터 다운로드",
])

# ══════════════════════════════════════════════
# Tab 1 — 실험 개요
# ══════════════════════════════════════════════
with ts[0]:
    try:
        if env_df.empty and growth_df.empty:
            st.info("🌿 사이드바에서 CSV 파일 2개를 업로드해 주세요.")
        else:
            grw_present = not growth_df.empty
            env_present = not env_df.empty

            st.subheader(f"🔬 {growth_school} — 극지식물 생육 비교 실험")
            st.markdown(
                """
**연구 배경 및 목적**

학교 온실/실험실에서 재배 중인 극지식물의 **개체별 생육 차이**를 분석합니다.
같은 환경에서 자라도 개체마다 잎 수, 지상부 길이, 생체중량(생중량)이 다를 수 있습니다.
이 대시보드는 개체 간 차이를 숫자(평균 · 표준편차 · 상관계수)와 그래프로 보여 줍니다.

> 💡 **통계 용어 간단 설명**
> - **평균(mean)**: 여러 값을 더해 개수로 나눈 대표값
> - **표준편차(STD)**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 수치 (클수록 편차가 큼)
> - **변동계수(CV)**: 표준편차를 평균으로 나눈 값. 단위가 다른 그룹끼리 흩어진 정도를 비교할 때 사용
> - **상관계수(r)**: 두 변수의 관계가 얼마나 직선적인지 −1~1 사이 값으로 표시 (1에 가까울수록 함께 증가)
""",
            )

            # 데이터 요약 표
            if grw_present or env_present:
                summary_rows = []
                if grw_present:
                    n_grw = len(growth_df)
                    summary_rows.append(
                        ["생육 결과 데이터", f"{n_grw}개체",
                         "개체별 생육 측정값",
                         "개체번호, 잎 수, 지상부·지하부 길이, 생중량, 지상부·지하부 질량"]
                    )
                if env_present:
                    n_env = len(env_df)
                    env_cols = [c for c in ["temperature","humidity","ph","ec"] if c in env_df.columns]
                    summary_rows.append(
                        ["환경 데이터", f"{n_env}건",
                         "시간별 환경 측정값",
                         ", ".join(env_cols) if env_cols else "(측정 열 없음)"]
                    )
                st.subheader("📋 데이터 요약")
                st.dataframe(
                    pd.DataFrame(summary_rows,
                                 columns=["데이터", "수량", "내용", "주요 열"]),
                    width="stretch",
                )

            # 지표 카드 4개
            env_vals = (
                env_df.dropna(subset=["temperature","humidity"])
                .drop_duplicates()
                if env_present else None
            )
            grw_vals = (
                growth_df.dropna(subset=["생중량"])
                if grw_present else None
            )

            ta = round(env_vals["temperature"].mean(), 1) if (env_vals is not None and len(env_vals) >= 2) else None
            ha = round(env_vals["humidity"].mean(),   1) if (env_vals is not None and len(env_vals) >= 2) else None
            ga = round(grw_vals["생중량"].mean(),     3) if (grw_vals is not None and len(grw_vals) >= 2) else None

            n_grw = len(grw_vals) if grw_vals is not None else 0

            st.metric("🧪 총 개체수",       n_grw                    if n_grw > 0 else "—")
            st.metric("🌡️ 평균 온도 (°C)",  ta                       if ta   is not None else "—")
            st.metric("💧 평균 습도 (%)",    ha                       if ha   is not None else "—")
            st.metric("⚖️ 평균 생중량 (g)", ga                       if ga   is not None else "—")

            # 생육 데이터 요약 통계 표
            if grw_present and len(growth_df) > 0:
                st.subheader("📊 생육 측정 항목 요약 (평균 · 표준편차 · 최소 · 최대)")
                num_cols = [c for c in ["잎 수","지상부 길이","지하부 길이","생중량","지상부 질량","지하부 질량"]
                            if c in growth_df.columns]
                if num_cols:
                    summ = growth_df[num_cols].describe().T.loc[:, ["mean","std","min","max"]].round(2)
                    summ.columns = ["평균", "표준편차", "최소", "최대"]
                    st.dataframe(summ, width="stretch")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ══════════════════════════════════════════════
# Tab 2 — 환경 데이터
# ══════════════════════════════════════════════
with ts[1]:
    try:
        if env_df.empty:
            st.info("🌿 사이드바에서 환경 데이터 CSV를 업로드해 주세요.")
        else:
            st.subheader(f"🌡️ {env_school} — 환경 데이터 분석")
            env_num = [c for c in ["temperature","humidity","ph","ec"] if c in env_df.columns]

            if not env_num:
                st.info("환경 데이터에서 temperature / humidity / ph / ec 열을 찾지 못했어요. 학교별 CSV 형식이 다를 수 있어요.")
            else:
                env_clean = env_df.dropna(subset=env_num)

                # 요약 표
                st.subheader("📋 환경 요약 (평균 · 최소 · 최대)")
                if len(env_clean) >= 2:
                    env_summary = env_clean[env_num].agg(["mean","min","max"]).T.round(2)
                    env_summary.columns = ["평균", "최소", "최대"]
                    st.dataframe(env_summary, width="stretch")

                # 날짜별 시계열 (일별 평균)
                st.subheader("📈 날짜별 환경 변화 (일별 평균)")
                env_clean = env_clean.copy()
                env_clean["date"] = env_clean["time"].dt.date
                daily = env_clean.groupby("date")[env_num].mean().reset_index()

                # 온도
                if "temperature" in daily.columns and len(daily) >= 2:
                    fig_t = go.Figure()
                    fig_t.add_trace(go.Scatter(
                        x=daily["date"], y=daily["temperature"],
                        mode="lines+markers", name="온도 (°C)",
                        line=dict(color="#e74c3c", width=2),
                        marker=dict(size=4),
                    ))
                    fig_t.update_layout(
                        title="날짜별 평균 온도",
                        xaxis_title="날짜", yaxis_title="온도 (°C)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        hovermode="x unified", height=280,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_t, width="stretch")

                # 습도
                if "humidity" in daily.columns and len(daily) >= 2:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(
                        x=daily["date"], y=daily["humidity"],
                        mode="lines+markers", name="습도 (%)",
                        line=dict(color="#3498db", width=2),
                        marker=dict(size=4),
                    ))
                    fig_h.update_layout(
                        title="날짜별 평균 습도",
                        xaxis_title="날짜", yaxis_title="습도 (%)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        hovermode="x unified", height=280,
                        margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_h, width="stretch")

                # pH · EC (두 열 중 하나라도 있을 때만)
                if {"ph","ec"} & set(daily.columns):
                    fig_p = go.Figure()
                    if "ph" in daily.columns and len(daily) >= 2:
                        fig_p.add_trace(go.Scatter(
                            x=daily["date"], y=daily["ph"],
                            mode="lines+markers", name="pH",
                            line=dict(color="#2ecc71", width=2),
                            marker=dict(size=4),
                        ))
                    if "ec" in daily.columns and len(daily) >= 2:
                        fig_p.add_trace(go.Scatter(
                            x=daily["date"], y=daily["ec"],
                            mode="lines+markers", name="EC (mS/cm)",
                            line=dict(color="#9b59b6", width=2),
                            marker=dict(size=4),
                        ))
                    fig_p.update_layout(
                        title="날짜별 pH · EC 변화",
                        xaxis_title="날짜", yaxis_title="값",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        hovermode="x unified", height=280,
                        margin=dict(l=40, r=20, t=40, b=40),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_p, width="stretch")

                # 원본 테이블 + CSV 다운로드
                with st.expander("📄 환경 데이터 원본 테이블"):
                    st.dataframe(env_df.sort_values("time"), width="stretch")
                    csv_bytes = env_df.sort_values("time").to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ 환경 CSV 다운로드",
                        csv_bytes,
                        file_name=f"{env_school}_환경정리.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ══════════════════════════════════════════════
# Tab 3 — 생육 결과
# ══════════════════════════════════════════════
with ts[2]:
    try:
        if growth_df.empty:
            st.info("🌿 사이드바에서 생육 결과 CSV를 업로드해 주세요.")
        else:
            st.subheader(f"📊 {growth_school} — 개체별 생육 결과 분석")
            g = growth_df.dropna(subset=["생중량"]).reset_index(drop=True)

            if len(g) < 2:
                st.info("생중량 값이 있는 개체가 2개 미만이에요. 데이터를 확인해 주세요.")
            else:
                n   = len(g)
                avg = round(g["생중량"].mean(), 3)
                std = round(g["생중량"].std(),  3)
                cv  = round(std / avg, 3) if avg > 0 else None
                mx  = round(g["생중량"].max(), 3)
                mn  = round(g["생중량"].min(), 3)

                idx_max = g["생중량"].idxmax()
                idx_min = g["생중량"].idxmin()
                id_max_raw = g.loc[idx_max, "개체번호"]
                id_min_raw = g.loc[idx_min, "개체번호"]

                # 개체번호: 정수면 정수 그대로, 글자면 글자 그대로
                id_max_disp = int(id_max_raw) if isinstance(id_max_raw, (int, np.integer)) else str(id_max_raw)
                id_min_disp = int(id_min_raw) if isinstance(id_min_raw, (int, np.integer)) else str(id_min_raw)

                # ─ 핵심 결과 카드
                st.markdown(f"**🥇 평균 생중량: {avg} g**   (표준편차 {std} g, CV {cv})")
                row = st.columns(4)
                row[0].metric("평균 생중량 (g)",        avg)
                row[1].metric("최대 생중량 (g)",        mx,  delta=f"개체 {id_max_disp}")
                row[2].metric("최소 생중량 (g)",        mn,  delta=f"개체 {id_min_disp}")
                row[3].metric("개체 수",                n)

                # ─ 개체별 생육 비교 막대 그래프 (공통 함수)
                def bar_fig(col, title, color, ylabel):
                    """col 열의 개체별 막대 그래프. 열이 없거나 데이터 부족 시 None 반환."""
                    if col not in g.columns:
                        return None
                    d = g.dropna(subset=[col]).copy()
                    if len(d) < 2:
                        return None
                    d = d.sort_values(col, ascending=True)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=d["개체번호"].astype(str),
                        y=d[col],
                        name=title,
                        marker_color=color,
                        marker_line_color="white",
                        marker_line_width=0.5,
                    ))
                    fig.update_layout(
                        title=title,
                        xaxis_title="개체번호",
                        yaxis_title=ylabel,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=300,
                        margin=dict(l=40, r=20, t=40, b=40),
                        showlegend=False,
                    )
                    return fig

                # 막대 그래프 4개
                st.subheader("📊 개체별 생육 비교")
                c1, c2 = st.columns(2)
                with c1:
                    bf = bar_fig("생중량",        "⭐ 평균 생중량 (g) — 개체별 비교", "#2ecc71", "생중량 (g)")
                    if bf: st.plotly_chart(bf, width="stretch")
                    else: st.info("생중량 데이터가 충분하지 않아요.")
                with c2:
                    lf = bar_fig("잎 수",         "평균 잎 수 (장)",               "#3498db", "잎 수 (장)")
                    if lf: st.plotly_chart(lf, width="stretch")
                    else: st.info("잎 수 데이터가 충분하지 않아요.")

                c3, c4 = st.columns(2)
                with c3:
                    sf = bar_fig("지상부 길이",   "평균 지상부 길이 (mm)",          "#e67e22", "지상부 길이 (mm)")
                    if sf: st.plotly_chart(sf, width="stretch")
                    else: st.info("지상부 길이 데이터가 충분하지 않아요.")
                with c4:
                    uf = bar_fig("지하부 길이",   "평균 지하부 길이 (mm)",          "#9b59b6", "지하부 길이 (mm)")
                    if uf: st.plotly_chart(uf, width="stretch")
                    else: st.info("지하부 길이 데이터가 충분하지 않아요.")

                # ─ 생중량 분포 (박스플롯)
                st.subheader("📦 생중량 분포 (박스플롯)")
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(
                    y=g["생중량"],
                    name="생중량 (g)",
                    fillcolor="#2ecc71",
                    marker=dict(color="#2ecc71"),
                    line=dict(color="#27ae60"),
                    boxpoints="outliers",
                    jitter=0.3,
                    pointpos=-1.8,
                ))
                fig_box.update_layout(
                    title=f"생중량 분포  (n={n}, 평균 {avg} g)",
                    yaxis_title="생중량 (g)",
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    height=280,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_box, width="stretch")

                # ─ 상위 25% vs 하위 25%
                q75 = g["생중량"].quantile(0.75)
                q25 = g["생중량"].quantile(0.25)
                high = g[g["생중량"] >= q75].sort_values("생중량", ascending=False)
                low  = g[g["생중량"] <= q25].sort_values("생중량", ascending=True)

                st.subheader("🏆 생중량 상위 25% vs 하위 25% 비교")
                hcols = st.columns(2)
                hcols[0].markdown(f"**생중량 상위 25%**  (기준: ≥ {round(q75, 2)} g,  {len(high)}개체)")
                hcols[1].markdown(f"**생중량 하위 25%**  (기준: ≤ {round(q25, 2)} g,  {len(low)}개체)")

                m_hi = st.columns(3)
                m_hi[0].metric("상위그룹 평균 생중량 (g)",
                               round(high["생중량"].mean(), 3) if len(high) > 0 else "—")
                m_hi[1].metric("상위그룹 평균 잎 수 (장)",
                               round(high["잎 수"].mean(), 1)   if "잎 수" in high.columns and len(high) > 0 else "—")
                m_hi[2].metric("상위그룹 평균 지상부 길이 (mm)",
                               round(high["지상부 길이"].mean(), 1) if "지상부 길이" in high.columns and len(high) > 0 else "—")

                m_lo = st.columns(3)
                m_lo[0].metric("하위그룹 평균 생중량 (g)",
                               round(low["생중량"].mean(), 3) if len(low) > 0 else "—")
                m_lo[1].metric("하위그룹 평균 잎 수 (장)",
                               round(low["잎 수"].mean(), 1)   if "잎 수" in low.columns and len(low) > 0 else "—")
                m_lo[2].metric("하위그룹 평균 지상부 길이 (mm)",
                               round(low["지상부 길이"].mean(), 1) if "지상부 길이" in low.columns and len(low) > 0 else "—")

                # ─ 상관관계 분석 (공통 함수): 추세선은 np.polyfit으로 직접 계산
                def scatter_fig(xcol, ycol, xtitle, ytitle, xcolor, ycolor):
                    if xcol not in g.columns or ycol not in g.columns:
                        return None
                    d = g.dropna(subset=[xcol, ycol])
                    if len(d) < 2:
                        return None
                    x = d[xcol].values
                    y = d[ycol].values
                    r  = round(np.corrcoef(x, y)[0, 1], 3)
                    m  = np.polyfit(x, y, 1)
                    line_x = np.linspace(x.min(), x.max(), 100)
                    line_y = np.polyval(m, line_x)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode="markers",
                        name=f"{xtitle} vs {ytitle}",
                        marker=dict(color=xcolor, size=8, line=dict(color="white", width=1)),
                        hovertemplate=f"{xtitle}: %{{x}}<br>{ytitle}: %{{y}}<extra></extra>",
                    ))
                    fig.add_trace(go.Scatter(
                        x=line_x, y=line_y,
                        mode="lines",
                        name=f"추세선  (r = {r})",
                        line=dict(color=ycolor, width=2, dash="dot"),
                    ))
                    fig.update_layout(
                        title=f"{xtitle} vs {ytitle}   (상관계수 r = {r})",
                        xaxis_title=xtitle,
                        yaxis_title=ytitle,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        height=380,
                        margin=dict(l=50, r=20, t=40, b=40),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    return fig

                st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")
                cp1, cp2 = st.columns(2)
                with cp1:
                    sc1 = scatter_fig("잎 수", "생중량", "잎 수 (장)", "생중량 (g)", "#3498db", "#e74c3c")
                    if sc1: st.plotly_chart(sc1, width="stretch")
                    else:   st.info("잎 수와 생중량 데이터가 충분하지 않아요.")
                with cp2:
                    sc2 = scatter_fig("지상부 길이", "생중량", "지상부 길이 (mm)", "생중량 (g)", "#2ecc71", "#8e44ad")
                    if sc2: st.plotly_chart(sc2, width="stretch")
                    else:   st.info("지상부 길이와 생중량 데이터가 충분하지 않아요.")

                st.info(
                    "💡 추세선은 np.polyfit으로 직접 계산한 선형 회귀선입니다. "
                    "상관계수(r)는 −1~1 사이 값으로, 1에 가까울수록 두 값이 함께 증가하는 경향이 강합니다."
                )

                # ─ 원본 데이터 + CSV 다운로드
                with st.expander("📄 생육 결과 데이터 원본"):
                    st.dataframe(growth_df.sort_values("개체번호"), width="stretch")
                    csv_g = growth_df.sort_values("개체번호").to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ 생육 결과 CSV 다운로드",
                        csv_g,
                        file_name=f"{growth_school}_생육정리.csv",
                        mime="text/csv",
                    )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ══════════════════════════════════════════════
# Tab 4 — 데이터 다운로드
# ══════════════════════════════════════════════
with ts[3]:
    try:
        st.subheader("📥 원본 데이터 다운로드")
        if env_df.empty:
            st.info("환경 데이터 CSV가 업로드되지 않았어요.")
        else:
            csv_e = env_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ 환경 데이터 CSV 다운로드",
                csv_e,
                file_name=f"{env_school}_환경원본.csv",
                mime="text/csv",
            )
        if growth_df.empty:
            st.info("생육 결과 CSV가 업로드되지 않았어요.")
        else:
            csv_g = growth_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ 생육 결과 CSV 다운로드",
                csv_g,
                file_name=f"{growth_school}_생육원본.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
