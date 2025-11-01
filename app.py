import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go
import anndata as ad, scanpy as sc
from pathlib import Path
from dri_core import compute_dri, annotate_tme

st.set_page_config(page_title="scProx — Precision Oncology Dashboard", layout="wide")
st.title("scProx — Precision Oncology Dashboard")

# ---- Sidebar ----
st.sidebar.header("Controls")

# Where the demo data lives in the repo (you pushed under data/raw/...)
SECTION_PATHS = {
    "Section 1": "data/raw/section1",
    "Section 2": "data/raw/section2",
}

data_mode = st.sidebar.selectbox(
    "Data source",
    ["Bundled demo data", "Upload .h5ad"],
    index=0,
)

adata = None

def read_h5ad_uploaded(uploaded_file):
    # anndata prefers a filename or h5py object; write to a temp file for reliability
    import tempfile
    from pathlib import Path
    tmp = tempfile.NamedTemporaryFile(suffix=".h5ad", delete=False)
    tmp.write(uploaded_file.getvalue())
    tmp.flush(); tmp.close()
    try:
        return ad.read_h5ad(tmp.name)
    finally:
        # clean up best-effort
        try:
            Path(tmp.name).unlink(missing_ok=True)
        except Exception:
            pass

if data_mode == "Bundled demo data":
    # Pick a section and auto-detect the first .h5ad in that folder
    section = st.sidebar.selectbox("Slide section", list(SECTION_PATHS.keys()))
    folder = Path(SECTION_PATHS[section])
    h5ads = sorted(folder.glob("*.h5ad"))
    if not h5ads:
        st.sidebar.error(f"No .h5ad found in {folder}")
    else:
        pick = st.sidebar.selectbox("Select file", [p.name for p in h5ads])
        adata = ad.read_h5ad(folder / pick)

elif data_mode == "Upload .h5ad":
    up = st.sidebar.file_uploader("Upload AnnData (.h5ad)", type=["h5ad"])
    if up is not None:
        try:
            adata = read_h5ad_uploaded(up)
        except Exception as ex:
            st.sidebar.error(f"Could not read uploaded file: {ex}")

targets = ["ESR1", "PNCK", "GAK", "BUB1", "BUB1B"]
e3s = ["CRBN", "VHL", "MDM2", "RNF114", "DCAF15"]
tg = st.sidebar.selectbox("Target", targets, index=0)
e3 = st.sidebar.selectbox("E3 ligase", e3s, index=0)
w_local = st.sidebar.slider("Local weight", 0.0, 1.0, 0.6, 0.05)
w_nbr = 1.0 - w_local
st.sidebar.caption(f"Neighbor weight = {w_nbr:.2f}")
run = st.sidebar.button("Compute / Load DRI")

# ---- Layout ----
c1, c2, c3 = st.columns([1.2, 1.1, 0.9])

with c1:
    st.subheader("Spatial panel")
    if adata is None:
        st.info("Load a .h5ad to show per-spot maps.")
    else:
        if run or "dri" not in adata.obs:
            try:
                adata.obs["dri"] = compute_dri(adata, tg, e3, w_local=w_local, w_nbr=w_nbr)
                st.success("DRI computed and stored in adata.obs['dri']")
            except Exception as ex:
                st.error(f"DRI failed: {ex}")

        if "spatial" in adata.obsm:
            coords = adata.obsm["spatial"]
            df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "DRI": adata.obs.get("dri", 0)})
            fig = px.scatter(df, x="x", y="y", color="DRI", color_continuous_scale="Plasma", render_mode="webgl")
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=450, margin=dict(l=10, r=10, b=10, t=30))
            st.plotly_chart(fig, use_container_width=True)
        elif "dri" in adata.obs:
            st.plotly_chart(px.histogram(adata.obs["dri"], nbins=40, title="DRI distribution"), use_container_width=True)

with c2:
    st.subheader("Co-localization proxy & signatures")
    if adata is not None:
        present = [g for g in [tg, e3] if g in adata.var_names]
        if present:
            X = adata[:, present].X
            X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
            means = X.mean(axis=0)
            st.plotly_chart(px.bar(x=present, y=means, labels={"x": "Gene", "y": "Mean expr"},
                                   title="Mean expression"), use_container_width=True)

        markers = {
            "Tumor": ["EPCAM", "KRT8", "KRT18", "KRT19"],
            "Stroma": ["COL1A1", "DCN", "LUM"],
            "Immune": ["PTPRC", "CD3D", "MS4A1", "LYZ"],
        }
        try:
            adata = annotate_tme(adata, markers)
            counts = adata.obs["dominant_TME"].value_counts().sort_index()
            st.plotly_chart(px.bar(counts, title="Dominant TME per spot"), use_container_width=True)
        except Exception as ex:
            st.info(f"TME scoring skipped: {ex}")
    else:
        st.caption("Load data to view expression and signatures.")

with c3:
    st.subheader("DRI Gauge & Export")
    mean_dri = float(np.nanmean(adata.obs["dri"])) if (adata is not None and "dri" in adata.obs) else 0.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(mean_dri * 100, 1),
        number={"suffix": " / 100"},
        title={"text": f"Mean DRI — {tg}×{e3}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.25},
            "steps": [
                {"range": [0, 40], "color": "#7a1e1e"},
                {"range": [40, 70], "color": "#a0781c"},
                {"range": [70, 100], "color": "#208f47"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=10, r=10, b=10, t=30))
    st.plotly_chart(fig, use_container_width=True)

    if adata is not None and "dri" in adata.obs:
        out = pd.DataFrame({
            "target": [tg],
            "e3": [e3],
            "w_local": [w_local],
            "w_nbr": [w_nbr],
            "mean_dri": [mean_dri],
        })
        st.download_button("Export selection (CSV)", out.to_csv(index=False), "scprox_selection.csv", "text/csv")
