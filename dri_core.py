import numpy as np
import scanpy as sc

def compute_dri(adata, target_gene, e3_gene, w_local=0.6, w_nbr=0.4):
    """
    Compute Degrader Readiness Index (DRI) using spatial local and neighbor terms.
    Formula: DRI = w_local*(z_target*z_e3)_local + w_nbr*(z_target*z_e3)_neighbors
    """
    # Ensure spatial connectivity info is present
    if 'spatial_connectivities' not in adata.obsp:
        try:
            import squidpy as sq
            sq.gr.spatial_neighbors(adata, coord_type='generic')
        except Exception as e:
            raise RuntimeError("No spatial_connectivities found and could not build neighbors.") from e

    # Extract gene expression
    tgt = adata[:, target_gene].X
    e3  = adata[:, e3_gene].X

    to_array = lambda x: x.toarray().ravel() if hasattr(x, "toarray") else np.asarray(x).ravel()
    tgt, e3 = to_array(tgt), to_array(e3)

    # Z-score normalization
    from scipy import stats
    zt = np.nan_to_num(stats.zscore(tgt, nan_policy="omit"))
    ze = np.nan_to_num(stats.zscore(e3,  nan_policy="omit"))

    # Local term
    local = zt * ze

    # Neighbor term using spatial adjacency
    A = adata.obsp['spatial_connectivities']
    deg = A.sum(axis=1).A1 + 1e-10
    nbr_zt = (A @ zt) / deg
    nbr_ze = (A @ ze) / deg
    nbr = nbr_zt * nbr_ze

    # Weighted sum
    dri = w_local * local + w_nbr * nbr
    return dri


def annotate_tme(adata, markers_dict):
    """
    Score TME signatures and assign dominant TME class.
    """
    for ct, genes in markers_dict.items():
        present = [g for g in genes if g in adata.var_names]
        if not present:
            adata.obs[f"{ct}_score"] = 0.0
            continue
        sc.tl.score_genes(adata, gene_list=present, score_name=f"{ct}_score", use_raw=False)

    cols = [f"{ct}_score" for ct in markers_dict]
    adata.obs["dominant_TME"] = adata.obs[cols].idxmax(axis=1).str.replace("_score", "")
    return adata
