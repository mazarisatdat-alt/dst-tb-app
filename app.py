import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Load model & data ─────────────────────────────────────────
@st.cache_resource
def load_assets():
    model  = joblib.load("model_GNN_LightGBM_DST.pkl")
    scaler = joblib.load("scaler_DST.pkl")
    df_ref = pd.read_csv("dominant_ALL.csv")
    df_emb = pd.read_csv("drug_embeddings_GNN_v2.csv")
    return model, scaler, df_ref, df_emb

model, scaler, df_ref, df_emb = load_assets()

gnn_cols     = [f"GNN_emb_{i+1}" for i in range(32)]
genomic_cols = ["Sensitivity_num", "Specificity_num", "PPV_num",
                "Present_pheno_R", "Present_pheno_S", "Dominance_Score"]
DRUGS        = ["RIF", "INH", "EMB", "PZA"]

# ── Fungsi prediksi DRI ───────────────────────────────────────
def prediksi_dri(profil_mutasi):
    hasil = {}
    for drug, mutasi_list in profil_mutasi.items():
        emb_row    = df_emb[df_emb["Drug"] == drug]
        gnn_values = emb_row[gnn_cols].values[0]

        if not mutasi_list:
            hasil[drug] = {
                "DRI"           : 0.0,
                "Status"        : "Sensitif",
                "Detail"        : [],
                "Tidak_Dikenal" : []
            }
            continue

        detail        = []
        tidak_dikenal = []

        for mutasi in mutasi_list:
            row = df_ref[
                (df_ref["Drug"] == drug) &
                (df_ref["Variant"] == mutasi)
            ]
            if row.empty:
                tidak_dikenal.append(mutasi)
                continue
            genomic_vals = row[genomic_cols].values[0]
            fitur        = np.concatenate([genomic_vals, gnn_values])
            fitur_scaled = scaler.transform(fitur.reshape(1, -1))
            prob         = float(model.predict_proba(fitur_scaled)[0][1])
            detail.append({"variant": mutasi, "dri": round(prob, 4)})

        if not detail and tidak_dikenal:
            hasil[drug] = {
                "DRI"           : None,
                "Status"        : "Tidak Dikenal",
                "Detail"        : [],
                "Tidak_Dikenal" : tidak_dikenal
            }
            continue

        dri_final = max(d["dri"] for d in detail) if detail else 0.0

        if dri_final >= 0.7:
            status = "Resisten"
        elif dri_final >= 0.4:
            status = "Borderline"
        else:
            status = "Sensitif"

        hasil[drug] = {
            "DRI"           : dri_final,
            "Status"        : status,
            "Detail"        : detail,
            "Tidak_Dikenal" : tidak_dikenal
        }

    resisten = [d for d, v in hasil.items() if v["Status"] == "Resisten"]
    if "RIF" in resisten and "INH" in resisten:
        kesimpulan  = "MDR-TB Terdeteksi"
        rekomendasi = ("Pasien resisten terhadap Rifampisin dan Isoniazid. "
                       "Disarankan beralih ke regimen pengobatan lini kedua "
                       "sesuai panduan WHO.")
        warna_kes   = "error"
    elif resisten:
        kesimpulan  = f"Resistensi Parsial ({', '.join(resisten)})"
        rekomendasi = ("Ditemukan resistensi pada satu atau lebih obat lini pertama. "
                       "Pertimbangkan penyesuaian regimen terapi.")
        warna_kes   = "warning"
    else:
        kesimpulan  = "TB Sensitif Obat"
        rekomendasi = ("Tidak ditemukan mutasi resisten. Regimen standar lini "
                       "pertama (HRZE) dapat dilanjutkan.")
        warna_kes   = "success"

    return hasil, kesimpulan, rekomendasi, warna_kes

# ── UI ────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Sistem DST-TB",
    page_icon  = "🧬",
    layout     = "wide"
)

st.markdown("## 🧬 Sistem DST-TB Berbasis GNN-LightGBM")
st.markdown("**Drug Susceptibility Testing — Prediksi Resistensi MDR-TB**")
st.markdown("*Input profil mutasi dari hasil TCM / Line Probe Assay*")
st.divider()

# ── INPUT ID PASIEN ───────────────────────────────────────────
st.markdown("### Identitas Pasien")
col_id1, col_id2 = st.columns([1, 2])
with col_id1:
    patient_id = st.text_input(
        label       = "ID Pasien",
        placeholder = "Contoh: MDR-2026-001",
        help        = "Masukkan ID pasien sesuai rekam medis"
    )
if patient_id:
    st.success(f"✓ Pasien: **{patient_id}**")

st.divider()

# ── INPUT MUTASI ──────────────────────────────────────────────
st.markdown("### Input Profil Mutasi Pasien")
st.caption("Ketik nama mutasi satu per baris. Kosongkan jika tidak ada mutasi terdeteksi.")

warna_obat = {"RIF": "🔴", "INH": "🟢", "EMB": "🔵", "PZA": "🟡"}
hints = {
    drug: df_ref[df_ref["Drug"] == drug]["Variant"].tolist()
    for drug in DRUGS
}

col1, col2    = st.columns(2)
profil_mutasi = {}

for i, drug in enumerate(DRUGS):
    col = col1 if i % 2 == 0 else col2
    with col:
        st.markdown(f"**{warna_obat[drug]} {drug}**")
        raw_input = st.text_area(
            label            = f"Mutasi {drug}",
            placeholder      = f"Ketik nama mutasi, satu per baris.\nContoh:\n{hints[drug][0] if hints[drug] else ''}",
            height           = 120,
            key              = f"input_{drug}",
            label_visibility = "collapsed"
        )
        mutasi_list = [m.strip() for m in raw_input.strip().split("\n")
                       if m.strip()]
        profil_mutasi[drug] = mutasi_list
        if mutasi_list:
            st.caption(f"✓ {len(mutasi_list)} mutasi diinput")

st.divider()

# ── TOMBOL ────────────────────────────────────────────────────
col_btn1, col_btn2, _ = st.columns([1, 1, 4])
with col_btn1:
    prediksi_btn = st.button(
        "🔬 Prediksi DRI", type="primary", use_container_width=True)
with col_btn2:
    reset_btn = st.button("🔄 Reset", use_container_width=True)

if reset_btn:
    st.rerun()

# ── HASIL ─────────────────────────────────────────────────────
if prediksi_btn:
    if not patient_id.strip():
        st.warning("⚠️ Harap masukkan ID Pasien terlebih dahulu.")
        st.stop()

    with st.spinner("Memproses prediksi..."):
        hasil, kesimpulan, rekomendasi, warna_kes = prediksi_dri(profil_mutasi)

    st.markdown(f"### Hasil Prediksi DRI — Pasien: `{patient_id}`")

    kartu_style = {
        "Resisten"      : ("background:#C62828; color:#FFFFFF;", "🔴"),
        "Borderline"    : ("background:#E65100; color:#FFFFFF;", "🟡"),
        "Sensitif"      : ("background:#2E7D32; color:#FFFFFF;", "🟢"),
        "Tidak Dikenal" : ("background:#424242; color:#FFFFFF;", "⚫"),
    }

    cols = st.columns(4)
    for col, drug in zip(cols, DRUGS):
        res          = hasil.get(drug, {"DRI": 0.0, "Status": "Sensitif",
                                        "Detail": [], "Tidak_Dikenal": []})
        style, emoji = kartu_style.get(res["Status"],
                       ("background:#2E7D32; color:#FFFFFF;", "🟢"))
        dri_text     = f"{res['DRI']:.2f}" if res["DRI"] is not None else "N/A"

        with col:
            st.markdown(
                f"""<div style="{style} border-radius:12px; padding:20px;
                    text-align:center; margin-bottom:8px;">
                    <h3 style="margin:0; color:inherit">{drug}</h3>
                    <h1 style="margin:10px 0; color:inherit">{dri_text}</h1>
                    <p style="margin:0; font-weight:bold; color:inherit">
                        {emoji} {res["Status"]}
                    </p>
                </div>""",
                unsafe_allow_html=True
            )
            if res.get("Detail"):
                with st.expander("Detail mutasi"):
                    for d in res["Detail"]:
                        st.write(f"• `{d['variant']}` → DRI: **{d['dri']:.4f}**")
            if res.get("Tidak_Dikenal"):
                with st.expander("⚠️ Mutasi tidak dikenal"):
                    for m in res["Tidak_Dikenal"]:
                        st.write(f"• `{m}`")
                    st.caption("Tidak ada di referensi WHO Catalogue. "
                               "Disarankan konfirmasi DST konvensional.")

    st.divider()

    st.markdown("### Kesimpulan Klinis")
    if warna_kes == "error":
        st.error(f"**🚨 {kesimpulan}**")
    elif warna_kes == "warning":
        st.warning(f"**⚠️ {kesimpulan}**")
    else:
        st.success(f"**✅ {kesimpulan}**")

    st.info(f"**Rekomendasi:** {rekomendasi}")
    st.caption("*Hasil prediksi bersifat pendukung keputusan klinis. "
               "Konfirmasi oleh tenaga medis tetap diperlukan. "
               "Sistem hanya mencakup mutasi dominan yang tervalidasi "
               "WHO Catalogue Second Edition 2023.*")

    st.divider()
    st.markdown("### Ringkasan Hasil")
    df_ringkasan = pd.DataFrame([
        {
            "ID Pasien"           : patient_id,
            "Obat"                : drug,
            "DRI"                 : f"{hasil[drug]['DRI']:.4f}"
                                     if hasil[drug]["DRI"] is not None else "N/A",
            "Status"              : hasil[drug]["Status"],
            "Mutasi Terdeteksi"   : ", ".join(
                [d["variant"] for d in hasil[drug]["Detail"]]
            ) or "Tidak ada",
            "Mutasi Tidak Dikenal": ", ".join(
                hasil[drug].get("Tidak_Dikenal", [])
            ) or "-"
        }
        for drug in DRUGS
    ])
    st.dataframe(df_ringkasan, use_container_width=True, hide_index=True)