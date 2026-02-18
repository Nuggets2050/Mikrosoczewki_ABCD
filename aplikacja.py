import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ==========================================================
# ABCD MATRICES (REDUCED COORDINATES y, nθ)
# ==========================================================

def M_free_space(d, n):
    return np.array([
        [1.0, d / n],
        [0.0, 1.0]
    ])

def M_spherical(n1, n2, R, flip_R=False):
    if flip_R:
        R = -R
    C = (n1 - n2) / R
    return np.array([
        [1.0, 0.0],
        [C,   1.0]
    ])

def calculate_system(elements, R_value, flip_R=False):
    M_total = np.eye(2)

    for el in elements:
        if el[0] == "free":
            d, n = el[1], el[2]
            M = M_free_space(d, n)

        elif el[0] == "spherical":
            n1, n2 = el[1], el[2]
            M = M_spherical(n1, n2, R_value, flip_R)

        M_total = M @ M_total

    return M_total

# ==========================================================
# GAUSSIAN BEAM
# ==========================================================

def zR_um(w0, n, lambda0):
    return np.pi * n * w0**2 / lambda0

def q_waist(w0, n, lambda0):
    return 1j * zR_um(w0, n, lambda0)

def apply_abcd(M, q):
    A, B = M[0,0], M[0,1]
    C, D = M[1,0], M[1,1]
    return (A*q + B) / (C*q + D)

def w_from_q(q, n, lambda0):
    invq = 1.0 / q
    im = np.imag(invq)
    if np.abs(im) < 1e-20:
        im = -1e-20
    w2 = -lambda0 / (np.pi * n * im)
    return np.sqrt(max(w2, 0.0))

# ==========================================================
# FOCUS FINDING (REFERENCE PROGRAM METHOD)
# ==========================================================

def propagate_q(q, z):
    return q + z

def focus_condition(z, q_out):
    qz = propagate_q(q_out, z)
    return np.real(1.0 / qz)

def find_focus_position(q_out, z_min=0.0, z_max=20000.0):
    try:
        sol = root_scalar(
            focus_condition,
            args=(q_out,),
            bracket=[z_min, z_max],
            method="brentq"
        )
        if sol.converged:
            return sol.root
        else:
            return np.nan
    except ValueError:
        return np.nan

# ==========================================================
# MEDIUM HELPERS
# ==========================================================

def get_starting_medium(elements):
    if not elements:
        return 1.0

    first = elements[0]

    if first[0] == "free":
        return first[2]

    elif first[0] == "spherical":
        return first[1]

    return 1.0

def get_output_medium(elements):
    if not elements:
        return 1.0

    last = elements[-1]

    if last[0] == "free":
        return last[2]

    elif last[0] == "spherical":
        return last[2]

    return 1.0

def fill_gradient(
            ax,
            z,
            w,
            cmap='viridis',
            alpha=1.0,
            r_scale=2.5,     # ile szerokości wiązki pokazać w pionie
            z_res=400,
            r_res=300
        ):
            """
            Rysuje mapę intensywności wiązki Gaussa I(r,z)
            """

            z = np.asarray(z)
            w = np.abs(np.asarray(w))

            # ===== zakres radialny =====
            r_max = r_scale * np.nanmax(w)
            if not np.isfinite(r_max) or r_max <= 0:
                r_max = 1.0

            r = np.linspace(-r_max, r_max, r_res)
            z_dense = np.linspace(z.min(), z.max(), z_res)

            # ===== interpolacja w(z) =====
            w_interp = np.interp(z_dense, z, w)
            W = w_interp[np.newaxis, :]
            R = r[:, np.newaxis]

            # ===== intensywność Gaussa =====
            with np.errstate(divide='ignore', invalid='ignore'):
                I = np.exp(-2 * (R**2) / (W**2))

            I[~np.isfinite(I)] = 0.0

            # ===== rysowanie =====
            im = ax.pcolormesh(
                z_dense,
                r,
                I,
                shading='auto',
                cmap=cmap,
                vmin=0,
                vmax=1,
                alpha=alpha
            )

            ax.set_xlim(z.min(), z.max())
            ax.set_ylim(-r_max, r_max)
            ax.set_xlabel("z [µm]")
            ax.set_ylabel("r [µm]")

            return im
def auto_ylim_full_curve(ax, y, lower_pct=1, upper_pct=99, margin_frac=0.08):
    """
    Pokazuje całą krzywą, ale ignoruje ekstremalne wartości (np. asymptoty).
    Oś dostosowuje się automatycznie przy zmianie parametrów.
    """

    y = np.asarray(y)
    y = y[np.isfinite(y)]

    if len(y) == 0:
        return

    # percentyle zamiast min/max → odporność na asymptoty
    ymin = np.percentile(y, lower_pct)
    ymax = np.percentile(y, upper_pct)

    if ymin == ymax:
        ymin -= 1
        ymax += 1

    margin = margin_frac * (ymax - ymin)

    ax.set_ylim(ymin - margin, ymax + margin)

# ==========================================================
# STREAMLIT UI
# ==========================================================

st.set_page_config(layout="wide")
st.title("Metoda ABCD dla układów mikrooptycznych")

if "elements" not in st.session_state:
    st.session_state.elements = []

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.header("Budowa układu")

    element_type = st.selectbox(
        "Typ elementu",
        ["propagacja w ośrodku", "Powierzchnia sferyczna"]
    )

    if element_type == "propagacja w ośrodku":
        d = st.number_input("Odległość d [µm]", value=100.0)
        n = st.number_input("Współczynnik załamania n", value=1.0)

        if st.button("Dodaj propagacje w ośrodku"):
            st.session_state.elements.append(("free", d, n))

    if element_type == "Powierzchnia sferyczna":
        n1 = st.number_input("n1", value=1.5)
        n2 = st.number_input("n2", value=1.0)

        if st.button("Dodaj powierzchnię sferyczną"):
            st.session_state.elements.append(("spherical", n1, n2))

    flip_R = st.checkbox("Odwróć znak R", value=False)

    if st.button("Wyczyść układ"):
        st.session_state.elements = []

# ==========================================================
# UKŁAD
# ==========================================================

st.subheader("Układ optyczny")

if st.session_state.elements:
    for i, el in enumerate(st.session_state.elements):
        if el[0] == "free":
            st.write(f"{i+1}. propagacja na odległość — d={el[1]} µm w ośrodku o współczynniku załamania n={el[2]}")
        else:
            st.write(f"{i+1}. Powierzchnia sferyczna między ośrodkiem n1={el[1]} przed soczewką, a ośrodkiem n2={el[2]} za soczewką") 
else:
    st.info("Brak elementów")

# ==========================================================
# OBLICZENIA
# ==========================================================

if st.session_state.elements:

    st.divider()



    # ======================================================
    # GAUSSIAN BEAM
    # ======================================================

    st.divider()
    st.subheader("Propagacja wiązki")

   # ======================================================
    # LAYOUT GŁÓWNY
    # ======================================================

    left, right = st.columns([1.35, 1.0])


    # ======================================================
    # LEWA KOLUMNA — PROPAGACJA WIĄZKI
    # ======================================================

    with left:

        with st.container(border=True):

            st.subheader("Propagacja wiązki")
            lambda0 = st.slider("Długość fali λ0 [µm]", 0.2, 2.0, 1.0, 0.001, format="%.3f")
            R_value = st.slider("Promień krzywizny R [µm]", 1.0, 1000.0, 100.0, 1.0)

            M_total = calculate_system(st.session_state.elements, R_value, flip_R)
            w0_in = st.slider("Waist wejściowy [µm]", 1.0, 20.0, 3.0)
            zmax = st.slider("Zakres propagacji [µm]", 10.0, 10000.0, 1000.0)

            n_start = get_starting_medium(st.session_state.elements)
            n_out = get_output_medium(st.session_state.elements)

            st.caption(f"Ośrodek początkowy: n = {n_start}")
            st.caption(f"Ośrodek końcowy: n = {n_out}")

            # ===== ABCD =====
            q_in = q_waist(w0_in, n_start, lambda0)
            q_out = apply_abcd(M_total, q_in)

            z_focus = find_focus_position(q_out)

            zR = np.imag(q_out)
            w_waist = np.sqrt(lambda0 * zR / (np.pi * n_out))

            z = np.linspace(0, zmax, 800)
            qz = q_out + z
            wz = np.array([w_from_q(qv, n_out, lambda0) for qv in qz])

            # ===== mapa wiązki =====
            fig, ax = plt.subplots(figsize=(7,3))
            fill_gradient(ax, z, wz, cmap='plasma', r_scale=3)
            ax.plot(z, wz, linewidth=1)
            ax.set_xlabel("z [µm]")
            ax.set_ylabel("w(z) [µm]")
            ax.grid(True, alpha=0.3)

            if np.isfinite(z_focus) and 0 <= z_focus <= zmax:
                ax.axvline(z_focus, linestyle="--", linewidth=1)

            st.pyplot(fig, width= 'stretch')

            # ===== metryki =====
            c1, c2, c3 = st.columns(3)
            c2.metric("Pozycja ogniska", f"{z_focus:.2f} µm")
            c3.metric("Minimalna plamka", f"{w_waist:.3f} µm")


    # ======================================================
    # PRAWA KOLUMNA — ANALIZA R
    # ======================================================

    with right:

        with st.container(border=True):

            st.subheader("Zależności od R")

            R_sweep = np.linspace(1.0, 1000.0, 200)
            z_list = []
            w_list = []

            for R_test in R_sweep:
                M_test = calculate_system(st.session_state.elements, R_test, flip_R)
                q_test = apply_abcd(M_test, q_in)

                zf = find_focus_position(q_test)
                z_list.append(zf)

                zR = np.imag(q_test)
                w_list.append(np.sqrt(lambda0 * zR / (np.pi * n_out)))

            idx = np.argmin(np.abs(R_sweep - R_value))

            # ===== wykresy obok siebie =====
            
            fig2, ax2 = plt.subplots(figsize=(3.5,2.6))
            ax2.plot(R_sweep, z_list)
            ax2.plot(R_sweep[idx], z_list[idx], "ro", markersize=6)
            ax2.set_xlabel("R [µm]")
            ax2.set_ylabel("Pozycja ogniska [µm]")
            ax2.grid(True, alpha=0.3)

            auto_ylim_full_curve(ax2, z_list)

            st.pyplot(fig2, width='stretch')

            # =========================
            # w(R)
            # =========================
            
            fig3, ax3 = plt.subplots(figsize=(3.5,2.6))
            ax3.plot(R_sweep, w_list)
            ax3.plot(R_sweep[idx], w_list[idx], "ro", markersize=6)
            ax3.set_xlabel("R [µm]")
            ax3.set_ylabel("Waist [µm]")
            ax3.grid(True, alpha=0.3)

            auto_ylim_full_curve(ax3, w_list)

            st.pyplot(fig3, width='stretch')


else:
    st.info("Dodaj elementy układu w panelu po lewej.")
