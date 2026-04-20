import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# ==========================================================
# COLORMAPS
# ==========================================================

INT_CMAPS = {
    "Hot ": "hot",
    "Inferno": "inferno",
    "Magma": "magma",
    "Plasma": "plasma",
    "Viridis": "viridis",
    "Turbo": "turbo",
    "Gray": "gray",
}

PHASE_CMAPS = {
    "Twilight (cyclic)": "twilight",
    "HSV (cyclic)": "hsv",
    "Turbo": "turbo",
    "Viridis": "viridis",
    "Gray": "gray",
}

# ==========================================================
# CONSTANTS
# ==========================================================

FIBER_WIDTH = 125.0
HALF_APERTURE = FIBER_WIDTH / 2.0

# ==========================================================
# SESSION STATE
# ==========================================================

if "elements" not in st.session_state:
    st.session_state.elements = []

if "R_value" not in st.session_state:
    st.session_state.R_value = 100.0

if "reference_metrics" not in st.session_state:
    st.session_state.reference_metrics = None

# ==========================================================
# ABCD MATRICES
# ==========================================================

def M_free_space(d, n):
    return np.array([[1.0, d / n],
                     [0.0, 1.0]])

def M_spherical(n1, n2, R, flip_R=False):
    if flip_R:
        R = -R
    C = (n1 - n2) / R
    return np.array([[1.0, 0.0],
                     [C,   1.0]])

def calculate_system(elements, R_value, flip_R=False):
    M_total = np.eye(2)

    for el in elements:
        if el[0] == "free":
            M = M_free_space(el[1], el[2])
        elif el[0] == "spherical":
            M = M_spherical(el[1], el[2], R_value, flip_R)

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
# FOCUS
# ==========================================================

def focus_condition(z, q_out):
    return np.real(1.0 / (q_out + z))

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
    except:
        pass
    return np.nan

# ==========================================================
# HELPERS
# ==========================================================

def get_starting_medium(elements):
    if not elements:
        return 1.0
    return elements[0][2] if elements[0][0] == "free" else elements[0][1]

def get_output_medium(elements):
    if not elements:
        return 1.0
    return elements[-1][2]

def auto_ylim_full_curve(ax, y, lower_pct=1, upper_pct=99, margin_frac=0.08):
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return

    ymin = np.percentile(y, lower_pct)
    ymax = np.percentile(y, upper_pct)
    margin = margin_frac * (ymax - ymin)
    ax.set_ylim(ymin - margin, ymax + margin)

# ==========================================================
# VISUALIZATION
# ==========================================================

def fill_gradient(ax, z, w, cmap='viridis', scale="Linear", r_scale=3):

    r_max = r_scale * np.nanmax(w)
    r = np.linspace(-r_max, r_max, 250)

    Z, R = np.meshgrid(z, r)
    W = np.tile(w, (len(r), 1))

    I = np.exp(-2 * R**2 / W**2)

    if scale == "Logarithmic":
        I = np.log10(I + 1e-12)
        vmin, vmax = -6, 0
    else:
        vmin, vmax = 0, 1

    ax.pcolormesh(Z, R, I, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("z [µm]")
    ax.set_ylabel("r [µm]")

def plot_beam_cross_section(ax, w_value, cmap, scale="Linear"):

    x = np.linspace(-2*w_value, 2*w_value, 300)
    y = np.linspace(-2*w_value, 2*w_value, 300)

    X, Y = np.meshgrid(x, y)
    I = np.exp(-2*(X**2 + Y**2)/w_value**2)

    if scale == "Logarithmic":
        I = np.log10(I + 1e-12)
        vmin, vmax = -6, 0
    else:
        vmin, vmax = 0, 1

    ax.imshow(
        I,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_aspect("equal")

# ==========================================================
# APP
# ==========================================================

st.set_page_config(layout="wide")
st.title("ABCD Matrix Method for Micro-Optical Systems")

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("Optical System")

    element_type = st.selectbox(
        "Type of element",
        ["Propagation in medium", "Spherical surface"]
    )

    if element_type == "Propagation in medium":
        d = st.number_input("Distance d [µm]", value=100.0)
        n = st.number_input("Refractive index n", value=1.0)

        if st.button("Add propagation in medium"):
            st.session_state.elements.append(("free", d, n))

    if element_type == "Spherical surface":
        n1 = st.number_input("n1", value=1.5)
        n2 = st.number_input("n2", value=1.0)

        if st.button("Add spherical surface"):
            st.session_state.elements.append(("spherical", n1, n2))

    flip_R = st.checkbox("Flip sign of R", value=False)

    if st.button("Clear system"):
        st.session_state.elements = []

    st.divider()

    int_cmap_label = st.selectbox(
        "Intensity colormap",
        list(INT_CMAPS.keys()),
        index=3
    )

    intensity_scale = st.selectbox(
        "Intensity scale",
        ["Linear", "Logarithmic"]
    )

int_cmap = INT_CMAPS[int_cmap_label]
R_value = st.session_state.R_value

# ==========================================================
# OPTICAL SYSTEM
# ==========================================================

st.subheader("Optical system")

if st.session_state.elements:

    col_left, col_right = st.columns([1.4, 1])

    # ======================================================
    # LEFT COLUMN
    # ======================================================

    with col_left:

        for i, el in enumerate(st.session_state.elements):

            if el[0] == "free":

                st.markdown(f"### {i+1}. Propagation in medium")

                c1, c2, c3 = st.columns([4,4,1])

                with c1:
                    d_new = st.number_input(
                        f"Distance d [µm] #{i+1}",
                        value=float(el[1]),
                        key=f"d_{i}"
                    )

                with c2:
                    n_new = st.number_input(
                        f"Refractive index n #{i+1}",
                        value=float(el[2]),
                        key=f"n_{i}"
                    )

                st.session_state.elements[i] = ("free", d_new, n_new)

            else:

                st.markdown(f"### {i+1}. Spherical surface")

                c1, c2, c3 = st.columns([4,4,1])

                with c1:
                    n1_new = st.number_input(
                        f"n1 #{i+1}",
                        value=float(el[1]),
                        key=f"n1_{i}"
                    )

                with c2:
                    n2_new = st.number_input(
                        f"n2 #{i+1}",
                        value=float(el[2]),
                        key=f"n2_{i}"
                    )

                with c3:
                    if st.button("X", key=f"del_{i}"):
                        st.session_state.elements.pop(i)
                        st.rerun()

                # NEW LOCATION OF R SLIDER
                st.session_state.R_value = st.slider(
                    f"Radius of curvature R [µm] #{i+1}",
                    1.0,
                    1000.0,
                    float(st.session_state.R_value),
                    key=f"R_slider_{i}"
                )

                st.session_state.elements[i] = ("spherical", n1_new, n2_new)

                st.divider()
                continue

            with c3:
                if st.button("X", key=f"del_{i}"):
                    st.session_state.elements.pop(i)
                    st.rerun()

            st.divider()

    # ======================================================
    # RIGHT COLUMN - OPTICAL LAYOUT
    # ======================================================

    with col_right:

        st.markdown("### Optical layout")

        fig_sys, ax_sys = plt.subplots(figsize=(7,3))
        ax_sys.axhline(0, color="gray", lw=1)

        x = 0
        blue = "royalblue"

        i = 0
        while i < len(st.session_state.elements):

            el = st.session_state.elements[i]

            if el[0] == "free":

                d = el[1]
                n = el[2]

                next_is_surface = (
                    i + 1 < len(st.session_state.elements)
                    and st.session_state.elements[i+1][0] == "spherical"
                )

                if next_is_surface:

                    y_limit = min(HALF_APERTURE, abs(R_value) - 1e-6)
                    y = np.linspace(-y_limit, y_limit, 400)

                    sag = abs(R_value) - np.sqrt(R_value**2 - y**2)

                    if flip_R:
                        x_curve = x + d - sag
                    else:
                        x_curve = x + d + sag

                    poly_x = [x, x_curve[0]]
                    poly_y = [HALF_APERTURE, HALF_APERTURE]

                    poly_x += list(x_curve[::-1])
                    poly_y += list(y[::-1])

                    poly_x += [x_curve[-1], x]
                    poly_y += [-HALF_APERTURE, -HALF_APERTURE]

                    ax_sys.fill(
                        poly_x,
                        poly_y,
                        facecolor=blue,
                        edgecolor="none",
                        alpha=0.25
                    )

                    ax_sys.plot([x, x_curve[0]], [HALF_APERTURE, HALF_APERTURE], color=blue, lw=1.5)
                    ax_sys.plot([x, x_curve[-1]], [-HALF_APERTURE, -HALF_APERTURE], color=blue, lw=1.5)
                    ax_sys.plot(x_curve, y, color=blue, lw=2)

                    ax_sys.text(
                        x + d/2,
                        HALF_APERTURE + 8,
                        f"n={n:.2f}",
                        ha="center"
                    )

                    x = max(x_curve)
                    i += 2
                    continue

                else:

                    rect = plt.Rectangle(
                        (x, -HALF_APERTURE),
                        d,
                        FIBER_WIDTH,
                        facecolor=blue,
                        edgecolor=blue,
                        alpha=0.25
                    )

                    ax_sys.add_patch(rect)

                    ax_sys.text(
                        x + d/2,
                        HALF_APERTURE + 8,
                        f"n={n:.2f}",
                        ha="center"
                    )

                    x += d

            elif el[0] == "spherical":

                y_limit = min(HALF_APERTURE, abs(R_value) - 1e-6)
                y = np.linspace(-y_limit, y_limit, 400)

                sag = abs(R_value) - np.sqrt(R_value**2 - y**2)

                if flip_R:
                    x_curve = x - sag
                else:
                    x_curve = x + sag

                ax_sys.plot(x_curve, y, color=blue, lw=2)
                x = max(x_curve)

            i += 1

        ax_sys.set_xlim(-10, max(100, x + 20))
        ax_sys.set_ylim(-80, 80)

        ax_sys.set_ylabel("y [µm]")
        ax_sys.set_yticks([-62.5, 0, 62.5])
        ax_sys.set_yticklabels(["-62.5", "0", "62.5"])

        ax_sys.set_xlabel("z [µm]")

        ax_sys.spines["top"].set_visible(False)
        ax_sys.spines["right"].set_visible(False)
        ax_sys.spines["left"].set_visible(True)
        ax_sys.spines["bottom"].set_visible(True)
        ax_sys.spines["left"].set_position(("axes", 0.0))

        st.pyplot(fig_sys, width="stretch")

else:
    st.info("No elements")

# ==========================================================
# CALCULATIONS + PLOTS
# ==========================================================

if st.session_state.elements:

    st.divider()
    st.subheader("Propagation of Gaussian beam")

    left, right = st.columns([1.35, 1.0])

    with left:

        with st.container(border=True):

            lambda0 = st.slider("Wavelength λ0 [µm]", 0.2, 2.0, 1.0, 0.001)

            M_total = calculate_system(
                st.session_state.elements,
                st.session_state.R_value,
                flip_R
            )

            w0_in = st.slider("Input waist [µm]", 1.0, 20.0, 3.0)
            zmax = st.slider("Propagation range [µm]", 10.0, 10000.0, 1000.0)

            n_start = get_starting_medium(st.session_state.elements)
            n_out = get_output_medium(st.session_state.elements)

            q_in = q_waist(w0_in, n_start, lambda0)
            q_out = apply_abcd(M_total, q_in)

            z_focus = find_focus_position(q_out)

            zR = np.imag(q_out)
            w_waist = np.sqrt(lambda0 * zR / (np.pi * n_out))

            z = np.linspace(0, zmax, 800)
            qz = q_out + z
            wz = np.array([w_from_q(qv, n_out, lambda0) for qv in qz])

            z_profile = st.slider(
                "Position of cross-section z [µm]",
                0.0,
                float(zmax),
                float(z_focus if np.isfinite(z_focus) else zmax/2)
            )

            w_profile = np.interp(z_profile, z, wz)

            cA, cB = st.columns([2.2,1.2])

            with cA:
                fig, ax = plt.subplots(figsize=(6,3))
                fill_gradient(ax, z, wz, cmap=int_cmap, scale=intensity_scale)
                ax.plot(z, wz, color="white")
                ax.axvline(z_profile, color="red")

                if np.isfinite(z_focus):
                    ax.axvline(z_focus, ls="--", color="cyan")

                st.pyplot(fig, width="stretch")

            with cB:
                fig2, ax2 = plt.subplots(figsize=(4,4))
                plot_beam_cross_section(ax2, w_profile, int_cmap, intensity_scale)
                st.pyplot(fig2, width="stretch")

            st.markdown("""
            <style>
            .big-ref {
                font-size: 42px;
                font-weight: 800;
                line-height: 1.1;
            }
            .big-ref-label {
                font-size: 20px;
                color: #888;
                margin-bottom: 4px;
            }
            .big-ref-delta {
                font-size: 22px;
                font-weight: 700;
                color: #00c853;
            }
            </style>
            """, unsafe_allow_html=True)
            
            c1, c2, c3, c4, c5 = st.columns([1.3, 1.5, 1.5, 1.5, 0.2])

            with c1:
                if st.button("Set as reference", use_container_width=True):
                    st.session_state.reference_metrics = {
                        "focus": float(z_focus) if np.isfinite(z_focus) else f"Brak ogniska",
                        "spot": float(w_waist) if np.isfinite(w_waist) else np.nan,
                    }

                if st.button("Clear reference", use_container_width=True):
                    st.session_state.reference_metrics = None
                    st.rerun()

            ref = st.session_state.reference_metrics

            with c2:
                delta_focus = z_focus - ref["focus"] if ref else 0
                st.markdown(f'''
                <div class="big-ref-label">Position of focus</div>
                <div class="big-ref">{z_focus:.2f} µm</div>
                <div class="big-ref-delta">{delta_focus:+.2f} µm</div>
                ''', unsafe_allow_html=True)

            with c3:
                delta_spot = w_waist - ref["spot"] if ref else 0
                st.markdown(f'''
                <div class="big-ref-label">Minimal spot size</div>
                <div class="big-ref">{w_waist:.3f} µm</div>
                <div class="big-ref-delta">{delta_spot:+.3f} µm</div>
                ''', unsafe_allow_html=True)
            with c4:
                st.markdown(f'''
                <div class="big-ref-label">Current spot size</div>
                <div class="big-ref">{w_profile:.3f} µm</div>
                <div class="big-ref-delta">z = {z_profile:.1f} µm</div>
                ''', unsafe_allow_html=True)

    with right:

        with st.container(border=True):

            st.subheader("Characteristics of microlens")

            R_sweep = np.linspace(1,1000,200)

            z_list = []
            w_list = []

            for R_test in R_sweep:

                M_test = calculate_system(
                    st.session_state.elements,
                    R_test,
                    flip_R
                )

                q_test = apply_abcd(M_test, q_in)

                zf = find_focus_position(q_test)
                z_list.append(zf)

                zR = np.imag(q_test)
                w_list.append(np.sqrt(lambda0*zR/(np.pi*n_out)))

            idx = np.argmin(np.abs(R_sweep - st.session_state.R_value))

            fig3, ax3 = plt.subplots(figsize=(3.5,2.6))
            ax3.plot(R_sweep, z_list)
            ax3.plot(R_sweep[idx], z_list[idx], "ro")
            ax3.set_xlabel("R [µm]")
            ax3.set_ylabel("Position of focus [µm]")
            auto_ylim_full_curve(ax3, z_list)
            st.pyplot(fig3, width="stretch")

            fig4, ax4 = plt.subplots(figsize=(3.5,2.6))
            ax4.plot(R_sweep, w_list)
            ax4.plot(R_sweep[idx], w_list[idx], "ro")
            ax4.set_xlabel("R [µm]")
            ax4.set_ylabel("Waist [µm]")
            auto_ylim_full_curve(ax4, w_list)
            st.pyplot(fig4, width="stretch")

else:
    st.info("Add elements to the system in the left panel.")
