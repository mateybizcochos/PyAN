import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp


# =====================================================================
# CONFIGURACION DE LA PAGINA
# =====================================================================

st.set_page_config(
    page_title="Simulador Aerodinamico - Capa Limite",
    layout="wide"
)


# =====================================================================
# 1. Planteo las ecuaciones para la calculadora
# =====================================================================

def resolver_blasius(eta_max=8.0, N_ref=1000):

    def blasius_ode(eta, y):
        return np.vstack((
            y[1],
            y[2],
            -0.5 * y[0] * y[2]
        ))

    def blasius_bc(ya, yb):
        return np.array([
            ya[0],
            ya[1],
            yb[1] - 1.0
        ])

    eta_ref_array = np.linspace(0, eta_max, N_ref)

    y_init = np.zeros((3, eta_ref_array.size))
    y_init[1] = eta_ref_array / eta_max

    sol = solve_bvp(
        blasius_ode,
        blasius_bc,
        eta_ref_array,
        y_init,
        tol=1e-8,
        max_nodes=10000
    )

    if not sol.success:
        raise RuntimeError(
            "El solver BVP no logro converger a una solucion de referencia."
        )

    return sol


def funcion_corriente(sol_ref, eta_val):
    return sol_ref.sol(eta_val)[0]


def perfil_velocidad(sol_ref, eta_val):
    return sol_ref.sol(eta_val)[1]


def gradiente_referencia(sol_ref, eta_val):
    return sol_ref.sol(eta_val)[2]


def diferencia_adelante(sol_ref, eta_vals, h):
    return (
        perfil_velocidad(sol_ref, eta_vals + h)
        - perfil_velocidad(sol_ref, eta_vals)
    ) / h


def diferencia_atras(sol_ref, eta_vals, h):
    return (
        perfil_velocidad(sol_ref, eta_vals)
        - perfil_velocidad(sol_ref, eta_vals - h)
    ) / h


def diferencia_centrada(sol_ref, eta_vals, h):
    return (
        perfil_velocidad(sol_ref, eta_vals + h)
        - perfil_velocidad(sol_ref, eta_vals - h)
    ) / (2 * h)


def segunda_derivada_centrada(sol_ref, eta_vals, h):
    return (
        funcion_corriente(sol_ref, eta_vals + h)
        - 2 * funcion_corriente(sol_ref, eta_vals)
        + funcion_corriente(sol_ref, eta_vals - h)
    ) / (h ** 2)


def extrapolacion_richardson(sol_ref, eta_vals, h):

    gradiente_h = diferencia_centrada(sol_ref, eta_vals, h)
    gradiente_h_medio = diferencia_centrada(sol_ref, eta_vals, h / 2)

    return (4 * gradiente_h_medio - gradiente_h) / 3


def calcular_p_vector(errores, h_vals):

    p_vals = []

    for i in range(len(h_vals) - 1):

        if errores[i + 1] > 1e-14 and errores[i] > 1e-14:
            p = np.log(errores[i] / errores[i + 1]) / np.log(
                h_vals[i] / h_vals[i + 1]
            )
            p_vals.append(round(p, 2))
        else:
            p_vals.append("Lim.")

    return p_vals


# =====================================================================
# 2. FUNCION EJECUTAR 
# Devuelve un diccionario con todo lo necesario para mostrar resultados
# =====================================================================

def ejecutar_calculo(U_inf, nu, x_pos, eta_max, N_nodos):

    y_max = eta_max * np.sqrt((nu * x_pos) / U_inf)

    sol_ref = resolver_blasius(eta_max=eta_max)

    eta_malla = np.linspace(0, eta_max, N_nodos)
    h = eta_malla[1] - eta_malla[0]
    eta_int = eta_malla[1:-1]

    gradiente_ref = gradiente_referencia(sol_ref, eta_int)

    gradiente_adelante = diferencia_adelante(sol_ref, eta_int, h)
    gradiente_atras = diferencia_atras(sol_ref, eta_int, h)
    gradiente_centrada = diferencia_centrada(sol_ref, eta_int, h)
    gradiente_segunda = segunda_derivada_centrada(sol_ref, eta_int, h)
    gradiente_richardson = extrapolacion_richardson(sol_ref, eta_int, h)

    err_abs = {
        "Adelante": np.abs(gradiente_ref - gradiente_adelante),
        "Atras": np.abs(gradiente_ref - gradiente_atras),
        "Centrada": np.abs(gradiente_ref - gradiente_centrada),
        "2da Deriv.": np.abs(gradiente_ref - gradiente_segunda),
        "Richardson": np.abs(gradiente_ref - gradiente_richardson),
    }

    umbral = 1e-6
    mask = np.abs(gradiente_ref) > umbral

    err_rel = {}
    for nombre, err in err_abs.items():
        rel = np.full_like(gradiente_ref, np.nan)
        rel[mask] = (err[mask] / np.abs(gradiente_ref[mask])) * 100
        err_rel[nombre] = rel

    normas = {
        nombre: {
            "L_inf": np.max(err),
            "L2": np.sqrt(np.mean(err ** 2)),
        }
        for nombre, err in err_abs.items()
    }

    # ---------------- Analisis de convergencia ----------------

    N_valores = np.array([20, 40, 80, 160, 320])
    h_valores = eta_max / (N_valores - 1)

    errores_L2 = {nombre: [] for nombre in err_abs}

    for N_step, h_step in zip(N_valores, h_valores):

        eta_step = np.linspace(0, eta_max, N_step)
        eta_int_step = eta_step[1:-1]

        ref_step = gradiente_referencia(sol_ref, eta_int_step)

        pasos = {
            "Adelante": diferencia_adelante(sol_ref, eta_int_step, h_step),
            "Atras": diferencia_atras(sol_ref, eta_int_step, h_step),
            "Centrada": diferencia_centrada(sol_ref, eta_int_step, h_step),
            "2da Deriv.": segunda_derivada_centrada(sol_ref, eta_int_step, h_step),
            "Richardson": extrapolacion_richardson(sol_ref, eta_int_step, h_step),
        }

        for nombre, aprox in pasos.items():
            error_step = np.abs(ref_step - aprox)
            errores_L2[nombre].append(np.sqrt(np.mean(error_step ** 2)))

    p_vectores = {
        nombre: calcular_p_vector(errores_L2[nombre], h_valores)
        for nombre in errores_L2
    }

    return {
        "sol_ref": sol_ref,
        "y_max": y_max,
        "h": h,
        "eta_int": eta_int,
        "gradiente_ref": gradiente_ref,
        "err_abs": err_abs,
        "err_rel": err_rel,
        "normas": normas,
        "N_valores": N_valores,
        "h_valores": h_valores,
        "errores_L2": errores_L2,
        "p_vectores": p_vectores,
        "eta_max": eta_max,
    }


# =====================================================================
# 3. GRAFICOS
# =====================================================================

def armar_figura(resultados):

    sol_ref = resultados["sol_ref"]
    eta_max = resultados["eta_max"]
    eta_int = resultados["eta_int"]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    # ---------- Grafico 1: perfil de velocidades ----------
    eta_plot = np.linspace(0, eta_max, 200)
    vel = perfil_velocidad(sol_ref, eta_plot)

    axs[0].plot(vel, eta_plot, "b-", linewidth=2.5, label="Perfil de Blasius")
    axs[0].set_title("Perfil de Velocidades")
    axs[0].set_xlabel(r"Velocidad Adimensional ($u/U_\infty$)")
    axs[0].set_ylabel(r"Altura Adimensional ($\eta$)")
    axs[0].grid(True, linestyle="--", alpha=0.7)
    axs[0].legend()

    # ---------- Grafico 2: error local ----------
    for nombre, err in resultados["err_abs"].items():
        axs[1].semilogy(eta_int, err, label=nombre, alpha=0.8)

    axs[1].set_title("Error Absoluto Local")
    axs[1].set_xlabel(r"Altura Adimensional ($\eta$)")
    axs[1].set_ylabel("Error Absoluto (Log)")
    axs[1].grid(True, linestyle="--", alpha=0.7)
    axs[1].legend(fontsize=8)

    # ---------- Grafico 3: convergencia global ----------
    h_val = resultados["h_valores"]
    marcadores = {"Adelante": "o-", "Atras": "v-", "Centrada": "s-",
                  "2da Deriv.": "d-", "Richardson": "^-"}

    for nombre, err_L2 in resultados["errores_L2"].items():
        axs[2].loglog(h_val, err_L2, marcadores[nombre], linewidth=2, label=nombre)

    idx_ref = len(h_val) // 2
    C1 = resultados["errores_L2"]["Adelante"][idx_ref] / h_val[idx_ref]
    C2 = resultados["errores_L2"]["Centrada"][idx_ref] / (h_val[idx_ref] ** 2)
    C4 = resultados["errores_L2"]["Richardson"][idx_ref] / (h_val[idx_ref] ** 4)

    axs[2].loglog(h_val, C1 * h_val, "k--", alpha=0.5, label="Ref. O(h)")
    axs[2].loglog(h_val, C2 * (h_val ** 2), "k:", alpha=0.5, label="Ref. O(h^2)")
    axs[2].loglog(h_val, C4 * (h_val ** 4), "k-.", alpha=0.5, label="Ref. O(h^4)")

    axs[2].set_title(r"Convergencia Global ($L_2$)")
    axs[2].set_xlabel(r"Paso adimensional ($h=\Delta\eta$)")
    axs[2].set_ylabel(r"Error Global $L_2$")
    axs[2].grid(True, which="both", ls="--", alpha=0.5)
    axs[2].legend(fontsize=8)

    fig.tight_layout()

    return fig


# =====================================================================
# 4. INTERFAZ STREAMLIT
# =====================================================================

st.title("Simulador Aerodinamico de Capa Limite - Diferencias Finitas")
st.caption("Ecuacion de Blasius | Problema de valores de contorno con solve_bvp | Comparacion de esquemas numericos")

with st.sidebar:

    st.header("Parametros Aerodinamicos")

    U_inf = st.number_input("Velocidad U_inf (m/s)", value=8.0, min_value=0.0001)
    nu = st.number_input(
        "Viscosidad cinematica (m^2/s)", value=1.5e-5, min_value=0.0000001,
        format="%.7f",
        help="Aire ~ 1.5e-5 m2/s. Agua ~ 1.0e-6 m2/s."
    )
    x_pos = st.number_input("Posicion x en la placa (m)", value=1.0, min_value=0.0001)

    st.header("Discretizacion Numerica")

    eta_max = st.number_input("Infinito computacional (eta max)", value=8.0, min_value=1.0)
    N_nodos = st.number_input("Cantidad de Nodos (N)", value=50, min_value=10, step=1)

    calcular = st.button("Calcular Simulacion", type="primary", use_container_width=True)

    if eta_max < 8.0:
        st.info("Se recomienda eta_max >= 8 para representar bien la corriente libre.")


# ---------------- Ejecucion del calculo ----------------

if calcular:
    try:
        with st.spinner("Resolviendo la ecuacion de Blasius..."):
            st.session_state["resultados"] = ejecutar_calculo(
                U_inf, nu, x_pos, eta_max, int(N_nodos)
            )
    except RuntimeError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Ocurrio un error durante el calculo:\n\n{e}")


# ---------------- Mostrar resultados si ya se calculo ----------------

if "resultados" in st.session_state:

    r = st.session_state["resultados"]

    col1, col2 = st.columns(2)
    col1.metric("Altura fisica de capa limite (y_max)", f"{r['y_max']:.5f} m")
    col2.metric("Paso adimensional (h = delta_eta)", f"{r['h']:.4f}")

    st.subheader("Graficos")
    fig = armar_figura(r)
    st.pyplot(fig)

    buffer_png = None
    import io
    buffer_png = io.BytesIO()
    fig.savefig(buffer_png, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        "Descargar graficos (.png)",
        data=buffer_png.getvalue(),
        file_name="graficos_blasius.png",
        mime="image/png"
    )

    st.subheader("Resumen de errores por metodo")

    tabla_resumen = pd.DataFrame([
        {
            "Metodo": nombre,
            "Orden teorico": {
                "Adelante": "O(h)", "Atras": "O(h)", "Centrada": "O(h^2)",
                "2da Deriv.": "O(h^2)", "Richardson": "O(h^4)"
            }[nombre],
            "L_inf": f"{v['L_inf']:.3e}",
            "L2": f"{v['L2']:.3e}",
        }
        for nombre, v in r["normas"].items()
    ])
    st.dataframe(tabla_resumen, use_container_width=True, hide_index=True)

    with st.expander("Tabla de convergencia global (L2) y orden observado (p)"):

        tabla_conv = pd.DataFrame({"N": r["N_valores"], "h": r["h_valores"]})
        for nombre in r["errores_L2"]:
            tabla_conv[f"L2 {nombre}"] = r["errores_L2"][nombre]
        st.dataframe(tabla_conv, use_container_width=True, hide_index=True)

        st.caption("Orden de convergencia observado: p = log(E_h / E_h/2) / log(h / (h/2))")
        tabla_p = pd.DataFrame(r["p_vectores"])
        tabla_p.insert(0, "Transicion", [
            f"N={r['N_valores'][i]} -> N={r['N_valores'][i+1]}"
            for i in range(len(r['N_valores']) - 1)
        ])
        st.dataframe(tabla_p, use_container_width=True, hide_index=True)

    with st.expander("Tabla detallada por nodo (error absoluto y relativo)"):

        tabla_nodo = pd.DataFrame({"eta": r["eta_int"], "f'' referencia": r["gradiente_ref"]})
        for nombre in r["err_abs"]:
            tabla_nodo[f"Err abs {nombre}"] = r["err_abs"][nombre]
            tabla_nodo[f"Err rel {nombre} (%)"] = r["err_rel"][nombre]
        st.dataframe(tabla_nodo, use_container_width=True, hide_index=True)

    # ---------------- Exportar informe de texto ----------------

    reporte = []
    reporte.append("REPORTE - SIMULADOR DE CAPA LIMITE (BLASIUS)")
    reporte.append(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    reporte.append(f"Parametros: U={U_inf} m/s, nu={nu} m2/s, x={x_pos} m")
    reporte.append(f"Eta max: {eta_max} | N nodos: {N_nodos} | h={r['h']:.6f}")
    reporte.append(f"Altura fisica capa limite (y_max): {r['y_max']:.5f} m\n")
    reporte.append("RESUMEN DE ERRORES POR METODO")
    reporte.append(tabla_resumen.to_string(index=False))
    reporte.append("")
    reporte.append("CONVERGENCIA GLOBAL (L2)")
    reporte.append(tabla_conv.to_string(index=False))
    reporte.append("")
    reporte.append("ORDEN DE CONVERGENCIA OBSERVADO (p)")
    reporte.append(tabla_p.to_string(index=False))

    st.download_button(
        "Descargar informe (.txt)",
        data="\n".join(reporte),
        file_name="informe_blasius.txt",
        mime="text/plain"
    )

else:
    st.info("Completa los parametros en el panel izquierdo y presiona 'Calcular Simulacion'.")
