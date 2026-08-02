import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime


# =====================================================================
# 1. MOTOR MATEMÁTICO (NÚCLEO)
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
            "El solver BVP no logró converger a una solución de referencia."
        )

    return sol


# ---------------------------------------------------------------------
# Solución de Blasius
# f(eta)   -> función de corriente
# f'(eta)  -> velocidad adimensional
# f''(eta) -> gradiente de velocidad
# ---------------------------------------------------------------------

def funcion_corriente(sol_ref, eta_val):
    return sol_ref.sol(eta_val)[0]


def perfil_velocidad(sol_ref, eta_val):
    return sol_ref.sol(eta_val)[1]


def gradiente_referencia(sol_ref, eta_val):
    return sol_ref.sol(eta_val)[2]


# ---------------------------------------------------------------------
# Diferencias finitas para f'(eta)
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Diferencia finita de segunda derivada
#
# f''(eta) ≈ [f(eta+h) - 2f(eta) + f(eta-h)] / h²
# ---------------------------------------------------------------------

def segunda_derivada_centrada(sol_ref, eta_vals, h):

    return (
        funcion_corriente(sol_ref, eta_vals + h)
        - 2 * funcion_corriente(sol_ref, eta_vals)
        + funcion_corriente(sol_ref, eta_vals - h)
    ) / (h ** 2)


# ---------------------------------------------------------------------
# Extrapolación de Richardson
#
# Se aplica sobre la derivada centrada:
#
# D_R = [4 D(h/2) - D(h)] / 3
#
# Como D_centrada tiene error O(h²),
# Richardson elimina el término O(h²) y queda O(h⁴).
# ---------------------------------------------------------------------

def extrapolacion_richardson(sol_ref, eta_vals, h):

    gradiente_h = diferencia_centrada(
        sol_ref,
        eta_vals,
        h
    )

    gradiente_h_medio = diferencia_centrada(
        sol_ref,
        eta_vals,
        h / 2
    )

    return (
        4 * gradiente_h_medio - gradiente_h
    ) / 3


# ---------------------------------------------------------------------
# Cálculo del orden de convergencia observado
# ---------------------------------------------------------------------

def calcular_p_vector(errores, h_vals):

    p_vals = []

    for i in range(len(h_vals) - 1):

        if errores[i + 1] > 1e-14 and errores[i] > 1e-14:

            p = np.log(
                errores[i] / errores[i + 1]
            ) / np.log(
                h_vals[i] / h_vals[i + 1]
            )

            p_vals.append(f"{p:.2f}")

        else:
            p_vals.append("Lím.")

    return p_vals


# =====================================================================
# 2. INTERFAZ GRÁFICA (GUI)
# =====================================================================

class AplicacionCapaLimite:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Simulador Aerodinámico de Capa Límite - Diferencias Finitas"
        )

        self.root.geometry("1350x850")

        self.sol_ref = None
        self.fig = None
        self.canvas = None
        self.reporte_texto = ""

        self.crear_interfaz()


    # -----------------------------------------------------------------
    # INTERFAZ
    # -----------------------------------------------------------------

    def crear_interfaz(self):

        panel_izq = ttk.Frame(
            self.root,
            padding="10",
            width=350
        )

        panel_izq.pack(
            side=tk.LEFT,
            fill=tk.Y
        )

        ttk.Label(
            panel_izq,
            text="Parámetros Aerodinámicos",
            font=("Arial", 11, "bold")
        ).pack(pady=5)


        ttk.Label(
            panel_izq,
            text="Velocidad U_inf (m/s):"
        ).pack(anchor=tk.W)

        self.entrada_U = ttk.Entry(panel_izq)
        self.entrada_U.insert(0, "10.0")
        self.entrada_U.pack(fill=tk.X, pady=2)


        ttk.Label(
            panel_izq,
            text="Viscosidad cinemática (m²/s):"
        ).pack(anchor=tk.W)

        self.entrada_nu = ttk.Entry(panel_izq)
        self.entrada_nu.insert(0, "1.5e-5")
        self.entrada_nu.pack(fill=tk.X, pady=2)


        ttk.Label(
            panel_izq,
            text="Posición x en la placa (m):"
        ).pack(anchor=tk.W)

        self.entrada_x = ttk.Entry(panel_izq)
        self.entrada_x.insert(0, "1.0")
        self.entrada_x.pack(fill=tk.X, pady=2)


        ttk.Separator(
            panel_izq,
            orient=tk.HORIZONTAL
        ).pack(fill=tk.X, pady=10)


        ttk.Label(
            panel_izq,
            text="Discretización Numérica",
            font=("Arial", 11, "bold")
        ).pack(pady=5)


        ttk.Label(
            panel_izq,
            text="Infinito computacional (eta max):"
        ).pack(anchor=tk.W)

        self.entrada_eta = ttk.Entry(panel_izq)
        self.entrada_eta.insert(0, "8.0")
        self.entrada_eta.pack(fill=tk.X, pady=2)


        ttk.Label(
            panel_izq,
            text="Cantidad de Nodos (N):"
        ).pack(anchor=tk.W)

        self.entrada_nodos = ttk.Entry(panel_izq)
        self.entrada_nodos.insert(0, "50")
        self.entrada_nodos.pack(fill=tk.X, pady=2)


        btn_calcular = ttk.Button(
            panel_izq,
            text="▶ Calcular Simulación",
            command=self.ejecutar_calculo
        )

        btn_calcular.pack(
            fill=tk.X,
            pady=15
        )


        ttk.Label(
            panel_izq,
            text="Resultados Físicos y Numéricos",
            font=("Arial", 11, "bold")
        ).pack(pady=5)


        self.texto_resultados = tk.Text(
            panel_izq,
            height=20,
            width=40,
            state=tk.DISABLED
        )

        self.texto_resultados.pack(
            fill=tk.X,
            pady=5
        )


        ttk.Label(
            panel_izq,
            text="Exportación",
            font=("Arial", 11, "bold")
        ).pack(pady=5)


        btn_exportar_grafico = ttk.Button(
            panel_izq,
            text="💾 Guardar Gráficos (.png)",
            command=self.exportar_grafico
        )

        btn_exportar_grafico.pack(
            fill=tk.X,
            pady=2
        )


        btn_exportar_reporte = ttk.Button(
            panel_izq,
            text="📄 Exportar Tablas (.txt)",
            command=self.exportar_reporte
        )

        btn_exportar_reporte.pack(
            fill=tk.X,
            pady=2
        )


        self.panel_grafico = ttk.Frame(
            self.root,
            padding="10"
        )

        self.panel_grafico.pack(
            side=tk.RIGHT,
            expand=True,
            fill=tk.BOTH
        )


    # =================================================================
    # CÁLCULO PRINCIPAL
    # =================================================================

    def ejecutar_calculo(self):

        try:

            # ---------------------------------------------------------
            # Lectura de parámetros
            # ---------------------------------------------------------

            U_inf = float(self.entrada_U.get())
            nu = float(self.entrada_nu.get())
            x_pos = float(self.entrada_x.get())
            eta_max = float(self.entrada_eta.get())
            N_nodos = int(self.entrada_nodos.get())


            # ---------------------------------------------------------
            # Validación
            # ---------------------------------------------------------

            if (
                U_inf <= 0
                or nu <= 0
                or x_pos <= 0
                or eta_max <= 0
            ):
                raise ValueError(
                    "Todos los parámetros deben ser mayores a cero."
                )


            if N_nodos < 10:

                raise ValueError(
                    "La cantidad de nodos N debe ser al menos 10."
                )


            # ---------------------------------------------------------
            # Conversión de eta a altura física
            # ---------------------------------------------------------

            y_max = eta_max * np.sqrt(
                (nu * x_pos) / U_inf
            )


            if eta_max < 8.0:

                messagebox.showinfo(
                    "Aviso Teórico",
                    "El valor de infinito (eta max) es menor a 8. "
                    "Para representar adecuadamente la corriente libre "
                    "se recomienda utilizar un valor aproximado a 8."
                )


            # ---------------------------------------------------------
            # Solución de referencia de Blasius
            # ---------------------------------------------------------

            try:

                self.sol_ref = resolver_blasius(
                    eta_max=eta_max
                )

            except RuntimeError as e:

                messagebox.showerror(
                    "Error de Convergencia",
                    str(e)
                )

                return


            # ---------------------------------------------------------
            # Malla principal
            # ---------------------------------------------------------

            eta_malla = np.linspace(
                0,
                eta_max,
                N_nodos
            )

            h = eta_malla[1] - eta_malla[0]

            # Se utilizan solamente nodos internos
            # para poder aplicar diferencias centradas
            eta_int = eta_malla[1:-1]


            # ---------------------------------------------------------
            # REFERENCIA
            #
            # En la ecuación de Blasius:
            #
            # f''' + 1/2 f f'' = 0
            #
            # solve_bvp entrega:
            # y[0] = f
            # y[1] = f'
            # y[2] = f''
            #
            # Por lo tanto:
            # gradiente_ref = f''
            # ---------------------------------------------------------

            gradiente_ref = gradiente_referencia(
                self.sol_ref,
                eta_int
            )


            # ---------------------------------------------------------
            # DIFERENCIAS FINITAS
            # ---------------------------------------------------------

            gradiente_adelante = diferencia_adelante(
                self.sol_ref,
                eta_int,
                h
            )

            gradiente_atras = diferencia_atras(
                self.sol_ref,
                eta_int,
                h
            )

            gradiente_centrada = diferencia_centrada(
                self.sol_ref,
                eta_int,
                h
            )


            # ---------------------------------------------------------
            # DERIVADA SEGUNDA
            #
            # f'' mediante diferencias finitas centradas
            # ---------------------------------------------------------

            gradiente_segunda_derivada = segunda_derivada_centrada(
                self.sol_ref,
                eta_int,
                h
            )


            # ---------------------------------------------------------
            # RICHARDSON
            # ---------------------------------------------------------

            gradiente_richardson = extrapolacion_richardson(
                self.sol_ref,
                eta_int,
                h
            )


            # =========================================================
            # ERRORES ABSOLUTOS
            # =========================================================

            err_abs_adelante = np.abs(
                gradiente_ref - gradiente_adelante
            )

            err_abs_atras = np.abs(
                gradiente_ref - gradiente_atras
            )

            err_abs_centrada = np.abs(
                gradiente_ref - gradiente_centrada
            )

            err_abs_segunda = np.abs(
                gradiente_ref - gradiente_segunda_derivada
            )

            err_abs_richardson = np.abs(
                gradiente_ref - gradiente_richardson
            )


            # =========================================================
            # ERRORES RELATIVOS
            # =========================================================

            umbral = 1e-6

            mask = (
                np.abs(gradiente_ref) > umbral
            )


            err_rel_adelante = np.full_like(
                gradiente_ref,
                np.nan
            )

            err_rel_atras = np.full_like(
                gradiente_ref,
                np.nan
            )

            err_rel_centrada = np.full_like(
                gradiente_ref,
                np.nan
            )

            err_rel_segunda = np.full_like(
                gradiente_ref,
                np.nan
            )

            err_rel_richardson = np.full_like(
                gradiente_ref,
                np.nan
            )


            err_rel_adelante[mask] = (
                err_abs_adelante[mask]
                / np.abs(gradiente_ref[mask])
            ) * 100


            err_rel_atras[mask] = (
                err_abs_atras[mask]
                / np.abs(gradiente_ref[mask])
            ) * 100


            err_rel_centrada[mask] = (
                err_abs_centrada[mask]
                / np.abs(gradiente_ref[mask])
            ) * 100


            err_rel_segunda[mask] = (
                err_abs_segunda[mask]
                / np.abs(gradiente_ref[mask])
            ) * 100


            err_rel_richardson[mask] = (
                err_abs_richardson[mask]
                / np.abs(gradiente_ref[mask])
            ) * 100


            # =========================================================
            # NORMAS GLOBALES
            # =========================================================

            norma_L_inf_adelante = np.max(
                err_abs_adelante
            )

            norma_L_2_adelante = np.sqrt(
                np.mean(err_abs_adelante ** 2)
            )


            norma_L_inf_atras = np.max(
                err_abs_atras
            )

            norma_L_2_atras = np.sqrt(
                np.mean(err_abs_atras ** 2)
            )


            norma_L_inf_centrada = np.max(
                err_abs_centrada
            )

            norma_L_2_centrada = np.sqrt(
                np.mean(err_abs_centrada ** 2)
            )


            norma_L_inf_segunda = np.max(
                err_abs_segunda
            )

            norma_L_2_segunda = np.sqrt(
                np.mean(err_abs_segunda ** 2)
            )


            norma_L_inf_richardson = np.max(
                err_abs_richardson
            )

            norma_L_2_richardson = np.sqrt(
                np.mean(err_abs_richardson ** 2)
            )


            # =========================================================
            # ANÁLISIS DE CONVERGENCIA
            # =========================================================

            N_valores = np.array([
                20,
                40,
                80,
                160,
                320
            ])

            h_valores = eta_max / (
                N_valores - 1
            )


            errores_L2_adelante = []
            errores_L2_atras = []
            errores_L2_centrada = []
            errores_L2_segunda = []
            errores_L2_richardson = []


            for N_step, h_step in zip(
                N_valores,
                h_valores
            ):

                eta_step = np.linspace(
                    0,
                    eta_max,
                    N_step
                )

                eta_int_step = eta_step[1:-1]


                gradiente_ref_step = gradiente_referencia(
                    self.sol_ref,
                    eta_int_step
                )


                error_adelante = np.abs(
                    gradiente_ref_step
                    - diferencia_adelante(
                        self.sol_ref,
                        eta_int_step,
                        h_step
                    )
                )


                error_atras = np.abs(
                    gradiente_ref_step
                    - diferencia_atras(
                        self.sol_ref,
                        eta_int_step,
                        h_step
                    )
                )


                error_centrada = np.abs(
                    gradiente_ref_step
                    - diferencia_centrada(
                        self.sol_ref,
                        eta_int_step,
                        h_step
                    )
                )


                error_segunda = np.abs(
                    gradiente_ref_step
                    - segunda_derivada_centrada(
                        self.sol_ref,
                        eta_int_step,
                        h_step
                    )
                )


                error_richardson = np.abs(
                    gradiente_ref_step
                    - extrapolacion_richardson(
                        self.sol_ref,
                        eta_int_step,
                        h_step
                    )
                )


                errores_L2_adelante.append(
                    np.sqrt(
                        np.mean(error_adelante ** 2)
                    )
                )


                errores_L2_atras.append(
                    np.sqrt(
                        np.mean(error_atras ** 2)
                    )
                )


                errores_L2_centrada.append(
                    np.sqrt(
                        np.mean(error_centrada ** 2)
                    )
                )


                errores_L2_segunda.append(
                    np.sqrt(
                        np.mean(error_segunda ** 2)
                    )
                )


                errores_L2_richardson.append(
                    np.sqrt(
                        np.mean(error_richardson ** 2)
                    )
                )


            # =========================================================
            # ORDEN DE CONVERGENCIA OBSERVADO
            # =========================================================

            p_adelante_vec = calcular_p_vector(
                errores_L2_adelante,
                h_valores
            )

            p_atras_vec = calcular_p_vector(
                errores_L2_atras,
                h_valores
            )

            p_centrada_vec = calcular_p_vector(
                errores_L2_centrada,
                h_valores
            )

            p_segunda_vec = calcular_p_vector(
                errores_L2_segunda,
                h_valores
            )

            # Richardson también se analiza experimentalmente.
            # Teóricamente debe tender a p ≈ 4.

            p_richardson_vec = calcular_p_vector(
                errores_L2_richardson,
                h_valores
            )


            # =========================================================
            # GRÁFICOS
            # =========================================================

            self.actualizar_graficos(
                eta_max,
                self.sol_ref,
                eta_int,
                err_abs_adelante,
                err_abs_atras,
                err_abs_centrada,
                err_abs_segunda,
                err_abs_richardson,
                h_valores,
                errores_L2_adelante,
                errores_L2_atras,
                errores_L2_centrada,
                errores_L2_segunda,
                errores_L2_richardson
            )


            # =========================================================
            # RESULTADOS EN GUI
            # =========================================================

            res_texto = (
                f"Altura física de capa límite (y): "
                f"{y_max:.5f} m\n"
            )

            res_texto += (
                f"Paso adimensional (h = Δη): "
                f"{h:.4f}\n"
            )

            res_texto += "-" * 35 + "\n"


            res_texto += (
                "DIFERENCIA HACIA ADELANTE O(h)\n"
            )

            res_texto += (
                f"L_inf: {norma_L_inf_adelante:.2e}\n"
                f"L_2: {norma_L_2_adelante:.2e}\n"
                f"p: [{', '.join(p_adelante_vec)}]\n\n"
            )


            res_texto += (
                "DIFERENCIA HACIA ATRÁS O(h)\n"
            )

            res_texto += (
                f"L_inf: {norma_L_inf_atras:.2e}\n"
                f"L_2: {norma_L_2_atras:.2e}\n"
                f"p: [{', '.join(p_atras_vec)}]\n\n"
            )


            res_texto += (
                "DIFERENCIA CENTRADA O(h²)\n"
            )

            res_texto += (
                f"L_inf: {norma_L_inf_centrada:.2e}\n"
                f"L_2: {norma_L_2_centrada:.2e}\n"
                f"p: [{', '.join(p_centrada_vec)}]\n\n"
            )


            res_texto += (
                "DERIVADA SEGUNDA CENTRADA O(h²)\n"
            )

            res_texto += (
                f"L_inf: {norma_L_inf_segunda:.2e}\n"
                f"L_2: {norma_L_2_segunda:.2e}\n"
                f"p: [{', '.join(p_segunda_vec)}]\n\n"
            )


            res_texto += (
                "EXTRAPOLACIÓN DE RICHARDSON O(h⁴)\n"
            )

            res_texto += (
                f"L_inf: {norma_L_inf_richardson:.2e}\n"
                f"L_2: {norma_L_2_richardson:.2e}\n"
                f"p observado: "
                f"[{', '.join(p_richardson_vec)}]\n"
            )


            self.texto_resultados.config(
                state=tk.NORMAL
            )

            self.texto_resultados.delete(
                1.0,
                tk.END
            )

            self.texto_resultados.insert(
                tk.END,
                res_texto
            )

            self.texto_resultados.config(
                state=tk.DISABLED
            )


            # =========================================================
            # REPORTE
            # =========================================================

            self.generar_reporte(
                U_inf,
                nu,
                x_pos,
                y_max,
                eta_max,
                N_nodos,
                h,
                eta_int,
                gradiente_ref,
                err_abs_adelante,
                err_abs_atras,
                err_abs_centrada,
                err_abs_segunda,
                err_abs_richardson,
                err_rel_adelante,
                err_rel_atras,
                err_rel_centrada,
                err_rel_segunda,
                err_rel_richardson,
                N_valores,
                h_valores,
                errores_L2_adelante,
                errores_L2_atras,
                errores_L2_centrada,
                errores_L2_segunda,
                errores_L2_richardson,
                p_adelante_vec,
                p_atras_vec,
                p_centrada_vec,
                p_segunda_vec,
                p_richardson_vec
            )


        except ValueError as e:

            messagebox.showerror(
                "Error de Validación",
                str(e)
            )


        except Exception as e:

            messagebox.showerror(
                "Error",
                f"Ocurrió un error durante el cálculo:\n\n{e}"
            )


    # =================================================================
    # REPORTE
    # =================================================================

    def generar_reporte(
        self,
        U_inf,
        nu,
        x_pos,
        y_max,
        eta_max,
        N_nodos,
        h,
        eta_int,
        gradiente_ref,
        err_abs_adelante,
        err_abs_atras,
        err_abs_centrada,
        err_abs_segunda,
        err_abs_richardson,
        err_rel_adelante,
        err_rel_atras,
        err_rel_centrada,
        err_rel_segunda,
        err_rel_richardson,
        N_valores,
        h_valores,
        errores_L2_adelante,
        errores_L2_atras,
        errores_L2_centrada,
        errores_L2_segunda,
        errores_L2_richardson,
        p_adelante_vec,
        p_atras_vec,
        p_centrada_vec,
        p_segunda_vec,
        p_richardson_vec
    ):

        self.reporte_texto = (
            "REPORTE TABULADO DE SIMULACIÓN - CAPA LÍMITE\n"
        )

        self.reporte_texto += (
            f"Fecha: "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        self.reporte_texto += (
            f"Parámetros: "
            f"U={U_inf} m/s, "
            f"nu={nu} m2/s, "
            f"x={x_pos} m\n"
        )

        self.reporte_texto += (
            f"Eta max (Infinito Computacional): "
            f"{eta_max:.4f}\n"
        )

        self.reporte_texto += (
            f"Altura física calculada (y_max): "
            f"{y_max:.5f} m\n"
        )

        self.reporte_texto += (
            f"Nodos (N): {N_nodos} | "
            f"Paso adimensional (h = Δη): "
            f"{h:.6f}\n\n"
        )


        # -------------------------------------------------------------
        # TABLA 1
        # -------------------------------------------------------------

        self.reporte_texto += (
            "TABLA 1: ERROR ABSOLUTO LOCAL POR NODO\n"
        )

        self.reporte_texto += (
            f"{'Eta':<10} | "
            f"{'Gradiente de referencia':<23} | "
            f"{'Err Adelante':<14} | "
            f"{'Err Atrás':<14} | "
            f"{'Err Centrada':<14} | "
            f"{'Err 2da Deriv.':<14} | "
            f"{'Err Richardson':<14}\n"
        )

        self.reporte_texto += "-" * 125 + "\n"


        for i in range(len(eta_int)):

            self.reporte_texto += (
                f"{eta_int[i]:<10.4f} | "
                f"{gradiente_ref[i]:<23.2e} | "
                f"{err_abs_adelante[i]:<14.2e} | "
                f"{err_abs_atras[i]:<14.2e} | "
                f"{err_abs_centrada[i]:<14.2e} | "
                f"{err_abs_segunda[i]:<14.2e} | "
                f"{err_abs_richardson[i]:<14.2e}\n"
            )


        # -------------------------------------------------------------
        # TABLA 2
        # -------------------------------------------------------------

        self.reporte_texto += (
            "\nTABLA 2: ERROR RELATIVO LOCAL POR NODO (%) "
            "[Valores con |f''| < 1e-6 omitidos]\n"
        )

        self.reporte_texto += (
            f"{'Eta':<10} | "
            f"{'Gradiente de referencia':<23} | "
            f"{'Err Adelante':<14} | "
            f"{'Err Atrás':<14} | "
            f"{'Err Centrada':<14} | "
            f"{'Err 2da Deriv.':<14} | "
            f"{'Err Richardson':<14}\n"
        )

        self.reporte_texto += "-" * 125 + "\n"


        for i in range(len(eta_int)):

            ea = (
                f"{err_rel_adelante[i]:.2e}"
                if not np.isnan(err_rel_adelante[i])
                else "NaN"
            )

            eb = (
                f"{err_rel_atras[i]:.2e}"
                if not np.isnan(err_rel_atras[i])
                else "NaN"
            )

            ec = (
                f"{err_rel_centrada[i]:.2e}"
                if not np.isnan(err_rel_centrada[i])
                else "NaN"
            )

            es = (
                f"{err_rel_segunda[i]:.2e}"
                if not np.isnan(err_rel_segunda[i])
                else "NaN"
            )

            er = (
                f"{err_rel_richardson[i]:.2e}"
                if not np.isnan(err_rel_richardson[i])
                else "NaN"
            )


            self.reporte_texto += (
                f"{eta_int[i]:<10.4f} | "
                f"{gradiente_ref[i]:<23.2e} | "
                f"{ea:<14} | "
                f"{eb:<14} | "
                f"{ec:<14} | "
                f"{es:<14} | "
                f"{er:<14}\n"
            )


        # -------------------------------------------------------------
        # TABLA 3
        # -------------------------------------------------------------

        self.reporte_texto += (
            "\nTABLA 3: CONVERGENCIA GLOBAL (NORMA L2)\n"
        )

        self.reporte_texto += (
            f"{'N Nodos':<10} | "
            f"{'Paso h':<12} | "
            f"{'Err Adelante':<14} | "
            f"{'Err Atrás':<14} | "
            f"{'Err Centrada':<14} | "
            f"{'Err 2da Deriv.':<14} | "
            f"{'Err Richardson':<14}\n"
        )

        self.reporte_texto += "-" * 115 + "\n"


        for i in range(len(N_valores)):

            self.reporte_texto += (
                f"{N_valores[i]:<10} | "
                f"{h_valores[i]:<12.6f} | "
                f"{errores_L2_adelante[i]:<14.2e} | "
                f"{errores_L2_atras[i]:<14.2e} | "
                f"{errores_L2_centrada[i]:<14.2e} | "
                f"{errores_L2_segunda[i]:<14.2e} | "
                f"{errores_L2_richardson[i]:<14.2e}\n"
            )


        # -------------------------------------------------------------
        # TABLA 4
        # -------------------------------------------------------------

        self.reporte_texto += (
            "\nTABLA 4: ORDEN DE CONVERGENCIA OBSERVADO (p)\n"
        )

        self.reporte_texto += (
            "El orden p se obtiene experimentalmente mediante:\n"
            "p = log(E_h / E_h/2) / log(h / (h/2))\n\n"
        )

        self.reporte_texto += (
            f"{'Transición':<15} | "
            f"{'Adelante':<12} | "
            f"{'Atrás':<12} | "
            f"{'Centrada':<12} | "
            f"{'2da Deriv.':<12} | "
            f"{'Richardson':<12}\n"
        )

        self.reporte_texto += "-" * 90 + "\n"


        for i in range(len(p_adelante_vec)):

            trans = (
                f"N={N_valores[i]} -> "
                f"N={N_valores[i + 1]}"
            )

            self.reporte_texto += (
                f"{trans:<15} | "
                f"{p_adelante_vec[i]:<12} | "
                f"{p_atras_vec[i]:<12} | "
                f"{p_centrada_vec[i]:<12} | "
                f"{p_segunda_vec[i]:<12} | "
                f"{p_richardson_vec[i]:<12}\n"
            )


    # =================================================================
    # GRÁFICOS
    # =================================================================

    def actualizar_graficos(
        self,
        eta_max,
        sol_ref,
        eta_int,
        e_adelante,
        e_atras,
        e_centrada,
        e_segunda,
        e_richardson,
        h_val,
        L2_adelante,
        L2_atras,
        L2_centrada,
        L2_segunda,
        L2_richardson
    ):

        if self.fig:

            plt.close(self.fig)


        for widget in self.panel_grafico.winfo_children():

            widget.destroy()


        self.fig, axs = plt.subplots(
            1,
            3,
            figsize=(14, 5)
        )


        # -------------------------------------------------------------
        # GRÁFICO 1: PERFIL DE VELOCIDADES
        # -------------------------------------------------------------

        eta_plot = np.linspace(
            0,
            eta_max,
            200
        )

        velocidad_adimensional = perfil_velocidad(
            sol_ref,
            eta_plot
        )


        axs[0].plot(
            velocidad_adimensional,
            eta_plot,
            'b-',
            linewidth=2.5,
            label="Perfil de Blasius"
        )

        axs[0].set_title(
            "Perfil de Velocidades"
        )

        axs[0].set_xlabel(
            "Velocidad Adimensional ($u / U_\\infty$)"
        )

        axs[0].set_ylabel(
            "Altura Adimensional ($\\eta$)"
        )

        axs[0].grid(
            True,
            linestyle='--',
            alpha=0.7
        )

        axs[0].legend()


        # -------------------------------------------------------------
        # GRÁFICO 2: ERROR LOCAL
        # -------------------------------------------------------------

        axs[1].semilogy(
            eta_int,
            e_adelante,
            label="Adelante $O(h)$",
            alpha=0.7
        )

        axs[1].semilogy(
            eta_int,
            e_atras,
            label="Atrás $O(h)$",
            alpha=0.7
        )

        axs[1].semilogy(
            eta_int,
            e_centrada,
            label="Centrada $O(h^2)$",
            alpha=0.8
        )

        axs[1].semilogy(
            eta_int,
            e_segunda,
            label="2da derivada $O(h^2)$",
            alpha=0.8
        )

        axs[1].semilogy(
            eta_int,
            e_richardson,
            label="Richardson $O(h^4)$",
            linewidth=2
        )

        axs[1].set_title(
            "Error Absoluto Local"
        )

        axs[1].set_xlabel(
            "Altura Adimensional ($\\eta$)"
        )

        axs[1].set_ylabel(
            "Error Absoluto (Log)"
        )

        axs[1].grid(
            True,
            linestyle='--',
            alpha=0.7
        )

        axs[1].legend(
            fontsize=8
        )


        # -------------------------------------------------------------
        # GRÁFICO 3: CONVERGENCIA
        # -------------------------------------------------------------

        axs[2].loglog(
            h_val,
            L2_adelante,
            'o-',
            linewidth=2,
            label="Adelante $O(h)$"
        )

        axs[2].loglog(
            h_val,
            L2_atras,
            'v-',
            linewidth=2,
            label="Atrás $O(h)$"
        )

        axs[2].loglog(
            h_val,
            L2_centrada,
            's-',
            linewidth=2,
            label="Centrada $O(h^2)$"
        )

        axs[2].loglog(
            h_val,
            L2_segunda,
            'd-',
            linewidth=2,
            label="2da derivada $O(h^2)$"
        )

        axs[2].loglog(
            h_val,
            L2_richardson,
            '^-',
            linewidth=2,
            label="Richardson $O(h^4)$"
        )


        # -------------------------------------------------------------
        # RECTAS DE REFERENCIA
        # -------------------------------------------------------------

        idx_ref = len(h_val) // 2


        C1 = (
            L2_adelante[idx_ref]
            / h_val[idx_ref]
        )

        C2 = (
            L2_centrada[idx_ref]
            / (h_val[idx_ref] ** 2)
        )

        C4 = (
            L2_richardson[idx_ref]
            / (h_val[idx_ref] ** 4)
        )


        # Referencia O(h)

        axs[2].loglog(
            h_val,
            C1 * h_val,
            'k--',
            alpha=0.5,
            label="Ref. $O(h)$"
        )


        # Referencia O(h²)

        axs[2].loglog(
            h_val,
            C2 * (h_val ** 2),
            'k:',
            alpha=0.5,
            label="Ref. $O(h^2)$"
        )


        # Referencia O(h⁴)

        axs[2].loglog(
            h_val,
            C4 * (h_val ** 4),
            'k-.',
            alpha=0.5,
            label="Ref. $O(h^4)$"
        )


        axs[2].set_title(
            "Convergencia Global ($L_2$)"
        )

        axs[2].set_xlabel(
            "Paso adimensional ($h = \\Delta\\eta$)"
        )

        axs[2].set_ylabel(
            "Error Global $L_2$"
        )

        axs[2].grid(
            True,
            which="both",
            ls="--",
            alpha=0.5
        )

        axs[2].legend(
            fontsize=8
        )


        self.fig.tight_layout()


        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=self.panel_grafico
        )

        self.canvas.draw()

        self.canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True
        )


    # =================================================================
    # EXPORTAR GRÁFICO
    # =================================================================

    def exportar_grafico(self):

        if self.fig is None:

            messagebox.showwarning(
                "Sin datos",
                "Primero debe ejecutar un cálculo."
            )

            return


        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png")
            ]
        )


        if filepath:

            self.fig.savefig(
                filepath,
                dpi=300
            )

            messagebox.showinfo(
                "Éxito",
                f"Gráficos guardados en:\n{filepath}"
            )


    # =================================================================
    # EXPORTAR REPORTE
    # =================================================================

    def exportar_reporte(self):

        if not self.reporte_texto:

            messagebox.showwarning(
                "Sin datos",
                "Primero debe ejecutar un cálculo."
            )

            return


        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text file", "*.txt")
            ]
        )


        if filepath:

            with open(
                filepath,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    self.reporte_texto
                )


            messagebox.showinfo(
                "Éxito",
                f"Tablas exportadas en:\n{filepath}"
            )


# =====================================================================
# 3. EJECUCIÓN
# =====================================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = AplicacionCapaLimite(
        root
    )

    root.mainloop()