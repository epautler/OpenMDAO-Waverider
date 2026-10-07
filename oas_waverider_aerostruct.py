
import numpy as np
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.utils.constants import grav_constant
import openmdao.api as om
import matplotlib.pyplot as plt

def cst_mesh(num_x=7, num_y=15, L=1.0):
    A0, A1, A2 = 0.175798, -0.251730, 0.175798
    semispan = 0.21893                                     # Star-CCM+ X(t) = 0.21893 - 0.437861 t

    y = np.linspace(-semispan, 0.0, num_y)
    x_le = 0.95 * L * np.abs(y) / semispan
    mesh = np.zeros((num_x, num_y, 3))
    for i, s in enumerate(np.linspace(0.0, 1.0, num_x)):
        x = x_le + s * (L - x_le)
        scale = np.maximum(x / L, 1e-12)
        t = np.clip((semispan - y / scale) / 0.437861, 0.0, 1.0)
        z = t * (1 - t) * (A0 * (1 - t) ** 2 + 2 * A1 * t * (1 - t) + A2 * t**2)
        mesh[i, :, 0] = x
        mesh[i, :, 1] = y
        mesh[i, :, 2] = scale * z
    return mesh

def build_problem():
    mesh = cst_mesh()
    twist_cp = np.zeros(2)

    surface = {
        "name": "wing",
        "symmetry": True,
        "S_ref_type": "wetted",
        "fem_model_type": "tube",
        "thickness_cp": np.array([0.002, 0.003]),
        "twist_cp": twist_cp,
        "mesh": mesh,

        "CL0": 0.0,
        "CD0": 0.015,
        "k_lam": 0.05,
        "t_over_c_cp": np.array([0.15]),  # thickness over chord ratio (NACA0015)
        "c_max_t": 0.303,  # chordwise location of maximum (NACA0015)
        "with_viscous": True,
        "with_wave": False,
        # Structural values are based on aluminum 7075
        "E": 70.0e9,
        "G": 30.0e9,
        "yield": 500.0e6,
        "safety_factor": 2.5,
        "mrho": 3.0e3,
        "fem_origin": 0.35,
        "wing_weight_ratio": 2.0,
        "struct_weight_relief": False,
        "distributed_fuel_weight": False,
        "exact_failure_constraint": False,
    }

    prob = om.Problem(reports=False)

    indep_var_comp = om.IndepVarComp()
    indep_var_comp.add_output("v", val=100.0, units="m/s")
    indep_var_comp.add_output("alpha", val=10.0, units="deg")
    indep_var_comp.add_output("Mach_number", val=0.29)
    indep_var_comp.add_output("re", val=6.8e6, units="1/m")
    indep_var_comp.add_output("rho", val=1.225, units="kg/m**3")
    indep_var_comp.add_output("CT", val=grav_constant * 17.0e-6, units="1/s")
    indep_var_comp.add_output("R", val=11.165e6, units="m")
    indep_var_comp.add_output("W0", val=0.4 * 3e5, units="kg")
    indep_var_comp.add_output("speed_of_sound", val=340.3, units="m/s")
    indep_var_comp.add_output("load_factor", val=1.0)
    indep_var_comp.add_output("empty_cg", val=np.zeros((3)), units="m")

    prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])
    aerostruct_group = AerostructGeometry(surface=surface)

    name = "wing"
    prob.model.add_subsystem(name, aerostruct_group)
    point_name = "AS_point_0"
    AS_point = AerostructPoint(surfaces=[surface])

    prob.model.add_subsystem(
        point_name,
        AS_point,
        promotes_inputs=[
            "v",
            "alpha",
            "Mach_number",
            "re",
            "rho",
            "CT",
            "R",
            "W0",
            "speed_of_sound",
            "empty_cg",
            "load_factor",
        ],
    )

    com_name = point_name + "." + name + "_perf"
    prob.model.connect(name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed")
    prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")
    prob.model.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")
    prob.model.connect(name + ".radius", com_name + ".radius")
    prob.model.connect(name + ".thickness", com_name + ".thickness")
    prob.model.connect(name + ".nodes", com_name + ".nodes")
    prob.model.connect(name + ".cg_location", point_name + "." + "total_perf." + name + "_cg_location")
    prob.model.connect(name + ".structural_mass", point_name + "." + "total_perf." + name + "_structural_mass")
    prob.model.connect(name + ".t_over_c", com_name + ".t_over_c")

    prob.setup()

    return prob


if __name__ == "__main__":
    solvers = {
        "NLBGS": lambda: om.NonlinearBlockGS(use_aitken=False),
        "NLBGS + Aitken": lambda: om.NonlinearBlockGS(use_aitken=True),  # OAS default
        "Newton": lambda: om.NewtonSolver(solve_subsystems=True),
    }

    for label, make_solver in solvers.items():
        prob = build_problem()

        coupled = prob.model.AS_point_0.coupled
        coupled.nonlinear_solver = make_solver()
        coupled.nonlinear_solver.options["maxiter"] = 100
        coupled.nonlinear_solver.options["atol"] = 1e-10
        coupled.nonlinear_solver.options["rtol"] = 1e-30

        coupled.nonlinear_solver.recording_options["record_abs_error"] = True
        recorder = om.SqliteRecorder(f"{label}.sql")
        coupled.nonlinear_solver.add_recorder(recorder)

        prob.run_model()
        prob.cleanup()

        cases = om.CaseReader(recorder._filepath).get_cases("root.AS_point_0.coupled.nonlinear_solver")
        res = [c.abs_err for k, c in enumerate(cases) if k == 0 or not c.name.endswith("Newton_subsolve|0")]
        plt.semilogy(range(1, len(res) + 1), res, marker="o", label=label)

        print(f"{label}:  CL = {prob.get_val('AS_point_0.wing_perf.CL')[0]:.5f}   "
              f"CD = {prob.get_val('AS_point_0.wing_perf.CD')[0]:.5f}   "
              f"tip deflection = {1e3 * prob.get_val('AS_point_0.coupled.wing.disp')[0, 2]:.4f} mm")

    om.n2(prob, outfile="n2_oas_waverider.html", show_browser=False)

    plt.axhline(1e-10, color="gray", ls="--", lw=0.8, label="atol = 1e-10")
    plt.xlabel("Nonlinear iteration")
    plt.ylabel("Absolute residual norm")
    plt.title("OpenAeroStruct waverider (CST p4): aerostructural convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.xticks(range(1, 7))
    plt.legend()
    plt.tight_layout()
    plt.savefig("convergence_oas_waverider.png", dpi=150)
