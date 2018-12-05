#!/usr/bin/env python3
import libsbml
import os
import sys
#sys.path.insert(0, os.path.split(os.path.split(os.getcwd())[0])[0])
sys.path.insert(0, '/home/dweindl/src/pyPESTO')

import pypesto
import sys
import amici
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pypesto.objective.amici_objective import (import_sbml_model,
                                               amici_objective_from_measurement_file)
import petab


models = ['Bachmann_MSB2011', 'beer_MolBioSystems2014', 'boehm_JProteomeRes2014',
 'Borghans_BiophysChem1997', 'Brannmark_JBC2010', 'Bruno_JExpBio2016',
'Chen_MSB2009', 'Crauste_CellSystems2017', 'Elowitz_Nature2000',
'Fiedler_BMC2016', 'Fujita_SciSignal2010', 'Hass_PONE2017',
'Isensee_JCB2018', 'Lucarelli_CellSystems_2018',
'Merkle_PCB2016', 'Raia_CancerResearch2011',
'Schwen_PONE2014','Sneyd_PNAS2002', 'Sobotta_Frontiers2017',
'Swameye_PNAS2003', 'Weber_BMC2015','Zheng_PNAS2012']


#model_root = os.path.abspath(os.path.join('Benchmark-Models', 'hackathon_contributions_new_data_format'))
model_root = '/home/dweindl/src/Benchmark-Models/hackathon_contributions_new_data_format/'

benchmark_model = 'Bruno_JExpBio2016' # 'Zheng_PNAS2012'
condition_filename = os.path.join(model_root, benchmark_model, f'experimentalCondition_{benchmark_model}.tsv')
measurement_filename = os.path.join(model_root, benchmark_model,f'measurementData_{benchmark_model}.tsv')
parameter_filename = os.path.join(model_root, benchmark_model, f'parameters_{benchmark_model}.tsv')
sbml_model_file = os.path.join(model_root, benchmark_model, f'model_{benchmark_model}.xml')
model_name = f'model_{benchmark_model}'
model_output_dir = f'deleteme-{model_name}'

rebuild = False
rebuild = True
if rebuild:
    import_sbml_model(sbml_model_file=sbml_model_file,
                      condition_file=condition_filename,
                      measurement_file=measurement_filename,
                      model_output_dir=model_output_dir,
                      model_name=model_name)


sys.path.insert(0, os.path.abspath(model_output_dir))
model_module = importlib.import_module(model_name)

model = model_module.getModel()
model.requireSensitivitiesForAllParameters()

solver = model.getSolver()
solver.setSensitivityMethod(amici.SensitivityMethod_forward)
solver.setSensitivityOrder(amici.SensitivityOrder_first)

print("Model parameters:", list(model.getParameterIds()))
print()
print("Model const parameters:", list(model.getFixedParameterIds()))
print()
print("Model outputs:   ", list(model.getObservableIds()))
print()
print("Model states:    ", list(model.getStateIds()))

from pypesto.logging import log_to_console
log_to_console()

# Create objective function instance from model and measurements
model.setParameterScale(amici.ParameterScaling_log10)

petab_problem = petab.OptimizationProblem(sbml_model_file,
                                    measurement_filename,
                                    condition_filename,
                                    parameter_filename)

objective = amici_objective_from_measurement_file(amici_model=model,
                                                  sbml_model=petab_problem.sbml_model,
                                                  condition_df=petab_problem.condition_df,
                                                measurement_df=petab_problem.measurement_df,
                                                amici_solver=solver)


# load nominal parameters from parameter description file
parameter_df = petab.get_parameter_df(parameter_filename)

#print(parameter_df)
model_parameters = set(model.getParameterIds())
parameter_df_parameters = set(parameter_df.index)

print("only in model:", model_parameters - parameter_df_parameters)
print("only in table", parameter_df_parameters - model_parameters)
print("opt param only not in table", set(objective.optimization_parameter_ids) - parameter_df_parameters)


nominal_x = parameter_df.loc[objective.optimization_parameter_ids, 'nominalValue'].values
#nominal_x = np.power(10, nominal_x)
for i, p in enumerate(model.getParameterIds()):
    if p.startswith('noise'):
        nominal_x[i] = -2.63 #np.power(10, -2.63)
print(nominal_x)


# evaluate with nominal parameters
llh = objective(x=nominal_x)
print(f'llh: {llh}, lh: {np.exp(llh)}')