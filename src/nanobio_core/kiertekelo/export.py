import os
import numpy as np
import pandas as pd
import matplotlib as mpl

from nanobio_core.epic_cardio.math_ops import get_avg_signal
from nanobio_core.epic_cardio.data_correction import correct_well
from nanobio_core.epic_cardio.cell_maxima import max_center_signals
from .plot import plot_well_coords, plot_well_signals, plot_cell_signal, plot_breakdown
from nanobio_core.epic_cardio.defs import WELL_NAMES

def export_results(params, path, 
        selected_ptss, filter_ptss, preprocessed_wells, time, phases, 
        raw_wells, full_time, full_phases, selected_range, breakdowns=[]):
        
    for name in selected_ptss.keys():
                
        ptss_selected = selected_ptss[name]
        
        if ptss_selected.shape[0] == 0:
            continue
        
        folder_path = os.path.join(path, name)
        if not os.path.exists(folder_path):
                os.mkdir(folder_path)
                
        lines_selected = []
        
        for i in range(ptss_selected.shape[0]):
            lines_selected.append(preprocessed_wells[name][0][:, ptss_selected[i, 1], ptss_selected[i, 0]])
        lines_selected = np.array(lines_selected)
            
        raw_lines_selected = []
        for i in range(ptss_selected.shape[0]):
            raw_lines_selected.append(raw_wells[name][:, ptss_selected[i, 1], ptss_selected[i, 0]])
        raw_lines_selected = np.array(raw_lines_selected)

        if params['coordinates']:
            pd.DataFrame(ptss_selected).to_csv(os.path.join(folder_path, name + '_coords.csv'), header=False)

        if params['preprocessed_signals']:
            pd.DataFrame(lines_selected).to_csv(os.path.join(folder_path, name + '.csv'), header=False)

        if params['raw_signals']:
            pd.DataFrame(raw_lines_selected).to_csv(os.path.join(folder_path, name + '_raw.csv'), header=False)

        if params['average_signal']:
            pd.DataFrame(get_avg_signal(raw_wells[name])).to_csv(os.path.join(folder_path, name + '_avg.csv'), header=False, index=False)
            
        well_mx = np.max(preprocessed_wells[name][0], axis=0)

        if params['max_well']:
            pd.DataFrame(well_mx).to_excel(os.path.join(folder_path, name + '_max.xlsx'))

        times = time
        cmap = mpl.cm.get_cmap('viridis', round(np.max(lines_selected)))
        norm = mpl.colors.Normalize(vmin=0, vmax=round(np.max(lines_selected)))

        if params['plot_signals_with_well']:
            plot_well_signals(folder_path, name, well_mx, times, lines_selected, phases, norm, cmap)
        
        if params['plot_well_with_coordinates']:
            plot_well_coords(folder_path, name, well_mx, ptss_selected, norm, cmap)
        
        if params['plot_cells_individually']:
            folder_path = os.path.join(path, name, "cells")
            if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
            for i in range(ptss_selected.shape[0]):
                plot_cell_signal(folder_path, name, well_mx, ptss_selected[i], i, times, phases, lines_selected[i,:], norm,cmap)
        
        if params['signal_parts_by_phases']:
            folder_path = os.path.join(path, name, "signal-parts")
            if not os.path.exists(folder_path):
                    os.mkdir(folder_path)

            for n, (start, end) in enumerate(zip([0] + phases, phases + [None])):
                selection = raw_lines_selected[:, start:end].copy()
                selection = (selection.T - selection.T[0]).T
                selection *= 1000
                pd.DataFrame(selection).to_csv(os.path.join(folder_path, f'{name}_{n}.csv'), header=False)
        
        if params['max_centered_signals']:    
            centered_lines = max_center_signals(lines_selected)
            pd.DataFrame(centered_lines).to_csv(os.path.join(folder_path, f'{name}_{n}_centered.csv'))
            
    if params['breakdown_signal']:
        breakdowns = {}
        for name in WELL_NAMES:
            line = np.mean(raw_wells[name], axis=(1,2))
            peak_until = full_phases[-1] + np.argmax(line[full_phases[-1]:])
            peak_until = peak_until if line[peak_until] > line[full_phases[-1] - 1] else full_phases[-1] - 1
            breakdowns[name] = peak_until
            
        print(breakdowns)
        
        plot_breakdown(path, WELL_NAMES, raw_wells, phases, breakdowns)
        
        for name in selected_ptss.keys():
            folder_path = os.path.join(path, name)
            ptss_selected = selected_ptss[name]
            
            slicer = slice(selected_range[0], breakdowns[name])
            # phases = [p for p in PHASES if p >= selected_range[0] and p < selected_range[1]]
            well_tmp = raw_wells[name][slicer]
            well_corr, _, _ = correct_well(well_tmp, coords=filter_ptss[name])
            breakdown_lines = []

            for i in range(ptss_selected.shape[0]):
                breakdown_lines.append(well_corr[:, ptss_selected[i, 1], ptss_selected[i, 0]])
            breakdown_lines = np.array(breakdown_lines)
            pd.DataFrame(breakdown_lines).to_csv(os.path.join(folder_path, name + '_breakdown.csv'), header=False)