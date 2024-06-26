import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def plot_well_coords(file_path, well_name, well_img, ptss, norm, cmap):
    fig, ax1 = plt.subplots(figsize = (16, 8))
    im = ax1.imshow(well_img)
    cl = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)
    cl.ax.set_title('WS(pm)', fontsize=12)
    for i in range(ptss.shape[0]):
        ax1.plot(ptss[:, 0], ptss[:, 1] , 'ro', markersize=5)
        ax1.text(ptss[i, 0] + 1, ptss[i, 1], f"{i}", color="white")
    ax1.set_ylim((79, 0))
    plt.tight_layout()
    plt.ioff()
    plt.savefig(os.path.join(file_path, well_name + '_well_coords.png'))
    plt.close()

def plot_well_signals(file_path, well_name, well_img, times, lines, phases, norm, cmap):
    lines_plot = lines.copy()
    for p in phases:
        lines_plot[:, p] = np.nan
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (19, 8))
    im = ax1.imshow(well_img)
    cl = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)
    cl.ax.set_title('WS(pm)', fontsize=12)
    for i in range(lines.shape[0]):
        ax2.plot(times, lines_plot[i, :])
    ax1.set_xlabel("Pixel", fontsize=12)
    ax1.set_ylabel("Pixel", fontsize=12)
    ax2.set_ylabel("WS(pm)", fontsize=12)
    ax2.set_xlabel("Time(s)", fontsize=12)
    ax1.set_ylim((79, 0))
    plt.tight_layout()
    plt.ioff()
    plt.savefig(os.path.join(file_path, well_name + '_well_signals.png'))
    plt.close()

def plot_well_signal_parts(file_path, names, wells, phases):
    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize = (19, 8))
    plt.subplots_adjust(wspace=.1, hspace=.3)
    for n, name in enumerate(names):
        ax = axs[int(n / 4), n % 4] 
        mn = np.mean(wells[name], axis=(1,2))
        mn -= mn[0]
        for phase in phases:
            ax.axvline(phase, c='r')
        ax.plot(mn, label=name)
        ax.set_title(name, fontsize=8)
    plt.tight_layout()
    plt.ioff()
    plt.savefig(os.path.join(file_path, "signal-parts.png"))
    plt.close()
    
def plot_cell_signal(file_path, well_name, well_img, pts, cell_id, time, phases, line, norm, cmap):    
    line_plot = line.copy()
    for p in phases:
        line_plot[p] = np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (19, 8))
    im = ax1.imshow(well_img)
    ax1.plot([pts[0]], [pts[1]] , 'ro', markersize=5)
    ax1.text(pts[0] + 1, pts[1], f"{cell_id}", color="white")
    cl = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)
    cl.ax.set_title('WS(pm)', fontsize=12)
    ax2.plot(time, line_plot)
    ax1.set_xlabel("Pixel", fontsize=12)
    ax1.set_ylabel("Pixel", fontsize=12)
    ax2.set_ylabel("WS(pm)", fontsize=12)
    ax2.set_xlabel("Time(s)", fontsize=12)
    ax1.set_ylim((79, 0))
    plt.tight_layout()
    plt.ioff()
    plt.savefig(os.path.join(file_path, well_name + f'_cell_{cell_id}.png'))
    plt.close()

def plot_breakdown(file_path, names, wells, phases, breakdowns):
    fig, axs = plt.subplots(3, 4, sharex=True, sharey=True, figsize = (19, 8))
    plt.subplots_adjust(wspace=.1, hspace=.3)
    for n, name in enumerate(names):
        ax = axs[int(n / 4), n % 4] 
        mn = np.mean(wells[name], axis=(1,2))
        mn -= mn[0]
        for phase in phases:
            ax.axvline(phase, c='g')
        axs[n // 4, n % 4].axvline(breakdowns[name], c='r')
        ax.plot(mn, label=name)
        ax.set_title(name, fontsize=8)

    plt.tight_layout()
    plt.ioff()
    plt.savefig(os.path.join(file_path, "breakdowns.png"))
    plt.close()