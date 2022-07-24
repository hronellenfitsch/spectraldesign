import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from functools import lru_cache

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgba, LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter

import seaborn as sns

from . import networks

cycle = [to_rgba(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

def flatten_alpha(over, under=np.array([1, 1, 1])):
    over = np.array(over)

    over_col = over[:3]
    over_alpha = over[3]

    return np.concatenate((over_col*over_alpha + under*(1 - over_alpha), [1.0]))

def sample_brillouin(netw, n_samples):
    a, b = netw.graph['periods']

    return [np.array([i*np.pi/a, j*np.pi/b])/(n_samples - 1) for i in range(n_samples) for j in range(n_samples)]

def sample_brillouin3d(netw, n_samples):
    a, b, c = netw.graph['periods']

    return [np.array([i*np.pi/a, j*np.pi/b, k*np.pi/c])/(n_samples - 1)
            for i in range(n_samples) for j in range(n_samples)
           for k in range(n_samples)]

def gap_sizes_2d(netw, k, opt, n_samples=21, sample_type='normal', sqrt=True,
                return_freqs=False):
    if sample_type == 'deformed':
        samplefun = netw.deformed_spectrum_at
    else:
        samplefun = netw.spectrum_at

    a, b = netw.graph['periods']

    specs = []
    for qx in np.linspace(-np.pi/a, np.pi/a, n_samples):
        for qy in np.linspace(-np.pi/b, np.pi/b, n_samples):
            q = np.array([qx, qy])
            spec = np.sort(samplefun(k, q))

            if sqrt:
                spec = np.sign(spec + 1e-10)*np.sqrt(np.abs(spec))

            # find gaps
            specs.append(spec)

    specs = np.array(specs)

    try:
        # Optimizer object
        uppers = specs[:,opt.gap_inds+1]
        lowers = specs[:,opt.gap_inds]
    except:
        # array of indices
        uppers = specs[:,opt+1]
        lowers = specs[:,opt]

    min_upper = np.min(uppers, axis=0)
    max_lower = np.max(lowers, axis=0)

    if return_freqs:
        return min_upper - max_lower, min_upper, max_lower
    else:
        return min_upper - max_lower

def gap_sizes_3d(netw, k, opt, n_samples=11, sample_type='normal', sqrt=True,
                return_freqs=False):
    if sample_type == 'deformed':
        samplefun = netw.deformed_spectrum_at
    else:
        samplefun = netw.spectrum_at

    a, b, c = netw.graph['periods']

    specs = []
    for qx in np.linspace(-np.pi/a, np.pi/a, n_samples):
        for qy in np.linspace(-np.pi/b, np.pi/b, n_samples):
            for qz in np.linspace(-np.pi/c, np.pi/c, n_samples):
                q = np.array([qx, qy, qz])
                spec = np.sort(samplefun(k, q))

                if sqrt:
                    spec = np.sign(spec + 1e-10)*np.sqrt(np.abs(spec))

                # find gaps
                specs.append(spec)

    specs = np.array(specs)

    try:
        # Optimizer object
        uppers = specs[:,opt.gap_inds+1]
        lowers = specs[:,opt.gap_inds]
    except:
        # array of indices
        uppers = specs[:,opt+1]
        lowers = specs[:,opt]

    min_upper = np.min(uppers, axis=0)
    max_lower = np.max(lowers, axis=0)

    if return_freqs:
        return min_upper - max_lower, min_upper, max_lower
    else:
        return min_upper - max_lower

def sample_spectrum_along(netw, k, q_initial, q_final, n_samples=40, sqrt=True, sample_type='normal',
                         return_array=True):
    q_vecs = (q_initial[:,np.newaxis] + (q_final[:, np.newaxis] - q_initial[:,np.newaxis])*np.linspace(0, 1, n_samples)).T

    if sample_type == 'deformed':
        samplefun = netw.deformed_spectrum_at
    elif sample_type == 'normal':
        samplefun = netw.spectrum_at
    else:
        samplefun = sample_type

    spectra = []
    for i in range(q_vecs.shape[0]):
        spec = samplefun(k, q_vecs[i,:])

        if sqrt:
            spec = np.sqrt(np.abs(spec))

        spectra.append(spec)

    dx = np.linalg.norm(q_vecs[1,:] - q_vecs[0,:])
    
    if return_array:
        return np.array(spectra), q_vecs, dx
    else:
        return spectra, q_vecs, dx

def sample_bz_spectrum_2d(netw, k, n_samples=31, sqrt=True, sample_type='normal'):
    a, b = netw.graph['periods']

    if sample_type == 'deformed':
        samplefun = netw.deformed_spectrum_at
    else:
        samplefun = netw.spectrum_at

    spectra = np.array([])
    for qx in np.linspace(-np.pi/a, np.pi/a, n_samples):
        for qy in np.linspace(-np.pi/b, np.pi/b, n_samples):
            q = np.array([qx, qy])
            spec = samplefun(k, q)

            if sqrt:
                spec = np.sign(spec + 1e-10)*np.sqrt(np.abs(spec))

            spectra = np.concatenate((spectra, spec))

    return spectra

def sample_bz_spectrum_3d(netw, k, n_samples=21, sqrt=True):
    a, b, c = netw.graph['periods']

    spectra = []
    for qx in np.linspace(0, np.pi/a, n_samples):
        for qy in np.linspace(0, np.pi/b, n_samples):
            for qz in np.linspace(0, np.pi/c, n_samples):
                q = np.array([qx, qy, qz])
                spec = netw.spectrum_at(k, q)

                if sqrt:
                    spec = np.sqrt(np.abs(spec))

                spectra.extend(spec)

    return spectra

def plot_spectrum(ax, netw, k, q_points, names, n_samples=20, sample_type='normal', show_gaps=None,
                  gap_size=gap_sizes_2d, n_samples_gs=81, sqrt=True, **kwargs):
    name_pts = [0]

    for i in range(len(q_points) - 1):
        spec, q_vecs, dx = sample_spectrum_along(netw, k, q_points[i], q_points[i+1], n_samples=n_samples,
                                                sample_type=sample_type, sqrt=sqrt)
        ax.plot(name_pts[i] + dx*np.linspace(0, 1, n_samples), spec, **kwargs)

        name_pts.append(name_pts[i] + dx)

    # plot gap in orange
    if show_gaps is not None:
        sz, upper, lower = gap_size(netw, k, show_gaps, return_freqs=True,
                                   n_samples=n_samples_gs)

        for l, u in zip(lower, upper):
            ax.fill_between([name_pts[0], name_pts[-1]], [l, l], [u, u],
                           color=cycle[1], alpha=0.4, zorder=-100)

    # plot separator lines
    for name_pt in name_pts[1:-1]:
        ax.axvline(name_pt, color='k', linewidth=0.66)

    # set ticks
    ax.set_xticks(name_pts)
    ax.set_xticklabels(names)
    ax.set_xlim(name_pts[0], name_pts[-1])
    ax.set_ylim(0)

    if show_gaps is not None:
        return lower, upper
    else:
        return None, None


# default colors
blue = sns.color_palette()[0]
blue_alpha = (blue[0], blue[1], blue[2], 0.8)
spectrum_color = blue_alpha

def plot_result_2d(netw, k, ax_netw, ax_stiff, ax_bands, ax_dos, sample_type='normal', dos_bins=101,
                  show_gaps=None, factor=1, n_samples_gs=20, n_samples=20, sqrt=True):
    a, b = netw.graph['periods']

    Γ = np.array([0, 0])
    M = np.array([0, np.pi/b])
    X = np.array([np.pi/a, np.pi/b])
    blue_muchalpha = flatten_alpha((cycle[0][0], cycle[0][1], cycle[0][2], 0.2))

    # Stiffness histogram
    if ax_stiff is not None:
        ax_stiff.hist(k, range=(0.1, 1.0), bins=21, density=False, histtype='step', orientation='horizontal')
        ax_stiff.set_xlabel('count')
        ax_stiff.xaxis.set_label_position('top')
        ax_stiff.xaxis.set_ticks_position('top')
        ax_stiff.yaxis.set_label_position('right')
        ax_stiff.yaxis.set_ticks_position('right')
        ax_stiff.set_ylabel('Stiffness $k$', labelpad=-12)
        ax_stiff.set_yticks([0.1, 1.0])

    # BZ spectrum
    if ax_dos is not None:
        bz_spec = sample_bz_spectrum_2d(netw, k, sample_type=sample_type)
        ax_dos.hist(bz_spec, bins=dos_bins, density=True, histtype='stepfilled',
                    facecolor=blue_muchalpha, edgecolor=blue_alpha, orientation='horizontal')
#         ax_dos.set_xlabel('ρ(ω)')
        ax_dos.set_ylabel(r"D.o.s. $\rho(\omega)$")
        ax_dos.set_ylim(0)
        ax_dos.yaxis.set_label_position('right')
        ax_dos.yaxis.set_ticks_position('right')

    # Network
    if ax_netw is not None:
        a, b = netw.graph['periods']
        netw.draw_edges_2d(factor*3*k, ax_netw, color=blue_alpha)
        ax_netw.set_xlim(0, a)
        ax_netw.set_ylim(0, b)
        ax_netw.set_xticks([])
        ax_netw.set_yticks([])

    # Band structure
    ls, us = plot_spectrum(ax_bands, netw, k, [M, Γ, X, M], ['M', 'Γ', 'X', 'M'],
                  color=blue_alpha, linewidth=0.7, sample_type=sample_type,
                 show_gaps=show_gaps, n_samples_gs=n_samples_gs, n_samples=n_samples, sqrt=sqrt)
    ax_bands.set_ylabel('freq.', labelpad=-8)

    yticks = ax_bands.get_yticks()
    ax_bands.set_yticks([yticks[0], yticks[-2]])

    # mark the gap in the D.O.S.
    if show_gaps is not None and ax_dos is not None:
        ax_dos.set_xlim(*ax_dos.get_xlim())
        for l, u in zip(ls, us):
            ax_dos.fill_between([0, ax_dos.get_xlim()[1]], [l, l], [u, u], color=cycle[1],
                               alpha=0.4, zorder=-100)


def render_network_3d(netw, ret, figsize=(6, 4)):
    f = plt.figure(figsize=figsize, dpi=300)

    ax1a = f.add_subplot(1, 1, 1, projection='3d')

    ax1a.axis('off')
    netw.draw_edges_3d(5*ret.x, ax1a, color=blue_alpha, plot_periodic=False)

    ax1a.view_init(20, 20)
    ax1a.dist = 6.75

    f.tight_layout(pad=0)
    f.canvas.draw()
#     f.savefig('figures/3d_tmp.png', dpi=600, bbox_inches='tight', pad_inches=0)
    data = np.frombuffer(f.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(f.canvas.get_width_height()[::-1] + (3,))

    plt.close(f)
    return data

def plot_result_3d(netw, ret, ax_netw, ax_stiff, ax_bands, ax_dos, show_gaps=None, dos_bins=201,
                   n_samples=21,
                  figsize=(6, 4)):
    a, b, c = netw.graph['periods']
    Γ3 = np.array([0,0,0])
    X3 = np.array([a,0,0])
    Y3 = np.array([0,b,0])
    Z3 = np.array([0,0,c])
    T3 = np.array([0,b,c])
    U3 = np.array([a,0,c])
    S3 = np.array([a,b,0])
    R3 = np.array([a,b,c])

    blue_muchalpha = flatten_alpha((cycle[0][0], cycle[0][1], cycle[0][2], 0.2))

    # Stiffness histogram
    ax_stiff.hist(ret.x, range=(0.1, 1.0), bins=21, density=False, histtype='step', orientation='horizontal')
    ax_stiff.set_xlabel('count')
    ax_stiff.xaxis.set_label_position('top')
    ax_stiff.xaxis.set_ticks_position('top')
    ax_stiff.yaxis.set_label_position('right')
    ax_stiff.yaxis.set_ticks_position('right')
    ax_stiff.set_ylabel('Stiffness $k$', labelpad=-12)
    ax_stiff.set_yticks([0.1, 1.0])

    # BZ spectrum
    bz_spec = sample_bz_spectrum_3d(netw, ret.x, n_samples=n_samples)
    ax_dos.hist(bz_spec, bins=dos_bins, density=True, histtype='stepfilled',
                facecolor=blue_muchalpha, edgecolor=blue_alpha, orientation='horizontal')
    ax_dos.set_ylabel(r'D.o.s. $\rho(\omega)$')
    ax_dos.set_ylim(0)
    ax_dos.yaxis.set_label_position('right')
    ax_dos.yaxis.set_ticks_position('right')

    # Network
    # pre-render network
    img = render_network_3d(netw, ret, figsize=figsize)

    ax_netw.imshow(img)
    ax_netw.set_xticks([])
    ax_netw.set_yticks([])

    # Band structure
    ls, us = plot_spectrum(ax_bands, netw, ret.x, [Γ3, X3, S3, Y3, Γ3, Z3, U3, R3, T3, Z3,],
                  ['Γ', 'X', 'S', 'Y', 'Γ', 'Z', 'U', 'R', 'T', 'Z'],
                  color=blue_alpha, linewidth=0.7,
                 show_gaps=show_gaps, gap_size=gap_sizes_3d, n_samples_gs=31)
    ax_bands.set_ylabel('freq.', labelpad=-8)

    # mark the gap in the D.O.S.
    if show_gaps is not None:
        ax_dos.set_xlim(*ax_dos.get_xlim())
        for l, u in zip(ls, us):
            ax_dos.fill_between([0, ax_dos.get_xlim()[1]], [l, l], [u, u], color=cycle[1],
                               alpha=0.4, zorder=-100)

    yticks = ax_bands.get_yticks()
    ax_bands.set_yticks([yticks[0], yticks[-2]])

def sigma2_matrix(netw, q=None):
    σ_2 = sp.sparse.bmat([[None, -1j*sp.sparse.eye(netw.number_of_nodes())],
                     [1j*sp.sparse.eye(netw.number_of_nodes()), None]])

    # Fouier transform
    if q is not None:
        dot_prods = np.dot(netw.graph['x'], q)
        phases = sp.sparse.diags(np.tile(np.exp(-1j*dot_prods), 2))
        σ_2 = phases.dot(σ_2).dot(phases.conj())

    return σ_2

def plot_band_structure_lubensky(network, k, a, b, Ω, ax, samples=20, draw_gap=True, fc='none', bandcol='k', draw_dos=True):
    """ Plot the band structure of the 2D periodic network
    with periods a and b.
    """
    Gamma = np.array([0., 0.])
    M = np.array([np.pi/a, np.pi/b])
    X = np.array([np.pi/a, 0])

    Q = network.graph['Q']
    K = sp.sparse.diags(k)

    sqrtK = sp.sparse.diags(np.sqrt(k))

#     σ_2 = sp.sparse.bmat([[None, -1j*sp.sparse.eye(network.number_of_nodes())],
#                          [1j*sp.sparse.eye(network.number_of_nodes()), None]])


    def get_spectrum_at(k, q):
        K = sp.sparse.diags(k)
        sqrtK = sp.sparse.diags(np.sqrt(k))

#         Q_q = networks.fourier_transform_Q(network, Q, q)
#         K_q = Q_q.dot(K).dot(Q_q.conj().transpose()).toarray()
#         u = sp.linalg.eigvalsh(K_q, turbo=True)

        Q_q = networks.fourier_transform_Q(network, Q, q)

        Q_sqrtK = Q_q.dot(sqrtK)
        σ_2 = sigma2_matrix(network, q)
#             U, S, Vh = sp.linalg.svd(Q_sqrtK)

#             # construct eigenstate of the "quantum" Hamiltonian
#             n_k = np.concatenate((U[:,n], Vh[n,:].conj()))

        H = sp.sparse.bmat([[2*Ω*σ_2, Q_sqrtK],
                           [Q_sqrtK.transpose().conj(), None]]).toarray()

#         H = sp.sparse.bmat([[None, sp.sparse.eye(2*network.number_of_nodes())],
#                            [-Q_sqrtK.dot(Q_sqrtK.transpose().conj()), -2j*Ω*σ_2]]).toarray()

        u = sp.linalg.eigvalsh(H, turbo=True)
#         u = np.abs(sp.linalg.eigvals(H))
#         u.sort()

#         print(v)

        return u

    def sample_direction(x, y):
        ts = np.linspace(0, 1, samples)
        spectra = []

        for t in ts:
            q = x + t*(y-x)
            u = get_spectrum_at(k, q)
            spectra.append(u)

        ts *= np.linalg.norm(x - y)
        return ts, spectra

    def sample_dos(k):
        xs = np.linspace(-np.pi/a, np.pi/a, int(samples/2)+1)
        ys = np.linspace(-np.pi/b, np.pi/b, int(samples/2)+1)

        states = []
        for x in xs:
            for y in ys:
                q = np.array([x, y])
                u = get_spectrum_at(k, q)
                states.extend(u)

        return np.array(states)

    def draw_the_gap(i0, specs):
        # find gap position
        i1 = i0 + 1
        lower = np.max([sp[i0] for sp in specs])
        upper = np.min([sp[i1] for sp in specs])

        xx0, xx1 = ax.get_xlim()
        ax.fill_between(np.linspace(xx0, xx1), lower, upper,
                             color='y', edgecolor=None, alpha=0.6, linewidth=0)
        ax.set_xlim(xx0, xx1)

        return lower, upper

    ts_1, spec_1 = sample_direction(M, Gamma)
    ts_2, spec_2 = sample_direction(Gamma, X)
    ts_3, spec_3 = sample_direction(X, M)

    ts_2 += ts_1[-1]
    ts_3 += ts_2[-1]

    ax.plot(ts_1, spec_1, color=bandcol, lw=0.3)
    ax.plot(ts_2, spec_2, color=bandcol, lw=0.3)
    ax.plot(ts_3, spec_3, color=bandcol, lw=0.3)

    ax.axvline(ts_1[-1], color='k', lw=0.75)
    ax.axvline(ts_2[-1], color='k', lw=0.75)

    ax.set_xticks([0, ts_1[-1], ts_2[-1], ts_3[-1]])
    ax.set_xticklabels([r'M', r'$\Gamma$', r'X', r'M'])

    ax.autoscale(tight=True)

    if draw_gap:
        lower, upper = draw_the_gap(Q.shape[1] + opt.gap_inds[0], spec_1 + spec_2 + spec_3)
    else:
        lower, upper = 0, 0

    ax.set_ylabel(r'Frequency $\omega$')

    # divide and plot density of states
    if draw_dos:
        states = sample_dos(k)
        states = states[states > 1e-8] # don't plot all those zero modes

        divider = make_axes_locatable(ax)
        ax_dos = divider.append_axes("right", size=0.6, pad=0.15)

        dos_range = (np.min(states), np.max(states))

        ax_dos.hist(states, orientation='horizontal', bins=200, histtype='stepfilled', normed=True, range=dos_range)

        ax_dos.set_ylabel(r'Density of states $\rho(\omega)$', labelpad=3)
        ax_dos.yaxis.set_ticks_position('right')
        ax_dos.yaxis.set_label_position('right')

        for tl in ax_dos.get_yticklabels():
            tl.set_visible(False)

    #     for tl in ax_dos.get_xticklabels():
    #         tl.set_horizontalalignment('left')

        _, y1 = ax.get_ylim()
        ax.set_ylim(0, y1*1.03)
        ax_dos.set_ylim(0, y1*1.03)
    else:
        ax_dos = None

    return ax_dos, lower, upper

def plot_band_structure_1d_lubensky(network, k, a, Ω, ax, samples=20, draw_gap=True, fc='none', bandcol='k',
                          number_bands=False, plot_localization=False, direction='x',
                          lw=2):
    """ Plot the band structure of the 2D periodic network
    with periods a and b.
    """
    if direction == 'x':
        Left = np.array([-np.pi/a, 0])
        Right = np.array([np.pi/a, 0])
    else:
        Left = np.array([0, -np.pi/a])
        Right = np.array([0, np.pi/a])

    Q = network.graph['Q']
    K = sp.sparse.diags(k)
    sqrtK = sp.sparse.diags(np.sqrt(k))

    def get_spectrum_at(k, q, localization=False):
        K = sp.sparse.diags(k)
        sqrtK = sp.sparse.diags(np.sqrt(k))

        Q_q = networks.fourier_transform_Q(network, Q, q)
        σ_2 = sigma2_matrix(network, q)

        Q_sqrtK = Q_q.dot(sqrtK)
        H = sp.sparse.bmat([[2*Ω*σ_2, Q_sqrtK],
                           [Q_sqrtK.transpose().conj(), None]]).toarray()

        u, v = sp.linalg.eigh(H, turbo=True)


        if localization:
            loc = np.sum(np.abs(v)**4, axis=0)
            return u, loc
        else:
            return u

    def sample_direction(x, y, localization=False):
        ts = np.linspace(0, 1, samples)
        spectra = []
        locs = []

        for t in ts:
            q = x + t*(y-x)
            u = get_spectrum_at(k, q, localization)

            if localization:
                u, loc = u
                locs.append(loc)

            spectra.append(u)

        ts *= np.linalg.norm(x - y)

        if localization:
            return ts, spectra, locs
        else:
            return ts, spectra

    def sample_dos(k):
        xs = np.linspace(-np.pi/a, np.pi/a, 2*samples + 1)

        states = []
        for x in xs:
            q = np.array([x, 0])
            u = get_spectrum_at(k, q)
            states.extend(u)

        return states

    def draw_the_gap(i0, specs):
        # find gap position
        i1 = i0 + 1
        lower = np.max([sp[i0] for sp in specs])
        upper = np.min([sp[i1] for sp in specs])

        xx0, xx1 = ax.get_xlim()
        ax.fill_between(np.linspace(xx0, xx1), lower, upper,
                             color='y', edgecolor=None, alpha=0.6, linewidth=0)
        ax.set_xlim(xx0, xx1)

        return lower, upper

    # plot the bands
    if plot_localization:
        ts_1, spec_1, loc = sample_direction(Left, Right, localization=plot_localization)

        spec_1 = np.array(spec_1)
        loc = np.array(loc)

        cmin, cmax = np.min(np.log10(loc)), np.max(np.log10(loc))
        for band, band_loc in zip(spec_1.T, loc.T):
            c_line = plot_colored_line(ax, ts_1 - np.pi/a, band, np.log10(band_loc), cmin=cmin, cmax=cmax,
                             lw=lw)
    else:
        ts_1, spec_1 = sample_direction(Left, Right, localization=plot_localization)

        ax.plot(ts_1-np.pi/a, spec_1, color=bandcol, lw=0.3)


#     ax.set_xticks([-np.pi/a, 0, np.pi/a])
#     ax.set_xticklabels(["-π/a", "0", "π/a"])

    ax.autoscale(tight=True)

    if draw_gap:
        draw_the_gap(opt.gap_inds[0], spec_1)


    ax.set_ylabel(r'Frequency $\omega$')

    if number_bands:
        i = int(samples/2)
        for j, s in enumerate(spec_1[i]):
            ax.text(0, s, str(j))

    # divide and plot density of states
#     states = sample_dos(k)

#     divider = make_axes_locatable(ax)
#     ax_dos = divider.append_axes("right", size=0.6, pad=0.15)

#     dos_range = (np.min(states), np.max(states))

#     ax_dos.hist(states, orientation='horizontal', bins=100, histtype='stepfilled', normed=True, range=dos_range)

#     ax_dos.set_ylabel(r'Density of states $\rho(\omega)$')
#     ax_dos.yaxis.set_ticks_position('right')
#     ax_dos.yaxis.set_label_position('right')

#     for tl in ax_dos.get_yticklabels():
#         tl.set_visible(False)

# #     for tl in ax_dos.get_xticklabels():
# #         tl.set_horizontalalignment('left')

#     _, y1 = ax.get_ylim()
#     ax.set_ylim(0, y1*1.03)
#     ax_dos.set_ylim(0, y1*1.03)

    ax_dos = None
    if plot_localization:
#         ax_cb = divider.append_axes("top", size=0.1, pad=0.5)
        return ax_dos, c_line, spec_1, loc

    return ax_dos

def get_modes_at(netw, k, q, Ω):
    K = sp.sparse.diags(k)

    sqrtK = sp.sparse.diags(np.sqrt(k))

    σ_2 = sigma2_matrix(netw, q)

    Q = netw.graph['Q']
    Q_q = networks.fourier_transform_Q(netw, Q, q)

    Q_sqrtK = Q_q.dot(sqrtK)
    H = sp.sparse.bmat([[2*Ω*σ_2, Q_sqrtK],
                       [Q_sqrtK.transpose().conj(), None]]).toarray()

    u, v = sp.linalg.eigh(H, turbo=True)

    return u, v[:σ_2.shape[0],:]

def get_modes_at_real(netw, k, q):
    K = sp.sparse.diags(k)

#     σ_2 = sigma2_matrix(netw, q=None)

    Q = netw.graph['Q']
    Q_q = networks.fourier_transform_Q(netw, Q, q)

    H = Q_q.dot(K).dot(Q_q.conj().transpose())

    u, v = sp.linalg.eigh(H.toarray(), turbo=True)

    return u, v

def draw_mode_loc(netw, v, ax):
    pos = netw.graph['pos']
    x = np.array([pos[n] for n in netw.graph['nodelist']])

    ax.scatter(x[:,0], x[:,1], s=100*v, c=sns.color_palette()[0], zorder=100, alpha=1.0)


def sample_dos(netw, k, Ω, samples):
    a, b = netw.graph['periods']

    xs = np.linspace(-np.pi/a, np.pi/a, int(samples/2)+1)
    ys = np.linspace(-np.pi/b, np.pi/b, int(samples/2)+1)

    states = []
    for x in xs:
        for y in ys:
            q = np.array([x, y])
            u, v = get_modes_at(netw, k, q, Ω)
            states.append(u)

    return np.array(states)

def all_gaps_at(netw, k, Ω, samples=11):
    """ Return a vector containing all gap sizes at given rotation rate.
    The entrire BZ is sampled with the given sampling.
    """
    states = sample_dos(netw, k, Ω, samples)

    min_bands = np.min(states, axis=0)
    max_bands = np.max(states, axis=0)

    gap_sizes = min_bands[1:] - max_bands[:-1]
    gap_sizes[gap_sizes < 0] = 0.0

    return gap_sizes

def plot_colored_line(ax, x, y, c, cmap='viridis', cmin=None, cmax=None, lw=1.0):
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(c.min() if cmin is None else cmin,
                         c.max() if cmax is None else cmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    # Set the values used for colormapping
    lc.set_array(c)
    lc.set_linewidth(lw)

    line = ax.add_collection(lc)

    return line
