import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def fisher_corner(F, fiducial, limits=None, labels=None, diag_gap=0.015,
                   prior_sigmas=None):
    """
    Parameters
    ----------
    F : (N,N) Fisher matrix
    fiducial : array-like
        Fiducial parameter values
    labels : list[str]
        Parameter names
    diag_gap : float
        Extra whitespace (in figure-fraction units) inserted between each
        diagonal Gaussian panel and the panel(s) below it in the same column.
    prior_sigmas : array-like, optional
        1-sigma prior width for each parameter, in the same order as
        `fiducial`. If given, each diagonal panel shows the prior width as a
        shaded band + dashed vertical lines at fiducial +/- prior_sigma, so
        the posterior (solid curve) can be visually compared against the
        prior it was regularized with.
    """

    cov = np.linalg.inv(F)

    npar = len(fiducial)

    if labels is None:
        labels = [f"p{i}" for i in range(npar)]

    fig, axes = plt.subplots(
        npar, npar,
        figsize=(2*npar, 2*npar)
    )

    for i in range(npar):

        sigma_i = np.sqrt(cov[i, i])

        # Diagonal: 1D Gaussian
        ax = axes[i, i]

        x = np.linspace(
            fiducial[i] - 4*sigma_i,
            fiducial[i] + 4*sigma_i,
            500
        )

        y = np.exp(
            -(x - fiducial[i])**2 /
            (2*sigma_i**2)
        )

        # Prior band drawn first so the posterior curve sits on top
        if prior_sigmas is not None:
            sigma_prior_i = prior_sigmas[i]

            ax.axvspan(
                fiducial[i] - sigma_prior_i,
                fiducial[i] + sigma_prior_i,
                color="gray",
                alpha=0.15,
                zorder=0,
                label="prior" if i == 0 else None,
            )

            for sign in (-1, 1):
                ax.axvline(
                    fiducial[i] + sign * sigma_prior_i,
                    color="gray",
                    ls="--",
                    lw=1,
                    zorder=1,
                )

        ax.plot(x, y, zorder=2)

        if limits is None:
            ax.set_xlim(
                fiducial[i] - 4*sigma_i,
                fiducial[i] + 4*sigma_i,
            )
        elif limits[i] is None:
            ax.set_xlim(
                fiducial[i] - 4*sigma_i,
                fiducial[i] + 4*sigma_i,
            )
        else:
            ax.set_xlim(
                limits[i][0],
                limits[i][1]
            )

        ax.axvline(fiducial[i], ls="--", color="black", lw=1, zorder=2)
        ax.set_yticks([])

        if i == npar - 1:
            ax.set_xlabel(labels[i])

        # Lower triangle: confidence ellipses
        for j in range(i):

            ax = axes[i, j]

            subcov = cov[np.ix_([j, i], [j, i])]

            vals, vecs = np.linalg.eigh(subcov)

            order = vals.argsort()[::-1]
            vals = vals[order]
            vecs = vecs[:, order]

            angle = np.degrees(
                np.arctan2(vecs[1, 0], vecs[0, 0])
            )

            # 68% and 95%
            scales = [1.52, 2.48]

            for scale in scales:

                width = 2 * scale * np.sqrt(vals[0])
                height = 2 * scale * np.sqrt(vals[1])

                ellipse = Ellipse(
                    (fiducial[j], fiducial[i]),
                    width,
                    height,
                    angle=angle,
                    fill=False
                )

                ax.add_patch(ellipse)

            ax.plot(
                fiducial[j],
                fiducial[i],
                marker="+"
            )

            sx = np.sqrt(cov[j, j])
            sy = np.sqrt(cov[i, i])
            if limits is None:
                ax.set_xlim(
                    fiducial[j] - 4*sx,
                    fiducial[j] + 4*sx,
                )
            elif limits[j] is None:
                ax.set_xlim(
                    fiducial[j] - 4*sx,
                    fiducial[j] + 4*sx
                )
            else:
                ax.set_xlim(
                    limits[j][0],
                    limits[j][1]
                )
            if limits is None:
                ax.set_ylim(
                    fiducial[i] - 4*sy,
                    fiducial[i] + 4*sy,
                )
            elif limits[i] is None:
                ax.set_ylim(
                    fiducial[i] - 4*sy,
                    fiducial[i] + 4*sy
                )
            else:
                ax.set_ylim(
                    limits[i][0],
                    limits[i][1]
                )

            if i == npar - 1:
                ax.set_xlabel(labels[j])

            if j == 0:
                ax.set_ylabel(labels[i])

        # Hide upper triangle
        for j in range(i+1, npar):
            axes[i, j].axis("off")

    # Hide ticks for non-edge plots, EXCEPT always keep x-ticks on the diagonal
    for i in range(npar):
        for j in range(npar):
            if j > i:
                continue  # upper triangle already hidden
            ax = axes[i, j]
            if j != 0:
                ax.set_yticks([])
            if i != npar - 1 and j != i:
                ax.set_xticks([])

    plt.subplots_adjust(wspace=0.1, hspace=0.05)

    # Add extra whitespace ONLY below each diagonal panel, without touching
    # spacing anywhere else in the grid.
    for i in range(npar):
        ax = axes[i, i]
        pos = ax.get_position()
        ax.set_position([
            pos.x0,
            pos.y0 + diag_gap,
            pos.width,
            pos.height - diag_gap
        ])

    if prior_sigmas is not None:
        handles, leg_labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, leg_labels, loc="upper right")

    return fig