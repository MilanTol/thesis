import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def fisher_corner(F, fiducial, limits, labels=None):
    """
    Parameters
    ----------
    F : (N,N) Fisher matrix
    fiducial : array-like
        Fiducial parameter values
    labels : list[str]
        Parameter names
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

        ax.plot(x, y)
        

        if limits[i] is None:
            ax.set_xlim(
                fiducial[i] - 4*sigma_i,
                fiducial[i] + 4*sigma_i,
            )
        else:
            ax.set_xlim(
                limits[i][0],
                limits[i][1]
            )
            
        ax.axvline(fiducial[i], ls="--")
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

            if limits[j] is None:
                ax.set_xlim(
                    fiducial[j] - 4*sx,
                    fiducial[j] + 4*sx
                )
            else:
                ax.set_xlim(
                    limits[j][0],
                    limits[j][1]
                )

            if limits[i] is None:
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

    plt.tight_layout()
    return fig