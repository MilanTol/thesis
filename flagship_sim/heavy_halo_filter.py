import numpy as np
import healpy as hp


def build_contamination_map(
    ra_bright,          # deg, shape (N_bright,)
    dec_bright,         # deg, shape (N_bright,)
    z_bright,           # shape (N_bright,)
    mag_bright,         # apparent magnitude, shape (N_bright,)
    mag_threshold,      # float — brighter than this = heavy halo proxy
    theta_arcmin=10.0,  # exclusion radius in arcmin 
    nside=4096,         # ~0.86 arcmin pixels; 
):
    """
    Build a full-sky HEALPix contamination map from a bright galaxy catalog.

    Each pixel stores the maximum redshift of any bright galaxy whose
    exclusion disc covers that pixel.  A source at redshift z_s is
    contaminated if map[pixel(source)] > z_s.

    Returns
    -------
    cont_map : np.ndarray, shape (hp.nside2npix(nside),), dtype float32
        Per-pixel maximum foreground redshift.  Zero means no bright galaxy.
    """
    # only consider galaxies that are brighter than threshold
    mask = mag_bright < mag_threshold 
    ra  = ra_bright[mask]
    dec = dec_bright[mask]
    z   = z_bright[mask]
    
    # initialize healpix map
    npix = hp.nside2npix(nside)
    cont_map = 10*np.ones(npix, dtype=np.float32)
    
    # convert exclusion radius to radians
    theta_rad = np.radians(theta_arcmin / 60.0)
    
    # Convert coordinates to healpy convention: colatitude phi, longitude theta
    phi   = np.radians(ra)          
    theta = np.radians(90.0 - dec)  
    
    for i in range(len(ra)): # loop over all bright galaxies
        vec = hp.ang2vec(theta[i], phi[i]) # convert to uni vector
        
        # use query_disc to process exclusion radius:
        # Finds all HEALPix pixels whose centres fall within theta_rad of the galaxy's position. 
        # Returns an array of integer pixel indices. 
        # inclusive=True means any pixel that is at all overlapped by the disc is included, 
        # not just pixels whose centre falls strictly inside 
        pixels = hp.query_disc(nside, vec, theta_rad, inclusive=True)
        
        # use np.minimum.at to store the minimum redshift in cont_map
        # So either the current galaxy, or the already stored galaxy in the map
        np.minimum.at(cont_map, pixels, z[i]) 
                
    return cont_map
    
    
def flag_contaminated_sources(
    ra_sources,    # deg, shape (N_sources,)
    dec_sources,   # deg, shape (N_sources,)
    z_sources,     # shape (N_sources,)
    cont_map,      # output of build_contamination_map
):
    """
    For each source, check whether a bright foreground galaxy lies along
    its line of sight.

    Returns
    -------
    contaminated : np.ndarray of bool, shape (N_sources,)
        True  -> discard any pair involving this source.
        False -> source is clean.
    """
    
    nside = hp.npix2nside(len(cont_map))
    
    # Convert coordinates of sources to healpy convention: colatitude phi, longitude theta
    phi   = np.radians(ra_sources)
    theta = np.radians(90.0 - dec_sources)
    
    # store the corresponding pixel to each source galaxy
    pixels = hp.ang2pix(nside, theta, phi)
    
    # if the source redshift is greater than any heavy halo in that pixel, flag as contaminated
    contaminated =  z_sources.astype(np.float32) > cont_map[pixels] 
    
    return contaminated
    