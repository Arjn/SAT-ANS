from urllib.parse import urlparse
from collections import OrderedDict
from astropy.utils.data import download_file
from astropy.utils import indent
from astropy import _erfa as erfa
from astropy.coordinates.builtin_frames.utils import get_jd12
from astropy.coordinates.orbital_elements import calc_moon
from astropy.coordinates import CartesianRepresentation#, get_body_barycentric_posvel, \
    #ICRS, GCRS, CartesianDifferential
from astropy.coordinates import solar_system_ephemeris
from astropy import units as u
from poliastro.constants import J2000
from poliastro.core.elements import rv2coe
from poliastro.util import transform as transform_vector

import numpy as np

__all__ = ["get_body_barycentric",
           "get_body_barycentric_posvel", "solar_system_ephemeris"]


DEFAULT_JPL_EPHEMERIS = 'de430'

"""List of kernel pairs needed to calculate positions of a given object."""
BODY_NAME_TO_KERNEL_SPEC = OrderedDict(
                                      (('sun', [(0, 10)]),
                                       ('mercury', [(0, 1), (1, 199)]),
                                       ('venus', [(0, 2), (2, 299)]),
                                       ('earth-moon-barycenter', [(0, 3)]),
                                       ('earth', [(0, 3), (3, 399)]),
                                       ('moon', [(0, 3), (3, 301)]),
                                       ('mars', [(0, 4)]),
                                       ('jupiter', [(0, 5)]),
                                       ('saturn', [(0, 6)]),
                                       ('uranus', [(0, 7)]),
                                       ('neptune', [(0, 8)]),
                                       ('pluto', [(0, 9)]))
                                      )

"""Indices to the plan94 routine for the given object."""
PLAN94_BODY_NAME_TO_PLANET_INDEX = OrderedDict(
    (('mercury', 1),
     ('venus', 2),
     ('earth-moon-barycenter', 3),
     ('mars', 4),
     ('jupiter', 5),
     ('saturn', 6),
     ('uranus', 7),
     ('neptune', 8)))

_EPHEMERIS_NOTE = """
You can either give an explicit ephemeris or use a default, which is normally
a built-in ephemeris that does not require ephemeris files.  To change
the default to be the JPL ephemeris::

    >>> from astropy.coordinates import solar_system_ephemeris
    >>> solar_system_ephemeris.set('jpl')  # doctest: +SKIP

Use of any JPL ephemeris requires the jplephem package
(https://pypi.python.org/pypi/jplephem).
If needed, the ephemeris file will be downloaded (and cached).

One can check which bodies are covered by a given ephemeris using::
    >>> solar_system_ephemeris.bodies
    ('earth', 'sun', 'moon', 'mercury', 'venus', 'earth-moon-barycenter', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune')
"""[1:-1]

def _get_kernel(value):
    """
    Try importing jplephem, download/retrieve from cache the Satellite Planet
    Kernel corresponding to the given ephemeris.
    """
    if value is None or value.lower() == 'builtin':
        return None

    if value.lower() == 'jpl':
        value = DEFAULT_JPL_EPHEMERIS

    if value.lower() in ('de430', 'de432s'):
        value = ('http://naif.jpl.nasa.gov/pub/naif/generic_kernels'
                 '/spk/planets/{:s}.bsp'.format(value.lower()))
    else:
        try:
            urlparse(value)
        except Exception:
            raise ValueError('{} was not one of the standard strings and '
                             'could not be parsed as a URL'.format(value))

    try:
        from jplephem.spk import SPK
    except ImportError:
        raise ImportError("Solar system JPL ephemeris calculations require "
                          "the jplephem package "
                          "(https://pypi.python.org/pypi/jplephem)")

    return SPK.open(download_file(value, cache=True))

def _get_body_barycentric_posvel(body, time, ephemeris=None,
                                 get_velocity=True, Ephemkernel=None):
    """Calculate the barycentric position (and velocity) of a solar system body.

    Parameters
    ----------
    body : str or other
        The solar system body for which to calculate positions.  Can also be a
        kernel specifier (list of 2-tuples) if the ``ephemeris`` is a JPL
        kernel.
    time : `~astropy.time.Time`
        Time of observation.
    ephemeris : str, optional
        Ephemeris to use.  By default, use the one set with
        ``astropy.coordinates.solar_system_ephemeris.set``
    get_velocity : bool, optional
        Whether or not to calculate the velocity as well as the position.

    Returns
    -------
    position : `~astropy.coordinates.CartesianRepresentation` or tuple
        Barycentric (ICRS) position or tuple of position and velocity.

    Notes
    -----
    No velocity can be calculated with the built-in ephemeris for the Moon.

    Whether or not velocities are calculated makes little difference for the
    built-in ephemerides, but for most JPL ephemeris files, the execution time
    roughly doubles.
    """

    # if ephemeris is None:
    #     ephemeris = solar_system_ephemeris.get()
    #     if ephemeris is None:
    #         raise ValueError(_EPHEMERIS_NOTE)
    #     kernel = solar_system_ephemeris.kernel
    # else:
    #     kernel = _get_kernel(ephemeris)

    kernel = _get_kernel(ephemeris) if Ephemkernel is None else Ephemkernel

    jd1, jd2 = get_jd12(time, 'tdb')
    if kernel is None:
        body = body.lower()
        earth_pv_helio, earth_pv_bary = erfa.epv00(jd1, jd2)
        if body == 'earth':
            body_pv_bary = earth_pv_bary

        elif body == 'moon':
            if get_velocity:
                raise KeyError("the Moon's velocity cannot be calculated with "
                               "the '{0}' ephemeris.".format(ephemeris))
            return calc_moon(time).cartesian

        else:
            sun_pv_bary = earth_pv_bary - earth_pv_helio
            if body == 'sun':
                body_pv_bary = sun_pv_bary
            else:
                try:
                    body_index = PLAN94_BODY_NAME_TO_PLANET_INDEX[body]
                except KeyError:
                    raise KeyError("{0}'s position and velocity cannot be "
                                   "calculated with the '{1}' ephemeris."
                                   .format(body, ephemeris))
                body_pv_helio = erfa.plan94(jd1, jd2, body_index)
                body_pv_bary = body_pv_helio + sun_pv_bary

        body_pos_bary = CartesianRepresentation(
            body_pv_bary[..., 0, :], unit=u.au, xyz_axis=-1, copy=False)
        if get_velocity:
            body_vel_bary = CartesianRepresentation(
                body_pv_bary[..., 1, :], unit=u.au/u.day, xyz_axis=-1,
                copy=False)

    else:
        if isinstance(body, str):
            # Look up kernel chain for JPL ephemeris, based on name
            try:
                kernel_spec = BODY_NAME_TO_KERNEL_SPEC[body.lower()]
            except KeyError:
                raise KeyError("{0}'s position cannot be calculated with "
                               "the {1} ephemeris.".format(body, ephemeris))
        else:
            # otherwise, assume the user knows what their doing and intentionally
            # passed in a kernel chain
            kernel_spec = body

        # jplephem cannot handle multi-D arrays, so convert to 1D here.
        jd1_shape = getattr(jd1, 'shape', ())
        if len(jd1_shape) > 1:
            jd1, jd2 = jd1.ravel(), jd2.ravel()
        # Note that we use the new jd1.shape here to create a 1D result array.
        # It is reshaped below.
        body_posvel_bary = np.zeros((2 if get_velocity else 1, 3) +
                                     getattr(jd1, 'shape', ()))
        for pair in kernel_spec:
            spk = kernel[pair]
            if spk.data_type == 3:
                # Type 3 kernels contain both position and velocity.
                posvel = spk.compute(jd1, jd2)
                if get_velocity:
                    body_posvel_bary += posvel.reshape(body_posvel_bary.shape)
                else:
                    body_posvel_bary[0] += posvel[:4]
            else:
                # spk.generate first yields the position and then the
                # derivative. If no velocities are desired, body_posvel_bary
                # has only one element and thus the loop ends after a single
                # iteration, avoiding the velocity calculation.
                for body_p_or_v, p_or_v in zip(body_posvel_bary,
                                               spk.generate(jd1, jd2)):
                    body_p_or_v += p_or_v

        body_posvel_bary.shape = body_posvel_bary.shape[:2] + jd1_shape
        body_pos_bary = CartesianRepresentation(body_posvel_bary[0],
                                                unit=u.km, copy=False)
        if get_velocity:
            body_vel_bary = CartesianRepresentation(body_posvel_bary[1],
                                                    unit=u.km/u.day, copy=False)

    return (body_pos_bary, body_vel_bary) if get_velocity else body_pos_bary


def get_body_barycentric_posvel(body, time, ephemeris=None, Ephemkernel=None):
    """Calculate the barycentric position and velocity of a solar system body.

    Parameters
    ----------
    body : str or other
        The solar system body for which to calculate positions.  Can also be a
        kernel specifier (list of 2-tuples) if the ``ephemeris`` is a JPL
        kernel.
    time : `~astropy.time.Time`
        Time of observation.
    ephemeris : str, optional
        Ephemeris to use.  By default, use the one set with
        ``astropy.coordinates.solar_system_ephemeris.set``

    Returns
    -------
    position, velocity : tuple of `~astropy.coordinates.CartesianRepresentation`
        Tuple of barycentric (ICRS) position and velocity.

    See also
    --------
    get_body_barycentric : to calculate position only.
        This is faster by about a factor two for JPL kernels, but has no
        speed advantage for the built-in ephemeris.

    Notes
    -----
    The velocity cannot be calculated for the Moon.  To just get the position,
    use :func:`~astropy.coordinates.get_body_barycentric`.

    """
    ephem_kernel = Ephemkernel
    return _get_body_barycentric_posvel(body, time, ephemeris, Ephemkernel=ephem_kernel)


get_body_barycentric_posvel.__doc__ += indent(_EPHEMERIS_NOTE)[4:]


def get_body_barycentric(body, time, ephemeris=None, Ephemkernel=None):
    """Calculate the barycentric position of a solar system body.

    Parameters
    ----------
    body : str or other
        The solar system body for which to calculate positions.  Can also be a
        kernel specifier (list of 2-tuples) if the ``ephemeris`` is a JPL
        kernel.
    time : `~astropy.time.Time`
        Time of observation.
    ephemeris : str, optional
        Ephemeris to use.  By default, use the one set with
        ``astropy.coordinates.solar_system_ephemeris.set``

    Returns
    -------
    position : `~astropy.coordinates.CartesianRepresentation`
        Barycentric (ICRS) position of the body in cartesian coordinates

    See also
    --------
    get_body_barycentric_posvel : to calculate both position and velocity.

    Notes
    -----
    """
    return _get_body_barycentric_posvel(body, time, ephemeris,
                                        get_velocity=False, Ephemkernel=None)


def body_centered_to_icrs(r, v, source_body, epoch=J2000, rotate_meridian=False, ephemeris=None, Ephemkernel=None):
    """Converts position and velocity body-centered frame to ICRS.

    Parameters
    ----------
    r : ~astropy.units.Quantity
        Position vector in a body-centered reference frame.
    v : ~astropy.units.Quantity
        Velocity vector in a body-centered reference frame.
    source_body : Body
        Source body.
    epoch : ~astropy.time.Time, optional
        Epoch, default to J2000.
    rotate_meridian : bool, optional
        Whether to apply the rotation of the meridian too, default to False.

    Returns
    -------
    r, v : tuple (~astropy.units.Quantity)
        Position and velocity vectors in ICRS.


    """
    ephem_kernel = Ephemkernel
    ra, dec, W = source_body.rot_elements_at_epoch(epoch)

    if rotate_meridian:
        r = transform_vector(r, -W, 'z')
        v = transform_vector(v, -W, 'z')

    r_trans1 = transform_vector(r, -(90 * u.deg - dec), 'x')
    r_trans2 = transform_vector(r_trans1, -(90 * u.deg + ra), 'z')

    v_trans1 = transform_vector(v, -(90 * u.deg - dec), 'x')
    v_trans2 = transform_vector(v_trans1, -(90 * u.deg + ra), 'z')

    icrs_frame_pos_coord, icrs_frame_vel_coord = get_body_barycentric_posvel(source_body.name, time=epoch, ephemeris=ephemeris, Ephemkernel=ephem_kernel)

    r_f = icrs_frame_pos_coord.xyz + r_trans2
    v_f = icrs_frame_vel_coord.xyz + v_trans2

    return r_f.to(r.unit), v_f.to(v.unit), np.array([ra.value,dec.value])



get_body_barycentric.__doc__ += indent(_EPHEMERIS_NOTE)[4:]