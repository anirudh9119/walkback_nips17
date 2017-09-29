


## -*- compile-command: "ipython ./vector_field_sim.py --pylab" -*-
""" 2D vector fields simulation and visualization
Tools to simulate and visualize 2d vector fields by defining their divergence-free and curl-free parts.
:platform: Unix, Windows
:synopsis:
Take from Ishmael
"""

from numpy import (exp, log, pi,
                   meshgrid, linspace,
                   array)

from numpy.random import (choice, )

from matplotlib.pyplot import (subplots, quiver, show,
                               setp)
from matplotlib.patheffects import (withStroke, )

def gaussian_pdf(x, y, mu_x, mu_y, sigma_x, sigma_y, rho=0, log_domain=False):
    """ computes the density of a bivariate gaussian.
    """
    x_centered = x - mu_x
    y_centered = y - mu_y
    x_std = (x_centered**2)/sigma_x**2
    y_std = (y_centered**2)/sigma_y**2
    z = x_std - (2 * rho * x_centered * y_centered)/(sigma_y * sigma_x) + y_std
    log_constant = -log(2 * pi) - log(sigma_x) - log(sigma_y) - 0.5 * log(1 - rho**2)
    log_pdf = log_constant - z/(2 * (1 - rho**2))
    return log_pdf if log_domain else exp(log_pdf)


def grad_gauss_2d(x, y, mu_x, mu_y, sigma_x, sigma_y, rho=0):
    """ return Gradient and skew gradient of a bivariate normal.
    """
    pdf = gaussian_pdf(x, y, mu_x, mu_y, sigma_x, sigma_y, rho=rho)
    dx = - (x - mu_x)/(sigma_x**2 * (1 - rho**2)) + (rho * (y - mu_y))/(sigma_x * sigma_y * (1 - rho**2))
    dy = - (y - mu_y)/(sigma_y**2 * (1 - rho**2)) + (rho * (x - mu_x))/(sigma_x * sigma_y * (1 - rho**2))
    grad = array([dx, dy]) * pdf
    skew_grad = array([-dy, dx]) * pdf
    return (grad, skew_grad)


def make_field(means, sigmas, components, rhos=None):
    """Creates a function corresponding to a two dimensional vector field with
    orthogonal divergence free and curl free components by summing the gradient
    and skew gradient of a gaussian mixture.
    """
    assert len(means) > 0
    if rhos is None:
        rhos = [0] * len(means)

    assert len(means) == len(sigmas) == len(components) == len(rhos)

    def Field(x, y):
        """ Returns 4-tuple where: (Vector field, grad vector field, grad div-free part,
        grad curl-free part)
        vector field = div-free part + curl-free part
        """
        grad_F_cons = 0
        grad_F_curl = 0
        grad_F = 0
        F_val = 0
        for i in range(len(means)):
            mu_x_i, mu_y_i = means[i]
            sigma_x_i, sigma_y_i = sigmas[i]
            rho_i = rhos[i]
            a_i = components[i]
            ## Computing gradients
            grad_i, skew_grad_i = grad_gauss_2d(x, y,
                                                mu_x_i, mu_y_i,
                                                sigma_x_i, sigma_y_i,
                                                rho_i)
            grad_F_cons += a_i * grad_i
            grad_F_curl += a_i * skew_grad_i
            grad_F += grad_F_cons + grad_F_curl
            ## Computing Vector field
            F_val += a_i * gaussian_pdf(x, y,
                                        mu_x_i, mu_y_i,
                                        sigma_x_i, sigma_y_i,
                                        rho_i)
        return (F_val, grad_F, grad_F_cons, grad_F_curl)
    return Field


def visualize_vector_field(x, y, dx, dy, p, **kwargs):
    """ Visualize a vector field in two dimensions.
    """

    #title = 'Vector Field: '
    #if 'title' in kwargs:
    #    title += kwargs['title']

    skip = (slice(None, None, 5), slice(None, None, 5))

    fig, ax = subplots()
    #tight_layout()

    quiver(x[skip], y[skip], dx[skip], dy[skip], p[skip])
    #ax.set(aspect=1, title=title)
    show()
    import pdb; pdb.set_trace()
    #plt.savefig('./figs/relu_mag_curl.pdf',bbox_inches='tight')


def visualize_contours(x, y, dx, dy, p, **kwargs):
    """
        visualize contours in two dimensions
    """
    title = 'contour plot: '
    if 'title' in kwargs:
        title += kwargs['title']
    density = 1.0
    if 'density' in kwargs:
        density = kwargs['density']
    cmap = 'gist_earth'
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    linewidth = 0.5  ##10 * hypot(dx, dy)
    if 'linewidth' in kwargs:
        linewidth = kwargs['linewidth']

    fig, ax = subplots()

    ax.streamplot(x, y, dx, dy, color=p, density=density,
                  cmap=cmap, linewidth=linewidth)

    cont = ax.contour(x, y, p, cmap=cmap, vmin=p.min(), vmax=p.max())
    labels = ax.clabel(cont)
    setp(labels, path_effects=[withStroke(linewidth=8, foreground='w')])
    ax.set(aspect=1, title=title)
    show()


def visualize_3d(x, y, p, **kwargs):
    """ visualize 3d density plot with projection
    """
    title = '3d density plot: '
    if 'title' in kwargs:
        title += kwargs['title']

    fig, ax = subplots()
    ax.plot_surface(x, y, p, rstride=4, cstride=4, alpha=0.25)
    cset = ax.contour(x, y, p, zdir='z')
    cset = ax.contour(x, y, p, zdir='x')
    cset = ax.contour(x, y, p, zdir='y')

    ax.set_xlim3d(x.min(), x.max())
    ax.set_ylim3d(y.min(), y.max())
    ax.set_zlim3d(0, 1)
    show()


class VectorField(object):
    "Sampling and Visualizing 2D Gaussian mixture vector Field"
    def __init__(self, means, sigmas, components, rhos=None):
        """ Computes the vector field Function over a grid
        means: list of 2d arrays reprensenting the means for each of the gaussian mixture components
        sigmas: list of 2d arrays representing standard deviations of the gaussian mixture components
        components: list of scalars representing the priors of the gaussian mixture components
        rhos: list of scalars representing the correlation of the gaussian mixture components
        """

        ## Registering parameters
        self.params = locals().copy()
        self.params.pop('self')
        ## Building vector field
        self._field = make_field(**self.params)
        self._grid_computed = False
        self._grid_x = None
        self._grid_y = None
        self.grid = None
        self.density = None
        self.vector_field = None
        self.curl_free = None
        self.div_free = None

    def compute_field(self, n_points=100, lx=-1, ux=1, ly=-1, uy=1):
        """
        n_points: Number of n_points in each dimension
        lx, ux : lower bound and upper bound of the grid in the x dimension
        ly, uy : lower bound and upper bound if the grid in the y dimension
        """

        self._grid_x = linspace(lx, ux, num=n_points)
        self._grid_y = linspace(ly, uy, num=n_points)
        self.grid = meshgrid(self._grid_x,
                             self._grid_y)
        fields = self._field(*self.grid)
        self.density = fields[0]
        self.vector_field = fields[1]
        self.curl_free = fields[2]
        self.div_free = fields[3]
        self._grid_computed = True

    def sample_field(self):
        """
        sample_size: Number of points to sample from the computed vector field
        replace: if True allow for potentially repeated examples in the sampling
        returns a dictionary containing and 2-uple containing the sampled (x, y)
        coordinates as well as the pdf and vector field along with it s
        curl-free and div-free part, evaluated at the sampled coordinates
        """

        if self._grid_computed:
            grid_array = array(self.grid)
            n_vars = grid_array.shape[0]
            n_points = grid_array.shape[1] * grid_array.shape[2]
            sampled_field = self._field(*self.grid)
            simulation = {'axes': grid_array.reshape((n_vars,
                                                      n_points)).T,
                          'density': sampled_field[0].reshape((1, n_points)).T,
                          'vector_field': sampled_field[1].reshape((n_vars,
                                                                    n_points)).T,
                          'curl_free': sampled_field[2].reshape((n_vars,
                                                                 n_points)).T,
                          'div_free': sampled_field[3].reshape((n_vars,
                                                                n_points)).T}
            return simulation

    def visualize_field(self, part='complete', mode='field', **kwargs):
        """ Visualize the computed vector field
        part: 'all' plot the sum of the curl-free and div-part,
        'curl_free' plot the curl_free part of the vector field,
        part: 'div_free' plot the div-free part of the vector field
        mode: 'field' plots the vector field,
        'contours' plots a streamplot with contours,
        '3d' plots a the density
        """
        if mode == '3d':
            raise NotImplementedError

        if self._grid_computed:

            grad = {'complete': self.vector_field,
                    'curl_free': self.curl_free,
                    'div_free': self.div_free}

            visualize = {'field': visualize_vector_field,
                         'contours': visualize_contours,
                         '3d': None}

            x, y = self.grid
            dx, dy = grad[part]
            p = self.density
            visualize[mode](x=x, y=y, dx=dx, dy=dy, p=p, title=part)




if __name__ == '__main__':


    ##,---------------------------------
    ##| Defining vector field parameters
    ##`---------------------------------
    ## A mixture of 5 uncorrelated gaussians

    means = [array((0.0, 0.0)),
             array((-0.5, 0.0)),
             array((0.0, 0.5)),
             array((0.5, 0.0)),
             array((0.0, -0.5))]
    sigmas = [array((0.25, 0.25)) for i in range(len(means))]
    rhos = [0.0] * len(means)
    ## Components with alternating sign to induce non trivial flow
    components = [((-1.0)**(i % 2))/len(means) for i in range(len(means))]

    ##,-------------------------------------
    ##| Instantiating synthetic vector field
    ##`-------------------------------------

    synthetic_field = VectorField(means, sigmas, components, rhos=rhos)

    ##,----------------
    ##| Computing grid
    ##`----------------
    synthetic_field.compute_field()

    ##,---------------------
    ##| Visualizing the grid
    ##`---------------------

    for part in ['complete', 'curl_free', 'div_free']:
        for mode in ['field', 'contours']:
            print('Visualizing the {0} vector field'.format(part))
            synthetic_field.visualize_field(part=part, mode=mode)

    ##,----------------------
    ##| Getting training data
    ##`----------------------
    sampled_dic = synthetic_field.sample_field()

