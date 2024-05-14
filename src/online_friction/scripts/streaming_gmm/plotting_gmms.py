import numpy as np
np.float = np.float64 
np.int = np.int32
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import os
from sklearn.mixture import GaussianMixture

def visualize_3d_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 3D
    Input: 
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''

    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection='3d')
    axes.set_xlim([0, 255])
    axes.set_ylim([0, 255])
    axes.set_zlim([0, 255])
    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=colors[i])
        plot_sphere(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)

    plt.title('3D GMM')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.view_init(35.246, 45)
    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/3D_GMM_demonstration.png', dpi=100, format='png')
    plt.show()


def plot_sphere(w=0, c=[0,0,0], r=[1, 1, 1], subdev=10, ax=None, sigma_multiplier=3):
    '''
        plot a sphere surface
        Input: 
            c: 3 elements list, sphere center
            r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0,subdev), 0.0:2.0 * pi:complex(0,subdev)]
    x = sigma_multiplier*r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier*r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier*r[2] * cos(phi) + c[2]
    cmap = cmx.ScalarMappable()
    cmap.set_cmap('jet')
    c = cmap.to_rgba(w)

    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)

    return ax

def visualize_2D_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 2D
    Input: 
        points: N X 2, sampled points
        w: n_gaussians, gmm weights
        mu: 2 X n_gaussians, gmm means
        stdev: 2 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''
    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = plt.gca()
    axes.set_xlim([-1, 1])
    axes.set_ylim([-1, 1])
    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        plt.scatter(points[idx, 0], points[idx, 1], alpha=0.3, c=colors[i])
        for j in range(8):
            axes.add_patch(
                patches.Ellipse(mu[:, i], width=(j+1) * stdev[0, i], height=(j+1) *  stdev[1, i], fill=False, color=[0.0, 0.0, 1.0, 1.0/(0.5*j+1)]))
        plt.title('GMM')
    plt.xlabel('X')
    plt.ylabel('Y')

    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/2D_GMM_demonstration.png', dpi=100, format='png')

    plt.show()

def test():
    ## Generate synthetic data
    N,D = 1000, 3 # number of points and dimenstinality

    if D == 2:
        #set gaussian ceters and covariances in 2D
        means = np.array([[0.5, 0.0],
                        [0, 0],
                        [-0.5, -0.5],
                        [-0.8, 0.3]])
        covs = np.array([np.diag([0.01, 0.01]),
                        np.diag([0.025, 0.01]),
                        np.diag([0.01, 0.025]),
                        np.diag([0.01, 0.01])])
    elif D == 3:
        # set gaussian ceters and covariances in 3D
        means = np.array([[0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [-0.5, -0.5, -0.5],
                        [-0.8, 0.3, 0.4]])
        covs = np.array([np.diag([0.01, 0.01, 0.03]),
                        np.diag([0.08, 0.01, 0.01]),
                        np.diag([0.01, 0.05, 0.01]),
                        np.diag([0.03, 0.07, 0.01])])
    n_gaussians = means.shape[0]

    points = []
    for i in range(len(means)):
        x = np.random.multivariate_normal(means[i], covs[i], N )
        points.append(x)
    points = np.concatenate(points)
    #fit the gaussian model
    gmm = GaussianMixture(n_components=n_gaussians, covariance_type='full') #diag to work directly
    gmm.fit(points)
    covariances_diag = np.asarray([np.diag(cov) for cov in (gmm.covariances_)])
    # print("n_gaussians: ",  gmm.covariances_)

    #visualize
    if D == 2:
        visualize_2D_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)
    elif D == 3:
        visualize_3d_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(covariances_diag).T)


import pickle
with open('/online_friction/res.data', 'rb') as f:
    new_data = pickle.load(f)
    covariances_diag = np.asarray([np.diag(cov) for cov in (new_data[3])])
    visualize_3d_gmm(new_data[0], new_data[2], new_data[1].T, np.sqrt(covariances_diag).T) #data = [self.supervoxels_rgb_mean,m_k,self.gmm_weights,self.sigma]
    