import matplotlib.pyplot as plt
import numpy as np
plt.ion()
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14

class Simulator:
    eps = 1e-16

    def __init__(self, num_agents = 15, max_iterations = 1000, step_size = None, \
                convergence_tol = 0.001, x_bounds = (0,1), y_bounds = (0, 1)):

        # convergence_tol : % of dimensions of the room
        self.convergence_tol = convergence_tol

        # Dimensions of the room
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

        self.step_size = step_size
        self.num_agents = num_agents
        self.max_iterations = max_iterations

        self.iteration = 0
        self.converged_at_iteration = None
        self.mean_step = []

        self.__initialize_positions()

        self.__choose_attractors()


    def __find_third_vertex(self, first_vertex, second_vertex):

        """ Returns both possible options for the third vertex that makes an
        equilateral triangle with two given points"""

        # Midpoint:
        mid_x, mid_y = 0.5*(first_vertex[0] + second_vertex[0]), 0.5*(first_vertex[1] + second_vertex[1])

        # Squared length of side of equilateral triangle:
        D2 = (first_vertex[0] - second_vertex[0])**2 + (first_vertex[1] - second_vertex[1])**2

        y_diff = first_vertex[1] - second_vertex[1]

        if y_diff < Simulator.eps:
            # avoid division by zero
            y_diff += Simulator.eps

        # Negative Reciprocal slope of line joining first and second vertex:
        slope = -(first_vertex[0] - second_vertex[0]) / y_diff

        # Intercept of perpendicular bisector line between first and second vertex:
        intercept = mid_y - slope * mid_x

        # For the quadratic formula:
        A = 1
        B = -2 * mid_x
        C = mid_x**2 - (3/4) * D2 /(slope**2 + 1)

        Rx = np.roots([A, B, C])
        Ry = slope*Rx + intercept

        vertex_options = (Rx, Ry)

        return vertex_options

    def __find_projections(self, target_location_x, target_location_y, current_x, current_y):
        R_vect = np.array([target_location_x - current_x, target_location_y - current_y])
        Rx_vect = np.array([target_location_x - current_x, 0])
        Ry_vect = np.array([0, target_location_y - current_y])

        # Make the distance travelled a proportion of R_vect
        x_projection = self.step_size * np.dot(Rx_vect, R_vect) / (np.linalg.norm(Rx_vect) + Simulator.eps)
        y_projection = self.step_size * np.dot(Ry_vect, R_vect) / (np.linalg.norm(Ry_vect) + Simulator.eps)

        signed_projection = np.sign(R_vect) * np.array([x_projection, y_projection])
        return (signed_projection[0], signed_projection[1])

    def __initialize_positions(self):

        # Container for the whole simulation:
        self.X = np.zeros((self.num_agents, self.max_iterations + 1))
        self.Y = np.zeros((self.num_agents, self.max_iterations + 1))

        # Initialize first positions:
        self.X[:,0] = np.random.rand(self.num_agents,)
        self.Y[:,0] = np.random.rand(self.num_agents,)

    def __choose_attractors(self):
        if self.num_agents < 3:
            raise Exception('The number of agents must be at least 3')

        # Populate the options for each agent to follow, anyone but herself
        options = np.arange(self.num_agents)
        options = np.tile(options,(len(options),1))
        options = options[~np.eye(options.shape[0],dtype=bool)].reshape(options.shape[0],-1)

        # Pick two random indices to options for two people to follow
        # (scale the random number by the range and round.)

        # Actually will need to loop here for the second agent because have to make sure not
        # choosing same two people:

        # Initialize
        follows = np.zeros((self.num_agents, 2))

        # First attractor:
        follows[:, 0, np.newaxis] = np.round( (options.shape[1] - 1) * np.random.rand(self.num_agents, 1) ).astype(int)

        # Second attractor:
        for agent in range(self.num_agents):
            firstDraw = follows[agent,0]

            # Initialize:
            secondDraw = firstDraw
            while secondDraw == firstDraw:
                # Want a different random draw from the options
                secondDraw = np.round( (options.shape[1] - 1) * np.random.rand() ).astype(int)
            follows[agent,1] = secondDraw

        follows=follows.astype(int)

        self.first_attractor = options[np.arange(options.shape[0]), follows[:,0], np.newaxis]
        self.second_attractor = options[np.arange(options.shape[0]), follows[:,1], np.newaxis]

    def _update_positions(self):
        """
        This allows each agent to jump directly to the third vertex that would create an equilateral triangle
        with the agent and the agent's two targets. However, everyone is jumping at the same time so these
        triangles are not likely to be formed until later in the simulation (if ever)
        """
        if self.step_size is not None:
            if self.step_size > 1:
                raise Exception('The step size should be less than 1')

        for agent in range(self.num_agents):

            # Find the points where you want to go to complete the triangle
            first_vertex = (self.X.item((self.first_attractor.item(agent), self.iteration)), \
               self.Y.item(self.first_attractor.item(agent), self.iteration))
            second_vertex = (self.X.item((self.second_attractor.item(agent), self.iteration)), \
               self.Y.item(self.second_attractor.item(agent), self.iteration))

            options_x, options_y = self.__find_third_vertex(first_vertex, second_vertex)

            # Find the closest of the two vertices to your current position, or the one that is inside the room:
            # For now, just don't update position if both are out of bounds

            out_of_bounds = (options_x > self.x_bounds[1]) | (options_x < self.x_bounds[0]) | \
                            (options_y > self.y_bounds[1]) | (options_y < self.y_bounds[0])

            options_x = options_x[~out_of_bounds]
            options_y = options_y[~out_of_bounds]

            current_x = self.X[agent, self.iteration]
            current_y = self.Y[agent, self.iteration]

            # Update the next position
            if len(options_x) > 1:
                # Distance to first & second options:
                D1 = ( (options_x[0] - current_x)**2 + (options_y[0] - current_y)**2 )**0.5
                D2 = ( (options_x[1] - current_x)**2 + (options_y[1] - current_y)**2 )**0.5
                closest_ind = np.argmin([D1, D2])

                if self.step_size is not None:
                    x_projection, y_projection = self.__find_projections(options_x.item(closest_ind), \
                                            options_y.item(closest_ind), current_x, current_y)
                    self.X[agent, self.iteration + 1] = current_x + x_projection
                    self.Y[agent, self.iteration + 1] = current_y + y_projection
                else:
                    self.X[agent, self.iteration + 1] = options_x[closest_ind]
                    self.Y[agent, self.iteration + 1] = options_y[closest_ind]

            elif len(options_x) == 1:
                if self.step_size is not None:
                    x_projection, y_projection = self.__find_projections(options_x.item(0), \
                                            options_y.item(0), current_x, current_y)
                    self.X[agent, self.iteration + 1] = current_x + x_projection
                    self.Y[agent, self.iteration + 1] = current_y + y_projection
                else:
                    self.X[agent, self.iteration + 1] = options_x
                    self.Y[agent, self.iteration + 1] = options_y

            else: # Don't change position
                self.X[agent, self.iteration + 1] = current_x
                self.Y[agent, self.iteration + 1] = current_y

    def plot_positions(self, initialize_plot, plot_sides = False, zoom = False):

        if initialize_plot:
            # Setting the x and y data explictly for dynamic plot update only works for plot, not scatter:
            # Going to follow the first attractor with a different color
            self.ax1.plot(self.X[0, self.iteration], self.Y[0, self.iteration], 'r.')
            self.ax1.plot(self.X[self.first_attractor.item(0), self.iteration], \
                         self.Y[self.first_attractor.item(0), self.iteration],'r+')
            self.ax1.plot(self.X[self.second_attractor.item(0), self.iteration], \
                         self.Y[self.second_attractor.item(0), self.iteration],'r+')
            self.ax1.plot(self.X[1:, self.iteration], self.Y[1:, self.iteration],'b.')

            self.ax1.set_aspect('equal')
            self.ax1.set_xlim(self.x_bounds[0], self.x_bounds[1])
            self.ax1.set_ylim(self.y_bounds[0], self.y_bounds[1])
            self.ax1.set_ylabel("Y")
            self.ax1.set_xlabel("X")
            self.ax1.set_title("Position of Agents")
        else:
            # Plot the new position
            self.ax1.set_title("Iteration = {}".format(self.iteration))
            for lin_num, line in enumerate(self.ax1.lines):
                if lin_num==0:
                    line.set_xdata(self.X[0, self.iteration])
                    line.set_ydata(self.Y[0, self.iteration])
                elif lin_num==1:
                    line.set_xdata(self.X[self.first_attractor.item(0), self.iteration - 1])
                    line.set_ydata(self.Y[self.first_attractor.item(0), self.iteration - 1])
                elif lin_num==2:
                    line.set_xdata(self.X[self.second_attractor.item(0), self.iteration - 1])
                    line.set_ydata(self.Y[self.second_attractor.item(0), self.iteration - 1])
                else:
                    line.set_xdata(self.X[1:, self.iteration])
                    line.set_ydata(self.Y[1:, self.iteration])

        self.fig.canvas.draw()
        # This is crucial for viewing the plots from the command line:
        try:
            plt.pause(0.5)
        except Exception:
            pass

        if plot_sides:
            for agent in range(self.num_agents):
                # Grab the positions for the attractors of each agent & plot the triangle in green at the end
                X_triangle = np.hstack((self.X[agent, self.iteration], \
                                        self.X[self.first_attractor.item(agent), self.iteration], \
                                        self.X[self.second_attractor.item(agent), self.iteration], \
                                        self.X[agent, self.iteration]))
                Y_triangle = np.hstack((self.Y[agent, self.iteration], \
                                        self.Y[self.first_attractor.item(agent), self.iteration], \
                                        self.Y[self.second_attractor.item(agent), self.iteration], \
                                        self.Y[agent, self.iteration]))
                self.ax1.plot(X_triangle, Y_triangle, '-g')
        if zoom:
            # Zoom In on the final positions
            self.ax1.set_xlim(0.9 * min(self.X[:, self.iteration]), 1.1 * max(self.X[:, self.iteration]))
            self.ax1.set_ylim(0.9 * min(self.Y[:, self.iteration]), 1.1 * max(self.Y[:, self.iteration]))
            self.ax1.set_aspect('equal')

    def run(self, plot_trajectories = True, plot_convergence = True):

        if plot_trajectories:
            self.fig, self.ax1 = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 8)) # two axes on figure
            self.plot_positions(initialize_plot = True)

        while self.iteration < self.max_iterations:

            # Check for convergence using mean step size for all agents:
            self.mean_step.append(np.mean( ( (self.X[:, self.iteration, np.newaxis] \
                                            - self.X[:, self.iteration - 1, np.newaxis] )**2 \
                                            + (self.Y[:, self.iteration, np.newaxis] \
                                            - self.Y[:, self.iteration - 1, np.newaxis] )**2 )**0.5 ) )

            # Define convergence as once the mean step size has dropped below the threshold for 100 iterations
            # Stop the simulation once converged.
            if self.iteration > 100: # Don't bother with convergence rules unless dealing with a significant simulation
                if all( ms <= self.convergence_tol for ms in self.mean_step[self.iteration - 100: self.iteration + 1] ):
                    self.converged_at_iteration = self.iteration
                    break

            self._update_positions()

            # Update
            self.iteration += 1

            if plot_trajectories:
                self.plot_positions(initialize_plot = False)

        if plot_convergence:
            # Plot the end positions of the agents, even if we weren't plotting
            # their trajectories throughout, along with the sides of the
            # triangles and the convergence graph
            plot_sides = True

            if not plot_trajectories:
                self.fig, self.ax1  = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 8))
                initialize = True
            else:
                initialize = False


            #if self.step_size is not None:
            #    zoom = True
            #else:
            #    zoom = False

            self.plot_positions(initialize, plot_sides)

            self.fig2, self.ax2  = plt.subplots(nrows = 1, ncols = 1, figsize=(8, 4))
            self.ax2.plot(self.mean_step)
            self.ax2.set_ylabel("Mean Step Size")
            self.ax2.set_xlabel("Iteration")
            self.ax2.set_title("Motion of Agents")


# Note if this converges before num_iterations, there will be zeros for last part of X,Y
