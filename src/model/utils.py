import matplotlib.pyplot as plt

class TrainMonitor():

    """
    a class for monitoring training procedure
    """

    def __init__(self, xlim=(0, 60000), ylim=(0, 1), color="red", fig_size=(7, 4)):
        self.xlim = xlim
        self.ylim = ylim
        self.color = color
        self.fig_size = fig_size

    def start(self):
        # initiate x and y axis for loss
        self.x = []
        self.y = []
        # initiate x axis for metrics
        self.fig = plt.figure(figsize=self.fig_size)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_ylim(self.ylim[0], self.ylim[1])
        self.ax.set_xlim(self.xlim[0], self.xlim[1])
        self.fig.canvas.draw()
        self.line, = self.ax.plot(self.x, self.y, '.', color=self.color)

    def update(self, x_new, y_new):
        self.y.append(y_new)
        self.x.append(x_new)
        self.line.set_xdata(self.x)
        self.line.set_ydata(self.y)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.0000000000001)