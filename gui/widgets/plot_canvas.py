"""Matplotlib canvas wrapper for embedding plots in tkinter."""
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class PlotCanvas(ttk.Frame):
    """A frame containing an embedded matplotlib figure with navigation toolbar."""

    def __init__(self, parent, figsize=(8, 6), dpi=100, **kwargs):
        """
        Create a plot canvas.

        Args:
            parent: Parent widget
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch
        """
        super().__init__(parent, **kwargs)

        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)

        # Toolbar at top
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(side='top', fill='x')
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # Canvas fills remaining space
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    def clear(self):
        """Clear all axes from the figure."""
        self.figure.clear()
        self.canvas.draw()

    def draw(self):
        """Redraw the canvas."""
        self.canvas.draw()

    def get_figure(self) -> Figure:
        """Get the matplotlib figure object."""
        return self.figure

    def update_from_figure(self, new_figure: Figure):
        """
        Update display from an external figure.
        Copies axes from new_figure to internal figure.
        """
        self.figure.clear()

        # Copy axes from new figure
        for ax in new_figure.axes:
            # Create new axes in our figure with same position
            new_ax = self.figure.add_subplot(ax.get_geometry()[0],
                                              ax.get_geometry()[1],
                                              ax.get_geometry()[2])

            # Copy content (simplified - for complex plots, use the figure directly)
            new_ax.set_title(ax.get_title())
            new_ax.set_xlabel(ax.get_xlabel())
            new_ax.set_ylabel(ax.get_ylabel())

        self.figure.tight_layout()
        self.canvas.draw()


class MultiPlotCanvas(ttk.Frame):
    """A frame containing multiple subplot panels."""

    def __init__(self, parent, nrows=1, ncols=1, figsize=(10, 8), dpi=100, **kwargs):
        """
        Create a multi-plot canvas.

        Args:
            parent: Parent widget
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            figsize: Figure size in inches
            dpi: Dots per inch
        """
        super().__init__(parent, **kwargs)

        self.nrows = nrows
        self.ncols = ncols
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.axes = []

        # Create subplots
        for i in range(nrows * ncols):
            ax = self.figure.add_subplot(nrows, ncols, i + 1)
            self.axes.append(ax)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)

        # Toolbar
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(side='top', fill='x')
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # Canvas
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    def clear(self):
        """Clear all subplots."""
        for ax in self.axes:
            ax.clear()
        self.canvas.draw()

    def draw(self):
        """Redraw the canvas."""
        self.figure.tight_layout()
        self.canvas.draw()

    def get_axes(self, index: int = 0):
        """Get a specific axes object."""
        return self.axes[index] if index < len(self.axes) else None
