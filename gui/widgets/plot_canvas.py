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

        self._selected_axes_idx = None

        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)

        # Track clicks for subplot selection
        self.canvas.mpl_connect('button_press_event', self._on_click)

        # Toolbar at top
        self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(side='top', fill='x')
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # Pop-out button alongside the toolbar
        self.popout_btn = ttk.Button(
            self.toolbar_frame, text="Pop Out", command=self._pop_out, width=8
        )
        self.popout_btn.pack(side='right', padx=5)

        # Canvas fills remaining space
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)

    def _on_click(self, event):
        """Track which axes was clicked for individual subplot pop-out."""
        if event.inaxes is not None:
            try:
                self._selected_axes_idx = self.figure.axes.index(event.inaxes)
            except ValueError:
                self._selected_axes_idx = None
        else:
            self._selected_axes_idx = None

    def _pop_out(self):
        """Open the selected subplot (or full figure) in a separate window.

        Click on a subplot first to select it, then click Pop Out to open
        just that subplot. If nothing is selected, pops out the whole figure.
        Preserves twin axes (e.g. RPM secondary axis) and colorbars.
        """
        import pickle

        # Clone the figure via pickle so the original stays embedded
        try:
            fig_copy = pickle.loads(pickle.dumps(self.figure))
        except Exception:
            return

        # If a specific subplot was clicked and figure has multiple axes,
        # extract just that subplot (plus its twins and colorbars)
        if self._selected_axes_idx is not None and len(self.figure.axes) > 1:
            selected_pos = self.figure.axes[self._selected_axes_idx].get_position()

            # Identify axes to keep: same position = twin axes, plus colorbars
            keep = set()
            for i, ax in enumerate(fig_copy.axes):
                pos = ax.get_position()
                # Same position (within tolerance) means twin axes
                if (abs(pos.x0 - selected_pos.x0) < 0.01
                        and abs(pos.y0 - selected_pos.y0) < 0.01
                        and abs(pos.width - selected_pos.width) < 0.01
                        and abs(pos.height - selected_pos.height) < 0.01):
                    keep.add(i)
                    # Find colorbars attached to artists on this axes
                    for artist in list(ax.images) + list(ax.collections):
                        if hasattr(artist, 'colorbar') and artist.colorbar is not None:
                            try:
                                cbar_idx = fig_copy.axes.index(artist.colorbar.ax)
                                keep.add(cbar_idx)
                            except ValueError:
                                pass

            if keep:
                to_remove = [ax for i, ax in enumerate(fig_copy.axes) if i not in keep]
                for ax in to_remove:
                    fig_copy.delaxes(ax)
                fig_copy.set_size_inches(8, 6)

                # Rescale remaining axes to fill the pop-out window.
                # Separate primary/twin axes from colorbars by relative area.
                remaining = list(fig_copy.axes)
                areas = {
                    id(ax): ax.get_position().width * ax.get_position().height
                    for ax in remaining
                }
                max_area = max(areas.values()) if areas else 0

                primary = [ax for ax in remaining if areas[id(ax)] > max_area * 0.25]
                colorbars = [ax for ax in remaining if areas[id(ax)] <= max_area * 0.25]

                has_cbar = len(colorbars) > 0
                main_rect = [0.10, 0.12, 0.72 if has_cbar else 0.85, 0.82]

                for ax in primary:
                    ax.set_position(main_rect)
                for cbar_ax in colorbars:
                    cbar_ax.set_position([
                        main_rect[0] + main_rect[2] + 0.02,
                        main_rect[1],
                        0.03,
                        main_rect[3],
                    ])

        window = tk.Toplevel()
        window.title("BAROx - Plot")
        window.geometry("1000x700")

        toolbar_frame = ttk.Frame(window)
        toolbar_frame.pack(side='top', fill='x')

        canvas = FigureCanvasTkAgg(fig_copy, master=window)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        canvas.draw()

    def clear(self):
        """Clear all axes from the figure."""
        self._selected_axes_idx = None
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
