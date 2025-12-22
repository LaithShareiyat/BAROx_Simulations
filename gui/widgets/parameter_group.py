"""Reusable parameter input group widget."""
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, Optional, Callable


class ParameterGroup(ttk.LabelFrame):
    """A labeled frame containing multiple parameter input fields."""

    def __init__(self, parent, title: str, parameters: list, **kwargs):
        """
        Create a parameter group.

        Args:
            parent: Parent widget
            title: Group title
            parameters: List of dicts with keys:
                - name: Parameter identifier
                - label: Display label
                - default: Default value
                - type: 'float', 'int', or 'bool'
                - tooltip: Optional tooltip text
        """
        super().__init__(parent, text=title, padding=(10, 5), **kwargs)

        self.entries: Dict[str, tk.Variable] = {}
        self.entry_widgets: Dict[str, ttk.Entry] = {}
        self._create_fields(parameters)

    def _create_fields(self, parameters: list):
        """Create input fields for all parameters."""
        for i, param in enumerate(parameters):
            name = param['name']
            label = param['label']
            default = param['default']
            param_type = param.get('type', 'float')

            # Create label
            lbl = ttk.Label(self, text=label, width=18, anchor='w')
            lbl.grid(row=i, column=0, sticky='w', padx=(5, 10), pady=2)

            # Create variable and entry
            if param_type == 'bool':
                var = tk.BooleanVar(value=default)
                widget = ttk.Checkbutton(self, variable=var)
            else:
                var = tk.StringVar(value=str(default))
                widget = ttk.Entry(self, textvariable=var, width=12)
                self.entry_widgets[name] = widget

                # Add validation
                widget.bind('<FocusOut>', lambda e, n=name, t=param_type: self._validate(n, t))

            widget.grid(row=i, column=1, sticky='w', padx=5, pady=2)
            self.entries[name] = var

            # Add unit label if present
            if 'unit' in param:
                unit_lbl = ttk.Label(self, text=param['unit'], foreground='gray')
                unit_lbl.grid(row=i, column=2, sticky='w', padx=(0, 5), pady=2)

    def _validate(self, name: str, param_type: str) -> bool:
        """Validate a parameter value."""
        try:
            value = self.entries[name].get()
            if param_type == 'float':
                float(value)
            elif param_type == 'int':
                int(value)

            # Reset style on valid
            if name in self.entry_widgets:
                self.entry_widgets[name].configure(style='TEntry')
            return True
        except ValueError:
            # Highlight invalid entry
            if name in self.entry_widgets:
                self.entry_widgets[name].configure(style='Invalid.TEntry')
            return False

    def get_values(self) -> Dict[str, Any]:
        """Get all parameter values as a dictionary."""
        values = {}
        for name, var in self.entries.items():
            val = var.get()
            # Try to convert to number
            try:
                if '.' in str(val):
                    values[name] = float(val)
                else:
                    values[name] = int(val)
            except (ValueError, TypeError):
                if isinstance(val, bool):
                    values[name] = val
                else:
                    values[name] = val
        return values

    def set_values(self, values: Dict[str, Any]):
        """Set parameter values from a dictionary."""
        for name, value in values.items():
            if name in self.entries:
                self.entries[name].set(str(value) if not isinstance(value, bool) else value)

    def set_state(self, state: str):
        """Enable or disable all entry widgets."""
        for widget in self.entry_widgets.values():
            widget.configure(state=state)


class CollapsibleParameterGroup(ttk.Frame):
    """A parameter group that can be collapsed/expanded."""

    def __init__(self, parent, title: str, parameters: list, initially_collapsed: bool = False, **kwargs):
        super().__init__(parent, **kwargs)

        self._collapsed = initially_collapsed

        # Header frame with toggle button
        self.header = ttk.Frame(self)
        self.header.pack(fill='x')

        self.toggle_btn = ttk.Button(
            self.header,
            text=f"{'>' if initially_collapsed else 'v'} {title}",
            command=self._toggle,
            width=25
        )
        self.toggle_btn.pack(anchor='w')

        # Content frame
        self.content = ParameterGroup(self, title="", parameters=parameters)
        if not initially_collapsed:
            self.content.pack(fill='x', padx=(10, 0))

        self.entries = self.content.entries
        self.entry_widgets = self.content.entry_widgets

    def _toggle(self):
        """Toggle collapsed state."""
        self._collapsed = not self._collapsed
        title = self.toggle_btn.cget('text')[2:]  # Remove prefix

        if self._collapsed:
            self.content.pack_forget()
            self.toggle_btn.configure(text=f"> {title}")
        else:
            self.content.pack(fill='x', padx=(10, 0))
            self.toggle_btn.configure(text=f"v {title}")

    def get_values(self) -> Dict[str, Any]:
        return self.content.get_values()

    def set_values(self, values: Dict[str, Any]):
        self.content.set_values(values)

    def set_state(self, state: str):
        self.content.set_state(state)
