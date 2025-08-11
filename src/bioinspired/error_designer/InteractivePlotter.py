"""Interactive Error Analysis Plotter with Window Splitter
Provides a sophisticated GUI interface for error analysis results using the generalized WindowSplitter framework.
Features:
- Blender-style window splitting for multiple plots
- Dynamic pane types for different plot types
- Real-time plot updates based on selections
- Professional dark theme interface
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Any, TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Import the WindowSplitter framework
from .WindowSplitter import BasePaneType, PaneContentProvider, WindowSplitterManager

if TYPE_CHECKING:
    from .ErrorDesignerBase import ErrorAnalysisResult


class PlotPaneType(BasePaneType):
    """Pane types for different plot types in error analysis."""
    
    def __init__(self, identifier: str, value: str, description: str = ""):
        self._identifier = identifier
        self._value = value
        self._description = description
    
    @property
    def value(self) -> str:
        return self._value
    
    @property
    def identifier(self) -> str:
        return self._identifier
    
    @property
    def description(self) -> str:
        return self._description


class PlotPaneTypes:
    """Available pane types for error analysis plots."""
    EMPTY = PlotPaneType("empty", "Empty", "No plot selected")
    LINE_PLOT = PlotPaneType("line_plot", "Line Plot", "Error vs Parameter (quantitative)")
    CATEGORICAL_PLOT = PlotPaneType("categorical_plot", "Categorical Plot", "Error by Category")
    PERFORMANCE_PLOT = PlotPaneType("performance_plot", "Performance Plot", "Runtime vs Parameter")
    ERROR_TIME_SERIES = PlotPaneType("error_time_series", "Error Time Series", "Error evolution over time")
    ERROR_COMPARISON = PlotPaneType("error_comparison", "Error Comparison", "Compare multiple error types")
    HEATMAP = PlotPaneType("heatmap", "Heatmap", "Parameter vs Time heatmap")
    PARAMETER_SPACE = PlotPaneType("parameter_space", "Parameter Space", "Multi-dimensional parameter view")
    STATISTICS_TABLE = PlotPaneType("statistics_table", "Statistics", "Error statistics summary")
    CONTROLS_PANEL = PlotPaneType("controls_panel", "Controls", "Plot controls and options")


class ErrorAnalysisContentProvider(PaneContentProvider):
    """Content provider for error analysis plots and controls."""
    
    def __init__(self, result: 'ErrorAnalysisResult'):
        """Initialize with error analysis result.
        
        Args:
            result: ErrorAnalysisResult from parameter sweep
        """
        self.result = result
        self.parameter_type = self._analyze_parameter_type()
        self.error_summaries = self._compute_error_summaries()
        self.error_types = list(self.result.error_metrics.keys())
        self.parameter_values = list(self.result.parameter_values)
        
        # Store pane-specific data
        self.pane_data = {}  # Will store selections and state for each pane
        self.plot_frames = {}  # Store references to plot frames for updates
        
    def _analyze_parameter_type(self) -> str:
        """Analyze the type of parameter being swept."""
        param_values = self.result.parameter_values
        
        if any(isinstance(v, str) for v in param_values):
            return "categorical"
        
        try:
            _ = [float(v) for v in param_values]
            return "quantitative"
        except (ValueError, TypeError):
            return "categorical"
    
    def _compute_error_summaries(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute error summaries from time-indexed error data."""
        summaries = {
            "rms_errors": {},
            "max_errors": {},
            "final_errors": {},
            "mean_errors": {},
        }
        
        for error_type in self.result.error_metrics.keys():
            for summary_type in summaries.keys():
                summaries[summary_type][error_type] = {}
            
            # Iterate through parameter combinations using combo keys
            for i, param_combination in enumerate(self.result.parameter_combinations):
                combo_key = f"combo_{i}"
                
                if combo_key in self.result.error_metrics[error_type]:
                    time_series = self.result.error_metrics[error_type][combo_key]
                    if time_series:
                        values = list(time_series.values())
                        
                        if values:
                            summaries["rms_errors"][error_type][combo_key] = np.sqrt(np.mean(np.array(values)**2))
                            summaries["max_errors"][error_type][combo_key] = max(values)
                            summaries["final_errors"][error_type][combo_key] = values[-1]
                            summaries["mean_errors"][error_type][combo_key] = np.mean(values)
        
        return summaries
    
    def get_available_pane_types(self) -> List[BasePaneType]:
        """Get all available pane types based on data structure."""
        pane_types = [
            PlotPaneTypes.EMPTY,
            PlotPaneTypes.PERFORMANCE_PLOT,
            PlotPaneTypes.ERROR_TIME_SERIES,
            PlotPaneTypes.ERROR_COMPARISON,
            PlotPaneTypes.STATISTICS_TABLE,
            PlotPaneTypes.CONTROLS_PANEL,
        ]
        
        if self.parameter_type == "categorical":
            pane_types.append(PlotPaneTypes.CATEGORICAL_PLOT)
        elif self.parameter_type == "quantitative":
            pane_types.extend([
                PlotPaneTypes.LINE_PLOT,
                PlotPaneTypes.HEATMAP,
                PlotPaneTypes.PARAMETER_SPACE,
            ])
        
        return pane_types
    
    def create_content(self, parent_frame: tk.Frame, pane_type: BasePaneType) -> None:
        """Create content for the given pane type."""
        # Create a unique ID for this pane to store its state
        pane_id = f"{pane_type.identifier}_{id(parent_frame)}"
        
        if pane_id not in self.pane_data:
            # Get available parameters from metadata
            available_params = []
            if 'all_parameters' in self.result.metadata:
                available_params = [p['name'] for p in self.result.metadata['all_parameters']]
            
            self.pane_data[pane_id] = {
                'error_type': self.error_types[0] if self.error_types else "",
                'summary_type': "rms_errors",
                'time_range': None,
                'selected_params': None,
                'x_axis_param': available_params[0] if available_params else None,
                'available_params': available_params,
            }
        
        if pane_type == PlotPaneTypes.EMPTY:
            self._create_empty_content(parent_frame)
        elif pane_type == PlotPaneTypes.LINE_PLOT:
            self._create_line_plot_content(parent_frame, pane_id)
        elif pane_type == PlotPaneTypes.CATEGORICAL_PLOT:
            self._create_categorical_plot_content(parent_frame, pane_id)
        elif pane_type == PlotPaneTypes.PERFORMANCE_PLOT:
            self._create_performance_plot_content(parent_frame, pane_id)
        elif pane_type == PlotPaneTypes.ERROR_TIME_SERIES:
            self._create_time_series_content(parent_frame, pane_id)
        elif pane_type == PlotPaneTypes.ERROR_COMPARISON:
            self._create_error_comparison_content(parent_frame, pane_id)
        elif pane_type == PlotPaneTypes.HEATMAP:
            self._create_heatmap_content(parent_frame, pane_id)
        elif pane_type == PlotPaneTypes.PARAMETER_SPACE:
            self._create_parameter_space_content(parent_frame, pane_id)
        elif pane_type == PlotPaneTypes.STATISTICS_TABLE:
            self._create_statistics_table_content(parent_frame, pane_id)
        elif pane_type == PlotPaneTypes.CONTROLS_PANEL:
            self._create_controls_panel_content(parent_frame, pane_id)
        else:
            self._create_empty_content(parent_frame)
    
    def _create_empty_content(self, parent_frame: tk.Frame):
        """Create content for empty pane."""
        label = tk.Label(
            parent_frame,
            text="Empty Pane\n\n📊\n\nSelect a plot type from the dropdown\nto display error analysis data",
            bg='#3c3c3c',
            fg='gray',
            font=('Arial', 12),
            justify=tk.CENTER
        )
        label.pack(expand=True)
    
    def _create_plot_with_controls(self, parent_frame: tk.Frame, pane_id: str, 
                                   plot_creator, show_error_selector=True, 
                                   show_summary_selector=True, additional_controls=None,
                                   custom_labels=None):
        """Create a plot with standard control elements."""
        # Controls frame at the top
        controls_frame = tk.Frame(parent_frame, bg='#404040', height=40)
        controls_frame.pack(fill=tk.X, side=tk.TOP)
        controls_frame.pack_propagate(False)
        
        current_col = 0
        
        # Error type selector
        if show_error_selector and self.error_types:
            tk.Label(controls_frame, text="Error Type:", bg='#404040', fg='white', 
                    font=('Arial', 8)).grid(row=0, column=current_col, sticky='w', padx=(5, 2))
            current_col += 1
            
            error_var = tk.StringVar(value=self.pane_data[pane_id]['error_type'])
            error_combo = ttk.Combobox(controls_frame, textvariable=error_var, 
                                     values=self.error_types, width=12, state='readonly',
                                     font=('Arial', 8))
            error_combo.grid(row=0, column=current_col, padx=(0, 10), sticky='w')
            error_combo.bind('<<ComboboxSelected>>', 
                           lambda e: self._on_error_type_change(pane_id, error_var.get(), plot_creator))
            current_col += 1
        
        # Summary type selector
        if show_summary_selector:
            tk.Label(controls_frame, text="Summary:", bg='#404040', fg='white',
                    font=('Arial', 8)).grid(row=0, column=current_col, sticky='w', padx=(5, 2))
            current_col += 1
            
            summary_var = tk.StringVar(value=self.pane_data[pane_id]['summary_type'])
            summary_combo = ttk.Combobox(controls_frame, textvariable=summary_var,
                                       values=["rms_errors", "max_errors", "final_errors", "mean_errors"],
                                       width=10, state='readonly', font=('Arial', 8))
            summary_combo.grid(row=0, column=current_col, padx=(0, 10), sticky='w')
            summary_combo.bind('<<ComboboxSelected>>',
                             lambda e: self._on_summary_type_change(pane_id, summary_var.get(), plot_creator))
            current_col += 1
        
        # Custom labels and additional controls
        if custom_labels and additional_controls:
            for (label_text, font_size), control in zip(custom_labels, additional_controls):
                tk.Label(controls_frame, text=label_text, bg='#404040', fg='white',
                        font=('Arial', font_size)).grid(row=0, column=current_col, sticky='w', padx=(5, 2))
                current_col += 1
                
                # Handle special control types
                if control.get('type') == 'x_param_selector':
                    pane_id_ctrl = control['pane_id']
                    plot_creator_ctrl = control['plot_creator']
                    
                    x_param_var = tk.StringVar(value=self.pane_data[pane_id_ctrl]['x_axis_param'] or "")
                    x_param_combo = ttk.Combobox(
                        controls_frame, textvariable=x_param_var,
                        values=self.pane_data[pane_id_ctrl]['available_params'],
                        width=12, state='readonly', font=('Arial', 8)
                    )
                    
                    # Create callback with proper closure
                    def make_x_param_callback(pid, var, pc):
                        return lambda e: self._on_x_param_change(pid, var.get(), pc)
                    
                    x_param_combo.bind('<<ComboboxSelected>>', 
                                     make_x_param_callback(pane_id_ctrl, x_param_var, plot_creator_ctrl))
                    x_param_combo.grid(row=0, column=current_col, padx=(0, 10), sticky='w')
                elif control.get('type') == 'param_selector_button':
                    pane_id_ctrl = control['pane_id']
                    plot_creator_ctrl = control['plot_creator']
                    
                    select_button = tk.Button(
                        controls_frame, text="Select Parameters", 
                        command=control['command'],
                        bg='#555555', fg='white', 
                        font=('Arial', 8), relief='raised', bd=1
                    )
                    select_button.grid(row=0, column=current_col, padx=(0, 10), sticky='w')
                else:
                    # Standard control with widget
                    control['widget'].grid(row=0, column=current_col, **control.get('grid_kwargs', {}))
                current_col += 1
        elif additional_controls:
            # Legacy support - no custom labels
            for control in additional_controls:
                if control.get('type') == 'x_param_selector':
                    # Handle special types even without custom labels
                    pane_id_ctrl = control['pane_id'] 
                    plot_creator_ctrl = control['plot_creator']
                    
                    x_param_var = tk.StringVar(value=self.pane_data[pane_id_ctrl]['x_axis_param'] or "")
                    x_param_combo = ttk.Combobox(
                        controls_frame, textvariable=x_param_var,
                        values=self.pane_data[pane_id_ctrl]['available_params'],
                        width=12, state='readonly', font=('Arial', 8)
                    )
                    
                    # Create callback with proper closure
                    def make_x_param_callback_legacy(pid, var, pc):
                        return lambda e: self._on_x_param_change(pid, var.get(), pc)
                    
                    x_param_combo.bind('<<ComboboxSelected>>', 
                                     make_x_param_callback_legacy(pane_id_ctrl, x_param_var, plot_creator_ctrl))
                    x_param_combo.grid(row=0, column=current_col, padx=(0, 10), sticky='w')
                elif control.get('type') == 'param_selector_button':
                    pane_id_ctrl = control['pane_id']
                    plot_creator_ctrl = control['plot_creator']
                    
                    select_button = tk.Button(
                        controls_frame, text="Select Parameters", 
                        command=control['command'],
                        bg='#555555', fg='white', 
                        font=('Arial', 8), relief='raised', bd=1
                    )
                    select_button.grid(row=0, column=current_col, padx=(0, 10), sticky='w')
                else:
                    control['widget'].grid(row=0, column=current_col, **control.get('grid_kwargs', {}))
                current_col += 1
        
        # Plot frame
        plot_frame = tk.Frame(parent_frame, bg='#3c3c3c')
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Store references for plot updates
        self.plot_frames[pane_id] = {
            'frame': plot_frame,
            'creator': plot_creator
        }
        
        # Create the plot
        plot_creator(plot_frame, pane_id)
    
    def _on_error_type_change(self, pane_id: str, new_error_type: str, plot_creator):
        """Handle error type change."""
        self.pane_data[pane_id]['error_type'] = new_error_type
        self._update_plot(pane_id)
    
    def _on_summary_type_change(self, pane_id: str, new_summary_type: str, plot_creator):
        """Handle summary type change."""
        self.pane_data[pane_id]['summary_type'] = new_summary_type
        self._update_plot(pane_id)
    
    def _update_plot(self, pane_id: str):
        """Update the plot for a specific pane."""
        if pane_id in self.plot_frames:
            plot_frame = self.plot_frames[pane_id]['frame']
            plot_creator = self.plot_frames[pane_id]['creator']
            
            # Clear the current plot
            for widget in plot_frame.winfo_children():
                widget.destroy()
            
            # Recreate the plot
            plot_creator(plot_frame, pane_id)
    
    def _create_line_plot_content(self, parent_frame: tk.Frame, pane_id: str):
        """Create line plot content with flexible parameter and error selection."""
        def create_plot(plot_frame, pane_id):
            fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
            ax = fig.add_subplot(111, facecolor='#3c3c3c')
            
            error_type = self.pane_data[pane_id]['error_type']
            summary_type = self.pane_data[pane_id]['summary_type']
            x_axis_param = self.pane_data[pane_id]['x_axis_param']
            
            if not error_type or not x_axis_param:
                ax.text(0.5, 0.5, 'Select Error Type and X-Axis Parameter', ha='center', va='center',
                       transform=ax.transAxes, color='white', fontsize=12)
            elif error_type not in self.error_summaries[summary_type]:
                ax.text(0.5, 0.5, 'No Data Available for Selected Error Type', ha='center', va='center',
                       transform=ax.transAxes, color='white', fontsize=12)
            else:
                # Group combinations by their parameter values (excluding x-axis parameter)
                grouped_data = self._group_combinations_by_parameters(x_axis_param, error_type, summary_type)
                
                if not grouped_data:
                    ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                           transform=ax.transAxes, color='white', fontsize=12)
                else:
                    # Plot each group with different colors/styles
                    colors = ['cyan', 'orange', 'lime', 'magenta', 'yellow', 'red', 'lightblue', 'lightgreen']
                    linestyles = ['-', '--', '-.', ':']
                    
                    for i, (group_key, group_data) in enumerate(grouped_data.items()):
                        color = colors[i % len(colors)]
                        linestyle = linestyles[(i // len(colors)) % len(linestyles)]
                        
                        x_values = group_data['x_values']
                        y_values = group_data['y_values']
                        
                        if x_values and y_values:
                            # Sort by x-values for proper line plotting
                            sorted_pairs = sorted(zip(x_values, y_values))
                            x_sorted, y_sorted = zip(*sorted_pairs)
                            
                            ax.loglog(x_sorted, y_sorted, 'o-', linewidth=2, markersize=6, 
                                    color=color, linestyle=linestyle, label=group_key, alpha=0.8)
                    
                    # Set axis labels and title
                    x_param_info = self._get_parameter_info(x_axis_param)
                    param_name = x_param_info.get('display_name', x_axis_param)
                    param_units = x_param_info.get('units', '')
                    
                    ax.set_xlabel(f"{param_name} {param_units}".strip(), color='white')
                    ax.set_ylabel(f"{summary_type.replace('_', ' ').title()} {error_type.replace('_', ' ').title()}", 
                                color='white')
                    ax.set_title(f"{error_type.replace('_', ' ').title()} vs {param_name}", color='white')
                    ax.grid(True, alpha=0.3, color='gray')
                    ax.tick_params(colors='white')
                    
                    # Add legend if multiple groups
                    if len(grouped_data) > 1:
                        ax.legend(loc='best', fontsize=8, facecolor='#404040', 
                                edgecolor='white', labelcolor='white', framealpha=0.9)
            
            # Style the plot
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            
            canvas = FigureCanvasTkAgg(fig, plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
        
        # Create controls with x-axis parameter selector
        additional_controls = []
        custom_labels = []
        
        if self.pane_data[pane_id]['available_params']:
            custom_labels.append(("X-Axis:", 8))
            
            # Note: The actual widget creation will be done in _create_plot_with_controls
            # We'll pass the creation parameters instead
            additional_controls.append({
                'type': 'x_param_selector',
                'pane_id': pane_id,
                'plot_creator': create_plot,
            })
        
        # Create the plot with enhanced controls
        self._create_plot_with_controls(parent_frame, pane_id, create_plot, 
                                      additional_controls=additional_controls,
                                      custom_labels=custom_labels if additional_controls else None)
    
    def _group_combinations_by_parameters(self, x_axis_param: str, error_type: str, summary_type: str) -> Dict[str, Dict]:
        """Group parameter combinations by non-x-axis parameters for line differentiation."""
        grouped_data = {}
        
        for i, param_combo in enumerate(self.result.parameter_combinations):
            combo_key = f"combo_{i}"
            
            if combo_key not in self.error_summaries[summary_type][error_type]:
                continue
                
            # Get x-axis value
            x_value = param_combo.get(x_axis_param)
            if x_value is None:
                continue
                
            # Create group key from other parameters (excluding x-axis parameter)
            other_params = {k: v for k, v in param_combo.items() if k != x_axis_param}
            group_key = self._format_parameter_group(other_params)
            
            # Initialize group if not exists
            if group_key not in grouped_data:
                grouped_data[group_key] = {'x_values': [], 'y_values': []}
            
            # Add data point
            y_value = self.error_summaries[summary_type][error_type][combo_key]
            grouped_data[group_key]['x_values'].append(x_value)
            grouped_data[group_key]['y_values'].append(y_value)
        
        return grouped_data
    
    def _format_parameter_group(self, params: Dict[str, Any]) -> str:
        """Format parameter combination for group labeling."""
        if not params:
            return "Base Case"
        
        param_strings = []
        for key, value in sorted(params.items()):
            # Get display name for parameter
            param_info = self._get_parameter_info(key)
            display_name = param_info.get('display_name', key)
            
            if hasattr(value, "value"):  # Enum
                param_strings.append(f"{display_name}={value.value}")
            else:
                param_strings.append(f"{display_name}={value}")
        
        return ", ".join(param_strings)
    
    def _group_performance_by_parameters(self, x_axis_param: str) -> Dict[str, Dict]:
        """Group performance data (computation times) by non-x-axis parameters."""
        grouped_data = {}
        
        for i, param_combo in enumerate(self.result.parameter_combinations):
            # Get computation time for this combination
            if i < len(self.result.computation_times):
                computation_time = self.result.computation_times[i]
            else:
                continue  # Skip if no computation time available
                
            # Get x-axis value
            x_value = param_combo.get(x_axis_param)
            if x_value is None:
                continue
                
            # Create group key from other parameters (excluding x-axis parameter)
            other_params = {k: v for k, v in param_combo.items() if k != x_axis_param}
            group_key = self._format_parameter_group(other_params)
            
            # Initialize group if not exists
            if group_key not in grouped_data:
                grouped_data[group_key] = {'x_values': [], 'y_values': []}
            
            # Add data point
            grouped_data[group_key]['x_values'].append(x_value)
            grouped_data[group_key]['y_values'].append(computation_time)
        
        return grouped_data
    
    def _group_time_series_by_parameters(self, error_type: str) -> Dict[str, List[Dict]]:
        """Group time series data by parameter combinations."""
        grouped_data = {}
        
        for i, param_combo in enumerate(self.result.parameter_combinations):
            combo_key = f"combo_{i}"
            
            if combo_key not in self.result.error_metrics[error_type]:
                continue
            
            time_series = self.result.error_metrics[error_type][combo_key]
            if not time_series:
                continue
            
            # Create group key from all parameters
            group_key = self._format_parameter_group(param_combo)
            
            # Initialize group if not exists
            if group_key not in grouped_data:
                grouped_data[group_key] = []
            
            # Add time series to group
            grouped_data[group_key].append(time_series)
        
        return grouped_data
    
    def _show_parameter_selection_dialog(self, pane_id: str, plot_creator):
        """Show dialog for selecting which parameter combinations to plot."""
        dialog = tk.Toplevel()
        dialog.title("Select Parameter Combinations")
        dialog.geometry("600x500")
        dialog.configure(bg='#2b2b2b')
        dialog.resizable(True, True)
        
        # Make dialog modal
        dialog.transient(self.plot_frames[pane_id]['frame'].winfo_toplevel())
        dialog.grab_set()
        
        # Title
        title_label = tk.Label(
            dialog,
            text="Select Parameter Combinations to Plot",
            bg='#2b2b2b',
            fg='white',
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=10)
        
        # Create scrollable frame
        canvas = tk.Canvas(dialog, bg='#3c3c3c', highlightthickness=0)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#3c3c3c')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Store checkbox variables
        checkbox_vars = {}
        current_selections = self.pane_data[pane_id].get('selected_combinations', [])
        
        # Create selection interface
        row = 0
        
        # Select All / None buttons
        button_frame = tk.Frame(scrollable_frame, bg='#3c3c3c')
        button_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        
        def select_all():
            for var in checkbox_vars.values():
                var.set(True)
        
        def select_none():
            for var in checkbox_vars.values():
                var.set(False)
        
        tk.Button(button_frame, text="Select All", bg='#4CAF50', fg='white',
                 command=select_all).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Select None", bg='#f44336', fg='white',
                 command=select_none).pack(side=tk.LEFT, padx=5)
        
        row += 1
        
        # Headers
        tk.Label(scrollable_frame, text="Select", bg='#404040', fg='white',
                font=('Arial', 10, 'bold')).grid(row=row, column=0, sticky='ew', padx=1, pady=1)
        tk.Label(scrollable_frame, text="Index", bg='#404040', fg='white',
                font=('Arial', 10, 'bold')).grid(row=row, column=1, sticky='ew', padx=1, pady=1)
        tk.Label(scrollable_frame, text="Parameter Combination", bg='#404040', fg='white',
                font=('Arial', 10, 'bold')).grid(row=row, column=2, sticky='ew', padx=1, pady=1)
        row += 1
        
        # Add each parameter combination
        for i, param_combo in enumerate(self.result.parameter_combinations):
            # Checkbox
            var = tk.BooleanVar()
            var.set(i in current_selections)
            checkbox_vars[i] = var
            
            checkbox = tk.Checkbutton(
                scrollable_frame,
                variable=var,
                bg='#3c3c3c',
                fg='white',
                selectcolor='#404040',
                activebackground='#3c3c3c',
                activeforeground='white'
            )
            checkbox.grid(row=row, column=0, padx=5, pady=2)
            
            # Index
            tk.Label(scrollable_frame, text=str(i), bg='#3c3c3c', fg='lightblue',
                    font=('Arial', 9)).grid(row=row, column=1, sticky='w', padx=5, pady=2)
            
            # Parameter combination description
            combo_text = self._format_parameter_group(param_combo)
            tk.Label(scrollable_frame, text=combo_text, bg='#3c3c3c', fg='white',
                    font=('Arial', 9), wraplength=400, justify=tk.LEFT).grid(
                    row=row, column=2, sticky='w', padx=5, pady=2)
            
            row += 1
        
        # Configure column weights
        scrollable_frame.columnconfigure(2, weight=1)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=10)
        
        # Bottom buttons
        button_frame = tk.Frame(dialog, bg='#2b2b2b')
        button_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10)
        
        def apply_selection():
            # Get selected indices
            selected = [i for i, var in checkbox_vars.items() if var.get()]
            self.pane_data[pane_id]['selected_combinations'] = selected
            
            # Update the plot
            self._update_plot(pane_id)
            dialog.destroy()
        
        def cancel():
            dialog.destroy()
        
        tk.Button(button_frame, text="Apply", bg='#4CAF50', fg='white',
                 font=('Arial', 10, 'bold'), command=apply_selection).pack(side=tk.RIGHT, padx=10)
        tk.Button(button_frame, text="Cancel", bg='#757575', fg='white',
                 font=('Arial', 10), command=cancel).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
    
    def _get_parameter_info(self, param_name: str) -> Dict[str, str]:
        """Get parameter information from metadata."""
        if 'all_parameters' in self.result.metadata:
            for param_info in self.result.metadata['all_parameters']:
                if param_info['name'] == param_name:
                    return param_info
        return {'display_name': param_name, 'units': ''}
    
    def _on_x_param_change(self, pane_id: str, new_x_param: str, plot_creator):
        """Handle x-axis parameter change."""
        self.pane_data[pane_id]['x_axis_param'] = new_x_param
        self._update_plot(pane_id)
    
    def _create_categorical_plot_content(self, parent_frame: tk.Frame, pane_id: str):
        """Create categorical plot content."""
        def create_plot(plot_frame, pane_id):
            fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
            ax = fig.add_subplot(111, facecolor='#3c3c3c')
            
            error_type = self.pane_data[pane_id]['error_type']
            summary_type = self.pane_data[pane_id]['summary_type']
            
            if error_type and error_type in self.error_summaries[summary_type]:
                param_labels = []
                errors = []
                
                for param_val in self.result.parameter_values:
                    if param_val in self.error_summaries[summary_type][error_type]:
                        param_labels.append(str(param_val))
                        errors.append(self.error_summaries[summary_type][error_type][param_val])
                
                if param_labels:
                    ax.semilogy(param_labels, errors, 'o-', linewidth=2, markersize=8, color='orange')
                    
                    param_name = self.result.metadata.get("parameter_display_name", "Parameter")
                    
                    ax.set_xlabel(param_name, color='white')
                    ax.set_ylabel(f"{summary_type.replace('_', ' ').title()} {error_type.replace('_', ' ').title()}", 
                                color='white')
                    ax.set_title(f"{error_type.replace('_', ' ').title()} by {param_name}", color='white')
                    ax.tick_params(axis='x', rotation=45, colors='white')
                    ax.tick_params(colors='white')
                    ax.grid(True, alpha=0.3, color='gray')
                else:
                    ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                           transform=ax.transAxes, color='white')
            else:
                ax.text(0.5, 0.5, 'Select Error Type', ha='center', va='center',
                       transform=ax.transAxes, color='white')
            
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white') 
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            
            canvas = FigureCanvasTkAgg(fig, plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
        
        self._create_plot_with_controls(parent_frame, pane_id, create_plot)
    
    def _create_performance_plot_content(self, parent_frame: tk.Frame, pane_id: str):
        """Create performance plot content with parameter differentiation."""
        def create_plot(plot_frame, pane_id):
            fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
            ax = fig.add_subplot(111, facecolor='#3c3c3c')
            
            x_axis_param = self.pane_data[pane_id]['x_axis_param']
            
            if not x_axis_param:
                ax.text(0.5, 0.5, 'Select X-Axis Parameter', ha='center', va='center',
                       transform=ax.transAxes, color='white', fontsize=12)
            else:
                # Group combinations by their parameter values (excluding x-axis parameter)
                grouped_data = self._group_performance_by_parameters(x_axis_param)
                
                if not grouped_data:
                    ax.text(0.5, 0.5, 'No Performance Data Available', ha='center', va='center',
                           transform=ax.transAxes, color='white', fontsize=12)
                else:
                    # Plot each group with different colors/styles
                    colors = ['green', 'orange', 'cyan', 'magenta', 'yellow', 'red', 'lightblue', 'lightgreen']
                    linestyles = ['-', '--', '-.', ':']
                    
                    for i, (group_key, group_data) in enumerate(grouped_data.items()):
                        color = colors[i % len(colors)]
                        linestyle = linestyles[(i // len(colors)) % len(linestyles)]
                        
                        x_values = group_data['x_values']
                        y_values = group_data['y_values']
                        
                        if x_values and y_values:
                            # Sort by x-values for proper line plotting
                            sorted_pairs = sorted(zip(x_values, y_values))
                            x_sorted, y_sorted = zip(*sorted_pairs)
                            
                            if self.parameter_type == "categorical":
                                x_labels = [str(x) for x in x_sorted]
                                ax.semilogy(range(len(x_labels)), y_sorted, 's-', 
                                          color=color, linestyle=linestyle, linewidth=2, markersize=6,
                                          label=group_key, alpha=0.8)
                                ax.set_xticks(range(len(x_labels)))
                                ax.set_xticklabels(x_labels, rotation=45)
                            else:
                                ax.loglog(x_sorted, y_sorted, 's-', 
                                        color=color, linestyle=linestyle, linewidth=2, markersize=6,
                                        label=group_key, alpha=0.8)
                    
                    # Set axis labels and title
                    x_param_info = self._get_parameter_info(x_axis_param)
                    param_name = x_param_info.get('display_name', x_axis_param)
                    param_units = x_param_info.get('units', '')
                    
                    ax.set_xlabel(f"{param_name} {param_units}".strip() if self.parameter_type != "categorical" else param_name,
                                 color='white')
                    ax.set_ylabel("Computation Time (s)", color='white')
                    ax.set_title(f"Performance vs {param_name}", color='white')
                    ax.grid(True, alpha=0.3, color='gray')
                    ax.tick_params(colors='white')
                    
                    # Add legend if multiple groups
                    if len(grouped_data) > 1:
                        ax.legend(loc='best', fontsize=8, facecolor='#404040', 
                                edgecolor='white', labelcolor='white', framealpha=0.9)
            
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            
            canvas = FigureCanvasTkAgg(fig, plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
        
        # Create controls with x-axis parameter selector
        additional_controls = []
        custom_labels = []
        
        if self.pane_data[pane_id]['available_params']:
            custom_labels.append(("X-Axis:", 8))
            
            additional_controls.append({
                'type': 'x_param_selector',
                'pane_id': pane_id,
                'plot_creator': create_plot,
            })
        
        # Create the plot with enhanced controls
        self._create_plot_with_controls(parent_frame, pane_id, create_plot, 
                                      show_error_selector=False, show_summary_selector=False,
                                      additional_controls=additional_controls,
                                      custom_labels=custom_labels if additional_controls else None)
    
    def _create_time_series_content(self, parent_frame: tk.Frame, pane_id: str):
        """Create time series plot content with parameter differentiation and selection."""
        def create_plot(plot_frame, pane_id):
            fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
            ax = fig.add_subplot(111, facecolor='#3c3c3c')
            
            error_type = self.pane_data[pane_id]['error_type']
            
            if error_type and error_type in self.result.error_metrics:
                # Get selected parameter combinations
                selected_combinations = self.pane_data[pane_id].get('selected_combinations', [])
                
                if not selected_combinations:
                    ax.text(0.5, 0.5, 'No Parameter Combinations Selected\nUse controls above to select curves to plot', 
                           ha='center', va='center', transform=ax.transAxes, color='white', fontsize=12)
                else:
                    # Plot selected time series
                    colors = ['cyan', 'orange', 'lime', 'magenta', 'yellow', 'red', 'lightblue', 'lightgreen']
                    linestyles = ['-', '--', '-.', ':']
                    
                    plotted_count = 0
                    
                    for i, combo_index in enumerate(selected_combinations):
                        combo_key = f"combo_{combo_index}"
                        
                        if combo_key in self.result.error_metrics[error_type]:
                            time_series = self.result.error_metrics[error_type][combo_key]
                            
                            if time_series:
                                times = list(time_series.keys())
                                errors = list(time_series.values())
                                
                                if times and errors:
                                    color = colors[i % len(colors)]
                                    linestyle = linestyles[(i // len(colors)) % len(linestyles)]
                                    
                                    # Create label from parameter combination
                                    param_combo = self.result.parameter_combinations[combo_index]
                                    series_label = self._format_parameter_group(param_combo)
                                    
                                    ax.semilogy(times, errors, linestyle, label=series_label, 
                                              alpha=0.8, linewidth=1.5, color=color)
                                    plotted_count += 1
                    
                    # Set axis labels and title
                    ax.set_xlabel("Time", color='white')
                    ax.set_ylabel(f"{error_type.replace('_', ' ').title()}", color='white')
                    ax.set_title(f"{error_type.replace('_', ' ').title()} Time Series", color='white')
                    ax.grid(True, alpha=0.3, color='gray')
                    ax.tick_params(colors='white')
                    
                    # Add legend with automatic positioning
                    if plotted_count > 1:
                        ax.legend(loc='best', fontsize=8, facecolor='#404040', 
                                 edgecolor='white', labelcolor='white', framealpha=0.9)
            else:
                ax.text(0.5, 0.5, 'Select Error Type', ha='center', va='center',
                       transform=ax.transAxes, color='white')
            
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            
            canvas = FigureCanvasTkAgg(fig, plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
        
        # Create parameter selection controls
        additional_controls = []
        custom_labels = []
        
        # Initialize selected combinations if not exists
        if 'selected_combinations' not in self.pane_data[pane_id]:
            # Default to first few combinations
            max_default = min(8, len(self.result.parameter_combinations))
            self.pane_data[pane_id]['selected_combinations'] = list(range(max_default))
        
        # Add parameter selection button - we'll create the actual widget in the controls handler
        custom_labels.append(("Select Curves:", 8))
        additional_controls.append({
            'type': 'param_selector_button',
            'pane_id': pane_id,
            'plot_creator': create_plot,
            'command': lambda: self._show_parameter_selection_dialog(pane_id, create_plot)
        })
        
        self._create_plot_with_controls(parent_frame, pane_id, create_plot, 
                                      show_summary_selector=False,
                                      additional_controls=additional_controls,
                                      custom_labels=custom_labels)
    
    def _create_error_comparison_content(self, parent_frame: tk.Frame, pane_id: str):
        """Create error comparison plot content."""
        def create_plot(plot_frame, pane_id):
            fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
            ax = fig.add_subplot(111, facecolor='#3c3c3c')
            
            summary_type = self.pane_data[pane_id]['summary_type']
            colors = ['cyan', 'orange', 'lime', 'magenta', 'yellow', 'red', 'lightblue', 'lightgreen']
            
            for i, error_type in enumerate(self.error_types):
                if error_type in self.error_summaries[summary_type]:
                    param_values = []
                    errors = []
                    
                    for param_val in self.result.parameter_values:
                        if param_val in self.error_summaries[summary_type][error_type]:
                            param_values.append(param_val)
                            errors.append(self.error_summaries[summary_type][error_type][param_val])
                    
                    if param_values:
                        if self.parameter_type == "categorical":
                            param_labels = [str(p) for p in param_values]
                            ax.semilogy(param_labels, errors, 'o-', linewidth=2, markersize=6,
                                      label=error_type.replace('_', ' ').title(), color=colors[i])
                        else:
                            ax.loglog(param_values, errors, 'o-', linewidth=2, markersize=6,
                                    label=error_type.replace('_', ' ').title(), color=colors[i])
            
            param_name = self.result.metadata.get("parameter_display_name", "Parameter")
            param_units = self.result.metadata.get("parameter_units", "")
            
            ax.set_xlabel(f"{param_name} {param_units}".strip() if self.parameter_type != "categorical" else param_name,
                         color='white')
            ax.set_ylabel(f"{summary_type.replace('_', ' ').title()}", color='white')
            ax.set_title("Error Type Comparison", color='white') 
            ax.grid(True, alpha=0.3, color='gray')
            ax.tick_params(colors='white')
            
            if self.parameter_type == "categorical":
                ax.tick_params(axis='x', rotation=45)
            
            ax.legend(loc='best', fontsize=8, facecolor='#404040', 
                     edgecolor='white', labelcolor='white', framealpha=0.9)
            
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            
            canvas = FigureCanvasTkAgg(fig, plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
        
        self._create_plot_with_controls(parent_frame, pane_id, create_plot, show_error_selector=False)
    
    def _create_heatmap_content(self, parent_frame: tk.Frame, pane_id: str):
        """Create heatmap plot content."""
        def create_plot(plot_frame, pane_id):
            fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
            ax = fig.add_subplot(111, facecolor='#3c3c3c')
            
            error_type = self.pane_data[pane_id]['error_type']
            
            if error_type and error_type in self.result.error_metrics:
                param_values = []
                all_times = set()
                
                # Collect all times and parameters
                for param_val in self.result.parameter_values:
                    if param_val in self.result.error_metrics[error_type]:
                        param_values.append(param_val)
                        time_series = self.result.error_metrics[error_type][param_val]
                        all_times.update(time_series.keys())
                
                if param_values and all_times:
                    times = sorted(all_times)
                    error_matrix = np.full((len(param_values), len(times)), np.nan)
                    
                    for i, param_val in enumerate(param_values):
                        time_series = self.result.error_metrics[error_type][param_val]
                        for j, time in enumerate(times):
                            if time in time_series:
                                error_matrix[i, j] = time_series[time]
                    
                    # Create heatmap
                    im = ax.imshow(error_matrix, aspect='auto', cmap='viridis', 
                                 interpolation='nearest', origin='lower')
                    
                    # Set labels
                    ax.set_xlabel("Time Index", color='white')
                    ax.set_ylabel("Parameter Index", color='white') 
                    ax.set_title(f"{error_type.replace('_', ' ').title()} Heatmap", color='white')
                    ax.tick_params(colors='white')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, label="Error")
                    cbar.ax.yaxis.label.set_color('white')
                    cbar.ax.tick_params(colors='white')
                else:
                    ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center',
                           transform=ax.transAxes, color='white')
            else:
                ax.text(0.5, 0.5, 'Select Error Type', ha='center', va='center',
                       transform=ax.transAxes, color='white')
            
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            
            canvas = FigureCanvasTkAgg(fig, plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
        
        self._create_plot_with_controls(parent_frame, pane_id, create_plot, show_summary_selector=False)
    
    def _create_parameter_space_content(self, parent_frame: tk.Frame, pane_id: str):
        """Create parameter space visualization content."""
        def create_plot(plot_frame, pane_id):
            fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
            ax = fig.add_subplot(111, facecolor='#3c3c3c')
            
            # This is a placeholder for multi-dimensional parameter visualization
            # In a real implementation, you might show PCA, t-SNE, or other dimensionality reduction
            ax.text(0.5, 0.5, 'Parameter Space Visualization\n\n(Coming Soon)\n\nWould show multi-dimensional\nparameter relationships',
                   ha='center', va='center', transform=ax.transAxes, color='white', fontsize=12)
            ax.set_title("Parameter Space", color='white')
            
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.tick_params(colors='white')
            
            canvas = FigureCanvasTkAgg(fig, plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self._create_plot_with_controls(parent_frame, pane_id, create_plot, 
                                      show_error_selector=False, show_summary_selector=False)
    
    def _create_statistics_table_content(self, parent_frame: tk.Frame, pane_id: str):
        """Create statistics table content."""
        # Title
        title_label = tk.Label(
            parent_frame,
            text="Error Analysis Statistics",
            bg='#3c3c3c',
            fg='white',
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=10)
        
        # Create scrollable frame for statistics
        canvas = tk.Canvas(parent_frame, bg='#3c3c3c', highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#3c3c3c')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add statistics for each error type
        row = 0
        for error_type in self.error_types:
            # Error type header
            header_frame = tk.Frame(scrollable_frame, bg='#404040')
            header_frame.grid(row=row, column=0, columnspan=5, sticky='ew', pady=(10, 0))
            
            tk.Label(header_frame, text=f"{error_type.replace('_', ' ').title()}", 
                    bg='#404040', fg='white', font=('Arial', 12, 'bold')).pack(pady=5)
            row += 1
            
            # Column headers
            headers = ['Parameter', 'RMS Error', 'Max Error', 'Final Error', 'Mean Error']
            for col, header in enumerate(headers):
                tk.Label(scrollable_frame, text=header, bg='#555555', fg='white',
                        font=('Arial', 10, 'bold'), relief='solid', bd=1).grid(
                        row=row, column=col, sticky='ew', padx=1, pady=1)
            row += 1
            
            # Data rows
            for param_val in self.result.parameter_values:
                # Parameter value
                tk.Label(scrollable_frame, text=str(param_val), bg='#3c3c3c', fg='lightblue',
                        font=('Arial', 9), relief='solid', bd=1).grid(
                        row=row, column=0, sticky='ew', padx=1, pady=1)
                
                # Statistics
                for col, summary_type in enumerate(['rms_errors', 'max_errors', 'final_errors', 'mean_errors'], 1):
                    if (error_type in self.error_summaries[summary_type] and 
                        param_val in self.error_summaries[summary_type][error_type]):
                        value = self.error_summaries[summary_type][error_type][param_val]
                        text = f"{value:.2e}" if value > 0.001 or value < -0.001 else f"{value:.6f}"
                    else:
                        text = "N/A"
                    
                    tk.Label(scrollable_frame, text=text, bg='#3c3c3c', fg='white',
                            font=('Arial', 9), relief='solid', bd=1).grid(
                            row=row, column=col, sticky='ew', padx=1, pady=1)
                
                row += 1
        
        # Configure column weights
        for col in range(5):
            scrollable_frame.columnconfigure(col, weight=1)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def _create_controls_panel_content(self, parent_frame: tk.Frame, pane_id: str):
        """Create controls panel content."""
        # Title
        title_label = tk.Label(
            parent_frame,
            text="Analysis Controls",
            bg='#3c3c3c',
            fg='white',
            font=('Arial', 14, 'bold')
        )
        title_label.pack(pady=10)
        
        # Global controls frame
        controls_frame = tk.Frame(parent_frame, bg='#3c3c3c')
        controls_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Export controls
        export_frame = tk.LabelFrame(controls_frame, text="Export Options", 
                                   bg='#404040', fg='white', font=('Arial', 11, 'bold'))
        export_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(export_frame, text="Save All Plots", width=15, height=2,
                 bg='#4CAF50', fg='white', font=('Arial', 10, 'bold'),
                 command=self._save_all_plots).pack(pady=5)
        
        tk.Button(export_frame, text="Export Data", width=15, height=2,
                 bg='#2196F3', fg='white', font=('Arial', 10, 'bold'),
                 command=self._export_data).pack(pady=5)
        
        # Analysis info
        info_frame = tk.LabelFrame(controls_frame, text="Analysis Information",
                                 bg='#404040', fg='white', font=('Arial', 11, 'bold'))
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Add analysis metadata
        info_text = tk.Text(info_frame, height=8, bg='#3c3c3c', fg='white',
                           font=('Courier', 9), wrap=tk.WORD)
        info_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Insert analysis information
        info_content = f"""Analysis Summary:
• Parameter: {self.result.metadata.get('parameter_display_name', 'Unknown')}
• Parameter Values: {len(self.parameter_values)} values
• Error Types: {len(self.error_types)} types
• Total Runs: {len(self.result.computation_times)}
• Parameter Type: {self.parameter_type.title()}

Error Types:
{chr(10).join(f"  • {et.replace('_', ' ').title()}" for et in self.error_types)}

Parameter Range:
  • Min: {min(self.parameter_values)}
  • Max: {max(self.parameter_values)}

Computation Times:
  • Total: {sum(self.result.computation_times):.2f}s
  • Average: {np.mean(self.result.computation_times):.2f}s
  • Min: {min(self.result.computation_times):.2f}s
  • Max: {max(self.result.computation_times):.2f}s"""
        
        info_text.insert(tk.END, info_content)
        info_text.configure(state=tk.DISABLED)
    
    def _save_all_plots(self):
        """Save all plots (placeholder)."""
        print("Save all plots functionality would be implemented here")
        
    def _export_data(self):
        """Export data (placeholder)."""
        print("Export data functionality would be implemented here")


class InteractivePlotter:
    """Main application class for the interactive plotter with window splitter."""
    
    def __init__(self, result: 'ErrorAnalysisResult', master=None):
        """Initialize the interactive plotter with splitter.
        
        Args:
            result: ErrorAnalysisResult from parameter sweep
            master: Parent Tkinter window (None for new window)
        """
        self.result = result
        self.master = master if master else tk.Tk()
        self.master.title("Interactive Error Analysis - Advanced Interface")
        self.master.geometry("1600x1000")
        self.master.configure(bg='#2b2b2b')
        
        self._setup_gui()
    
    def _setup_gui(self):
        """Set up the main GUI."""
        # Title bar
        title_frame = tk.Frame(self.master, bg='#404040', height=60)
        title_frame.pack(fill=tk.X, side=tk.TOP)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="Interactive Error Analysis - Advanced Multi-Pane Interface",
            bg='#404040',
            fg='white',
            font=('Arial', 16, 'bold')
        )
        title_label.pack(expand=True)
        
        # Instructions
        info_frame = tk.Frame(self.master, bg='#404040', height=40)
        info_frame.pack(fill=tk.X, side=tk.BOTTOM)
        info_frame.pack_propagate(False)
        
        info_text = ("Split panes using ⬌/⬍ buttons • Change plot types via dropdowns • "
                    "Drag splitters to resize • Make panes small to delete them • "
                    "Professional error analysis with multiple simultaneous views")
        tk.Label(info_frame, text=info_text, bg='#404040', fg='white',
                font=('Arial', 9), wraplength=1500, justify=tk.LEFT).pack(pady=8, padx=10)
        
        # Main container for splitter
        main_container = tk.Frame(self.master, bg='#2b2b2b')
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create content provider
        content_provider = ErrorAnalysisContentProvider(self.result)
        
        # Create splitter manager with a good starting pane type
        parameter_type = content_provider.parameter_type
        default_pane_type = (PlotPaneTypes.LINE_PLOT if parameter_type == "quantitative" 
                            else PlotPaneTypes.CATEGORICAL_PLOT)
        
        self.splitter_manager = WindowSplitterManager(main_container, default_pane_type)
        self.splitter_manager.set_content_provider(content_provider)
        
        # Store reference to content provider for potential later use
        self.content_provider = content_provider
    
    def show(self):
        """Show the application."""
        if hasattr(self.master, "mainloop"):
            self.master.mainloop()


def create_advanced_interactive_plot(result: 'ErrorAnalysisResult', master=None):
    """Create an advanced interactive plot with window splitter.
    
    Args:
        result: ErrorAnalysisResult from parameter sweep
        master: Parent Tkinter window (None for new window)
    
    Returns:
        InteractivePlotterWithSplitter instance
    """
    plotter = InteractivePlotter(result, master)
    plotter.show()
    return plotter


if __name__ == "__main__":
    # Demo/test functionality
    print("InteractivePlotterWithSplitter - Advanced Error Analysis Interface")
    print("Import this module and use create_advanced_interactive_plot() with your ErrorAnalysisResult")
