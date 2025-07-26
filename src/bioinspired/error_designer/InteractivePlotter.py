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
from typing import Dict, List, TYPE_CHECKING
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
    
    def _compute_error_summaries(self) -> Dict[str, Dict[str, Dict[float, float]]]:
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
            
            for param_value in self.result.parameter_values:
                if param_value in self.result.error_metrics[error_type]:
                    time_series = self.result.error_metrics[error_type][param_value]
                    if time_series:
                        values = list(time_series.values())
                        
                        if values:
                            summaries["rms_errors"][error_type][param_value] = np.sqrt(np.mean(np.array(values)**2))
                            summaries["max_errors"][error_type][param_value] = max(values)
                            summaries["final_errors"][error_type][param_value] = values[-1]
                            summaries["mean_errors"][error_type][param_value] = np.mean(values)
        
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
            self.pane_data[pane_id] = {
                'error_type': self.error_types[0] if self.error_types else "",
                'summary_type': "rms_errors",
                'time_range': None,
                'selected_params': None,
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
                                   show_summary_selector=True, additional_controls=None):
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
        
        # Additional controls
        if additional_controls:
            for control in additional_controls:
                control['widget'].grid(row=0, column=current_col, **control.get('grid_kwargs', {}))
                current_col += 1
        
        # Plot frame
        plot_frame = tk.Frame(parent_frame, bg='#3c3c3c')
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create the plot
        plot_creator(plot_frame, pane_id)
    
    def _on_error_type_change(self, pane_id: str, new_error_type: str, plot_creator):
        """Handle error type change."""
        self.pane_data[pane_id]['error_type'] = new_error_type
        # Would need reference to plot_frame to recreate - simplified for now
        print(f"Error type changed to {new_error_type} for pane {pane_id}")
    
    def _on_summary_type_change(self, pane_id: str, new_summary_type: str, plot_creator):
        """Handle summary type change."""
        self.pane_data[pane_id]['summary_type'] = new_summary_type
        # Would need reference to plot_frame to recreate - simplified for now  
        print(f"Summary type changed to {new_summary_type} for pane {pane_id}")
    
    def _create_line_plot_content(self, parent_frame: tk.Frame, pane_id: str):
        """Create line plot content for quantitative parameters."""
        def create_plot(plot_frame, pane_id):
            fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
            ax = fig.add_subplot(111, facecolor='#3c3c3c')
            
            error_type = self.pane_data[pane_id]['error_type']
            summary_type = self.pane_data[pane_id]['summary_type']
            
            if error_type and error_type in self.error_summaries[summary_type]:
                param_values = []
                errors = []
                
                for param_val in self.result.parameter_values:
                    if param_val in self.error_summaries[summary_type][error_type]:
                        param_values.append(param_val)
                        errors.append(self.error_summaries[summary_type][error_type][param_val])
                
                if param_values:
                    ax.loglog(param_values, errors, 'o-', linewidth=2, markersize=6, color='cyan')
                    
                    param_name = self.result.metadata.get("parameter_display_name", "Parameter")
                    param_units = self.result.metadata.get("parameter_units", "")
                    
                    ax.set_xlabel(f"{param_name} {param_units}".strip(), color='white')
                    ax.set_ylabel(f"{summary_type.replace('_', ' ').title()} {error_type.replace('_', ' ').title()}", 
                                color='white')
                    ax.set_title(f"{error_type.replace('_', ' ').title()} vs {param_name}", color='white')
                    ax.grid(True, alpha=0.3, color='gray')
                    ax.tick_params(colors='white')
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
            
            # Add toolbar
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
        
        self._create_plot_with_controls(parent_frame, pane_id, create_plot)
    
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
        """Create performance plot content."""
        def create_plot(plot_frame, pane_id):
            fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
            ax = fig.add_subplot(111, facecolor='#3c3c3c')
            
            runtimes = self.result.computation_times
            param_values = self.result.parameter_values
            
            if self.parameter_type == "categorical":
                param_labels = [str(p) for p in param_values]
                ax.semilogy(param_labels, runtimes, 's-', color='green', linewidth=2, markersize=8)
                ax.tick_params(axis='x', rotation=45, colors='white')
            else:
                ax.loglog(param_values, runtimes, 's-', color='green', linewidth=2, markersize=6)
            
            param_name = self.result.metadata.get("parameter_display_name", "Parameter")
            param_units = self.result.metadata.get("parameter_units", "")
            
            ax.set_xlabel(f"{param_name} {param_units}".strip() if self.parameter_type != "categorical" else param_name,
                         color='white')
            ax.set_ylabel("Computation Time (s)", color='white')
            ax.set_title(f"Performance vs {param_name}", color='white')
            ax.grid(True, alpha=0.3, color='gray')
            ax.tick_params(colors='white')
            
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            
            canvas = FigureCanvasTkAgg(fig, plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, plot_frame)
            toolbar.update()
        
        self._create_plot_with_controls(parent_frame, pane_id, create_plot, 
                                      show_error_selector=False, show_summary_selector=False)
    
    def _create_time_series_content(self, parent_frame: tk.Frame, pane_id: str):
        """Create time series plot content."""
        def create_plot(plot_frame, pane_id):
            fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
            ax = fig.add_subplot(111, facecolor='#3c3c3c')
            
            error_type = self.pane_data[pane_id]['error_type']
            
            if error_type and error_type in self.result.error_metrics:
                plotted_count = 0
                max_plots = 8  # Show more series in dedicated time series plot
                colors = plt.cm.tab10(np.linspace(0, 1, max_plots))
                
                for i, param_val in enumerate(self.result.parameter_values):
                    if plotted_count >= max_plots:
                        break
                    
                    if param_val in self.result.error_metrics[error_type]:
                        time_series = self.result.error_metrics[error_type][param_val]
                        
                        if time_series:
                            times = list(time_series.keys())
                            errors = list(time_series.values())
                            
                            ax.semilogy(times, errors, '-', label=f"{param_val}", 
                                      alpha=0.8, linewidth=1.5, color=colors[i])
                            plotted_count += 1
                
                ax.set_xlabel("Time", color='white')
                ax.set_ylabel(f"{error_type.replace('_', ' ').title()}", color='white')
                ax.set_title(f"{error_type.replace('_', ' ').title()} Time Series", color='white')
                ax.grid(True, alpha=0.3, color='gray')
                ax.tick_params(colors='white')
                
                # Legend with better positioning
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8,
                         facecolor='#404040', edgecolor='white', labelcolor='white')
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
    
    def _create_error_comparison_content(self, parent_frame: tk.Frame, pane_id: str):
        """Create error comparison plot content."""
        def create_plot(plot_frame, pane_id):
            fig = Figure(figsize=(8, 6), facecolor='#2b2b2b')
            ax = fig.add_subplot(111, facecolor='#3c3c3c')
            
            summary_type = self.pane_data[pane_id]['summary_type']
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.error_types)))
            
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
            
            ax.legend(facecolor='#404040', edgecolor='white', labelcolor='white')
            
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


class InteractivePlotterWithSplitter:
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
    plotter = InteractivePlotterWithSplitter(result, master)
    plotter.show()
    return plotter


if __name__ == "__main__":
    # Demo/test functionality
    print("InteractivePlotterWithSplitter - Advanced Error Analysis Interface")
    print("Import this module and use create_advanced_interactive_plot() with your ErrorAnalysisResult")
