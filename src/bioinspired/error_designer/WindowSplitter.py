"""Generalized Window Splitter Framework
Uses a binary tree structure to manage pane splits and deletions.
Each node is either a split (with 2 children) or a leaf (with content).

This framework is designed to be extended for specific applications by:
1. Implementing the BasePaneType abstract class for your pane types
2. Implementing the PaneContentProvider abstract class for custom content creation
3. Optionally extending the WindowSplitterManager for application-specific behavior
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional, Any, TypeVar
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod


class SplitDirection(Enum):
    """Direction for splitting a pane."""
    HORIZONTAL = "horizontal"  # Split into top/bottom
    VERTICAL = "vertical"      # Split into left/right


# Type variable for pane types
PaneTypeVar = TypeVar('PaneTypeVar', bound='BasePaneType')


class BasePaneType(ABC):
    """Abstract base class for pane types.
    
    Implement this class to define the types of content that can be displayed
    in panes for your specific application.
    """
    
    @property
    @abstractmethod
    def value(self) -> str:
        """The display value for this pane type."""
        pass
    
    @property
    @abstractmethod
    def identifier(self) -> str:
        """A unique identifier for this pane type."""
        pass
    
    def __eq__(self, other):
        if isinstance(other, BasePaneType):
            return self.identifier == other.identifier
        return False
    
    def __hash__(self):
        return hash(self.identifier)
    
    def __str__(self):
        return self.value


class PaneContentProvider(ABC):
    """Abstract base class for providing content to panes.
    
    Implement this class to define how content is created for different pane types.
    """
    
    @abstractmethod
    def create_content(self, parent_frame: tk.Frame, pane_type: BasePaneType) -> None:
        """Create content for the given pane type in the parent frame.
        
        Args:
            parent_frame: The tkinter Frame where content should be added
            pane_type: The type of pane to create content for
        """
        pass
    
    @abstractmethod
    def get_available_pane_types(self) -> List[BasePaneType]:
        """Get all available pane types for this provider."""
        pass


@dataclass
class PaneConfig:
    """Configuration for a pane."""
    pane_type: BasePaneType
    title: str = "Untitled"
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


class PaneNode(ABC):
    """Abstract base class for pane tree nodes."""
    
    def __init__(self, parent: Optional['SplitNode'] = None):
        self.parent = parent
        self.widget: Optional[tk.Widget] = None
        self.min_size_threshold = 50
        self.is_marked_for_deletion = False
    
    @abstractmethod
    def render(self, parent_widget: tk.Widget, splitter_manager: 'WindowSplitterManager'):
        """Render this node and its children in the given parent widget."""
        pass
    
    @abstractmethod
    def find_leaf_by_widget(self, widget: tk.Widget) -> Optional['LeafNode']:
        """Find the leaf node that owns the given widget."""
        pass
    
    @abstractmethod
    def get_all_leaves(self) -> List['LeafNode']:
        """Get all leaf nodes in this subtree."""
        pass
    
    def mark_for_deletion(self):
        """Mark this node for deletion."""
        self.is_marked_for_deletion = True
    
    def unmark_for_deletion(self):
        """Unmark this node for deletion."""
        self.is_marked_for_deletion = False


class LeafNode(PaneNode):
    """A leaf node containing actual pane content."""
    
    def __init__(self, config: PaneConfig, parent: Optional['SplitNode'] = None):
        super().__init__(parent)
        self.config = config
        self.content_provider: Optional[PaneContentProvider] = None
        self.main_frame: Optional[tk.Frame] = None
        self.header_frame: Optional[tk.Frame] = None
        self.content_frame: Optional[tk.Frame] = None
        self.pane_dropdown: Optional[ttk.Combobox] = None
        self.title_label: Optional[tk.Label] = None
    
    def render(self, parent_widget: tk.Widget, splitter_manager: 'WindowSplitterManager'):
        """Render this leaf pane."""
        if self.main_frame:
            self.main_frame.destroy()
        
        # Ensure parent widget is valid
        try:
            parent_widget.winfo_exists()
        except tk.TclError:
            print("Parent widget is invalid, cannot render leaf")
            return
        
        # Main container
        self.main_frame = tk.Frame(parent_widget, bg='#2b2b2b', relief='solid', bd=1)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self.widget = self.main_frame
        
        # Header with controls
        self._create_header(splitter_manager)
        
        # Content area
        self.content_frame = tk.Frame(self.main_frame, bg='#3c3c3c')
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Apply gray effect if marked for deletion
        if self.is_marked_for_deletion:
            self._apply_gray_effect()
        
        # Create content
        self._update_content()
        
        # Bind configure event for size monitoring
        self.main_frame.bind('<Configure>', lambda e: self._on_configure(e, splitter_manager))
    
    def _create_header(self, splitter_manager: 'WindowSplitterManager'):
        """Create the header with dropdown and split buttons."""
        self.header_frame = tk.Frame(self.main_frame, bg='#404040', height=30)
        self.header_frame.pack(fill=tk.X, side=tk.TOP)
        self.header_frame.pack_propagate(False)
        
        # Pane type dropdown
        tk.Label(self.header_frame, text="Type:", bg='#404040', fg='white', 
                font=('Arial', 8)).pack(side=tk.LEFT, padx=(5, 2))
        
        # Get available pane types from the content provider
        available_types = []
        if self.content_provider:
            available_types = [ptype.value for ptype in self.content_provider.get_available_pane_types()]
        
        self.pane_dropdown = ttk.Combobox(
            self.header_frame,
            values=available_types,
            width=15,
            state='readonly',
            font=('Arial', 8)
        )
        if available_types:
            self.pane_dropdown.set(self.config.pane_type.value)
        self.pane_dropdown.pack(side=tk.LEFT, padx=(0, 5))
        self.pane_dropdown.bind('<<ComboboxSelected>>', self._on_type_change)
        
        # Add a separator and title
        separator = tk.Frame(self.header_frame, bg='#606060', width=1, height=20)
        separator.pack(side=tk.LEFT, padx=5)
        
        self.title_label = tk.Label(
            self.header_frame, 
            text=self.config.pane_type.value,
            bg='#404040', 
            fg='#cccccc',
            font=('Arial', 8, 'bold')
        )
        self.title_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Split buttons
        split_frame = tk.Frame(self.header_frame, bg='#404040')
        split_frame.pack(side=tk.RIGHT, padx=5)
        
        # Close/Delete button (X)
        close_btn = tk.Button(
            split_frame,
            text="✕",
            width=2,
            height=1,
            font=('Arial', 8),
            fg='#ff6666',
            command=lambda: splitter_manager.delete_pane(self)
        )
        close_btn.pack(side=tk.LEFT, padx=1)
        
        # Horizontal split button
        h_split_btn = tk.Button(
            split_frame, 
            text="⬌", 
            width=2, 
            height=1,
            font=('Arial', 8),
            command=lambda: splitter_manager.split_pane(self, SplitDirection.HORIZONTAL)
        )
        h_split_btn.pack(side=tk.LEFT, padx=1)
        
        # Vertical split button
        v_split_btn = tk.Button(
            split_frame, 
            text="⬍", 
            width=2, 
            height=1,
            font=('Arial', 8),
            command=lambda: splitter_manager.split_pane(self, SplitDirection.VERTICAL)
        )
        v_split_btn.pack(side=tk.LEFT, padx=1)
    
    def _on_type_change(self, event=None):
        """Handle pane type change."""
        new_type_str = self.pane_dropdown.get()
        if not self.content_provider:
            return
        
        # Find the corresponding pane type
        available_types = self.content_provider.get_available_pane_types()
        new_type = next((pt for pt in available_types if pt.value == new_type_str), None)
        
        if new_type and new_type != self.config.pane_type:
            self.config.pane_type = new_type
            
            # Update title label
            if self.title_label:
                self.title_label.configure(text=new_type.value)
            
            self._update_content()
    
    def _update_content(self):
        """Update the content area based on the current pane type."""
        # Clear existing content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Create content based on pane type
        if self.content_provider:
            self.content_provider.create_content(self.content_frame, self.config.pane_type)
        else:
            self._create_default_content()
    
    def _create_default_content(self):
        """Create default content when no content provider is available."""
        label = tk.Label(
            self.content_frame, 
            text=f"No Content Provider\n\nPane Type: {self.config.pane_type.value}\n\nSet a content provider to display custom content",
            bg='#3c3c3c', 
            fg='gray',
            font=('Arial', 10),
            justify=tk.CENTER
        )
        label.pack(expand=True)
    
    def _on_configure(self, event, splitter_manager: 'WindowSplitterManager'):
        """Handle resize events to check for deletion threshold."""
        if event.widget != self.main_frame:
            return
        
        # Only check size if we have a parent (not root)
        if not self.parent:
            return
        
        # Check if this pane is too small
        if self.parent.direction == SplitDirection.HORIZONTAL:
            size = event.height
        else:
            size = event.width
        
        if size < self.min_size_threshold:
            if not self.is_marked_for_deletion:
                self.mark_for_deletion()
                self._apply_gray_effect()
        else:
            if self.is_marked_for_deletion:
                self.unmark_for_deletion()
                self._remove_gray_effect()
    
    def _apply_gray_effect(self):
        """Apply gray effect to indicate pending deletion."""
        gray_color = '#666666'
        
        if self.main_frame:
            self.main_frame.configure(bg=gray_color)
        if self.header_frame:
            self.header_frame.configure(bg=gray_color)
        if self.content_frame:
            self.content_frame.configure(bg=gray_color)
        
        # Gray out all child widgets recursively
        self._gray_out_children(self.main_frame, gray_color)
    
    def _remove_gray_effect(self):
        """Remove gray effect and restore normal colors."""
        if self.main_frame:
            self.main_frame.configure(bg='#2b2b2b')
        if self.header_frame:
            self.header_frame.configure(bg='#404040')
        if self.content_frame:
            self.content_frame.configure(bg='#3c3c3c')
        
        # Restore all child widgets recursively
        self._restore_children_colors(self.main_frame)
    
    def _gray_out_children(self, widget, gray_color):
        """Recursively gray out all child widgets."""
        for child in widget.winfo_children():
            try:
                if not hasattr(child, '_original_bg'):
                    try:
                        child._original_bg = child.cget('bg')
                    except tk.TclError:
                        child._original_bg = None
                
                if child._original_bg:
                    child.configure(bg=gray_color)
                    
                self._gray_out_children(child, gray_color)
            except tk.TclError:
                pass
    
    def _restore_children_colors(self, widget):
        """Recursively restore original colors of all child widgets."""
        for child in widget.winfo_children():
            try:
                if hasattr(child, '_original_bg') and child._original_bg:
                    child.configure(bg=child._original_bg)
                    delattr(child, '_original_bg')
                    
                self._restore_children_colors(child)
            except tk.TclError:
                pass
    
    def find_leaf_by_widget(self, widget: tk.Widget) -> Optional['LeafNode']:
        """Find the leaf node that owns the given widget."""
        if self.widget == widget or self.main_frame == widget:
            return self
        return None
    
    def get_all_leaves(self) -> List['LeafNode']:
        """Get all leaf nodes (just this one)."""
        return [self]


class SplitNode(PaneNode):
    """A split node containing two child nodes."""
    
    def __init__(self, direction: SplitDirection, left: PaneNode, right: PaneNode, 
                 parent: Optional['SplitNode'] = None):
        super().__init__(parent)
        self.direction = direction
        self.left = left
        self.right = right
        # Explicitly set parent references for children
        self.left.parent = self
        self.right.parent = self
        self.paned_window: Optional[tk.PanedWindow] = None
        self.is_dragging = False
    
    def render(self, parent_widget: tk.Widget, splitter_manager: 'WindowSplitterManager'):
        """Render this split and its children."""
        if self.paned_window:
            self.paned_window.destroy()
        
        # Ensure parent widget is valid
        try:
            parent_widget.winfo_exists()
        except tk.TclError:
            print("Parent widget is invalid, cannot render")
            return
        
        # Create PanedWindow
        orient = tk.VERTICAL if self.direction == SplitDirection.HORIZONTAL else tk.HORIZONTAL
        self.paned_window = tk.PanedWindow(
            parent_widget, 
            orient=orient,
            bg='#555555', 
            sashwidth=6, 
            sashrelief=tk.FLAT,
            bd=0, 
            sashpad=0, 
            showhandle=False
        )
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        self.widget = self.paned_window
        
        # Create frames for children - these will be the direct children of the PanedWindow
        left_container = tk.Frame(self.paned_window, bg='#2b2b2b')
        right_container = tk.Frame(self.paned_window, bg='#2b2b2b')
        
        # Add to PanedWindow
        self.paned_window.add(left_container, minsize=30)
        self.paned_window.add(right_container, minsize=30)
        
        # Render children - pass the containers, not frames inside PanedWindow
        self.left.render(left_container, splitter_manager)
        self.right.render(right_container, splitter_manager)
        
        # After rendering, set equal sizes for the panes
        parent_widget.update_idletasks()  # Ensure the widget is properly sized
        # Only distribute space for this specific split, not recursively
        self._distribute_space_equally()
        
        # Bind events for deletion behavior
        self.paned_window.bind('<B1-Motion>', lambda e: self._on_sash_drag(e, splitter_manager))
        self.paned_window.bind('<Button-1>', self._on_sash_press)
        self.paned_window.bind('<ButtonRelease-1>', lambda e: self._on_sash_release(e, splitter_manager))
    
    def _distribute_space_equally(self):
        """Distribute space equally between the two panes."""
        try:
            # Wait a bit for the paned window to be fully rendered
            self.paned_window.update_idletasks()
            
            if self.direction == SplitDirection.HORIZONTAL:
                # Horizontal split - divide height
                total_height = self.paned_window.winfo_height()
                if total_height > 60:  # Only if we have reasonable space
                    half_height = (total_height - 6) // 2  # Account for sash width
                    self.paned_window.sash_place(0, 0, half_height)
            else:
                # Vertical split - divide width  
                total_width = self.paned_window.winfo_width()
                if total_width > 60:  # Only if we have reasonable space
                    half_width = (total_width - 6) // 2  # Account for sash width
                    self.paned_window.sash_place(0, half_width, 0)
        except tk.TclError:
            pass  # Ignore errors during initial rendering
    
    def _on_sash_press(self, event):
        """Handle sash press to start dragging."""
        self.is_dragging = True
    
    def _on_sash_drag(self, event, splitter_manager: 'WindowSplitterManager'):
        """Handle sash dragging to show deletion preview."""
        if not self.is_dragging:
            return
        
        # Check sizes and mark for deletion if too small
        self._check_child_sizes_for_deletion()
    
    def _on_sash_release(self, event, splitter_manager: 'WindowSplitterManager'):
        """Handle sash release to actually delete marked panes."""
        if not self.is_dragging:
            return
        
        self.is_dragging = False
        
        # Delete any marked children
        if self.left.is_marked_for_deletion:
            splitter_manager.delete_pane(self.left)
        elif self.right.is_marked_for_deletion:
            splitter_manager.delete_pane(self.right)
    
    def _check_child_sizes_for_deletion(self):
        """Check if children are too small and mark for deletion."""
        try:
            # Get the actual containers that hold the children
            panes = self.paned_window.panes()
            if len(panes) != 2:
                return
            
            left_container = panes[0]
            right_container = panes[1]
            
            if self.direction == SplitDirection.HORIZONTAL:
                left_size = left_container.winfo_height()
                right_size = right_container.winfo_height()
            else:
                left_size = left_container.winfo_width()
                right_size = right_container.winfo_width()
            
            # Mark/unmark left child
            if left_size < self.left.min_size_threshold:
                if not self.left.is_marked_for_deletion:
                    self.left.mark_for_deletion()
                    if isinstance(self.left, LeafNode):
                        self.left._apply_gray_effect()
            else:
                if self.left.is_marked_for_deletion:
                    self.left.unmark_for_deletion()
                    if isinstance(self.left, LeafNode):
                        self.left._remove_gray_effect()
            
            # Mark/unmark right child
            if right_size < self.right.min_size_threshold:
                if not self.right.is_marked_for_deletion:
                    self.right.mark_for_deletion()
                    if isinstance(self.right, LeafNode):
                        self.right._apply_gray_effect()
            else:
                if self.right.is_marked_for_deletion:
                    self.right.unmark_for_deletion()
                    if isinstance(self.right, LeafNode):
                        self.right._remove_gray_effect()
        
        except tk.TclError:
            pass  # Ignore errors during resize
    
    def find_leaf_by_widget(self, widget: tk.Widget) -> Optional['LeafNode']:
        """Find the leaf node that owns the given widget."""
        left_result = self.left.find_leaf_by_widget(widget)
        if left_result:
            return left_result
        return self.right.find_leaf_by_widget(widget)
    
    def get_all_leaves(self) -> List['LeafNode']:
        """Get all leaf nodes in this subtree."""
        return self.left.get_all_leaves() + self.right.get_all_leaves()


class WindowSplitterManager:
    """Manager class for the binary tree window splitting system."""
    
    def __init__(self, parent: tk.Widget, default_pane_type: BasePaneType):
        """Initialize the window splitter manager.
        
        Args:
            parent: The parent widget to contain the splitter
            default_pane_type: The default pane type for new panes
        """
        self.parent = parent
        self.default_pane_type = default_pane_type
        self.root: PaneNode = None
        self.content_provider: Optional[PaneContentProvider] = None
        self.newly_created_split: Optional['SplitNode'] = None
        
        self._setup_root()
    
    def _setup_root(self):
        """Set up the root pane."""
        self.root = LeafNode(PaneConfig(pane_type=self.default_pane_type))
        self.root.render(self.parent, self)
    
    def set_content_provider(self, provider: PaneContentProvider):
        """Set the content provider for creating pane content."""
        self.content_provider = provider
        self._apply_provider_to_all_leaves()
    
    def _apply_provider_to_all_leaves(self):
        """Apply the content provider to all existing leaves."""
        for leaf in self.root.get_all_leaves():
            leaf.content_provider = self.content_provider
            leaf._update_content()
    
    def split_pane(self, leaf: LeafNode, direction: SplitDirection):
        """Split a leaf pane in the specified direction."""
        if not isinstance(leaf, LeafNode):
            return
        
        print(f"Splitting pane. Current root type: {type(self.root).__name__}")
        
        # Create new leaf for the split with default type
        available_types = self.content_provider.get_available_pane_types() if self.content_provider else [self.default_pane_type]
        empty_type = available_types[0] if available_types else self.default_pane_type
        
        new_leaf = LeafNode(PaneConfig(pane_type=empty_type))
        new_leaf.content_provider = self.content_provider
        
        # Store the original parent before creating the split
        original_parent = leaf.parent
        
        # Create split node - don't pass parent yet to avoid circular reference
        split = SplitNode(direction, leaf, new_leaf, None)
        
        # Update tree structure
        if original_parent:
            # Replace leaf with split in parent
            if original_parent.left == leaf:
                original_parent.left = split
            else:
                original_parent.right = split
            # Set the parent for the split
            split.parent = original_parent
        else:
            # This was the root
            self.root = split
            split.parent = None
        
        print(f"After split. New root type: {type(self.root).__name__}")
        print(f"Total leaves in system: {len(self.get_all_leaves())}")
        
        # Store reference to the newly created split for space distribution
        self.newly_created_split = split
        
        # Re-render entire tree from root to avoid widget path issues
        self._render_entire_tree()
    
    def delete_pane(self, node: PaneNode):
        """Delete a pane node from the tree."""
        if not node.parent:
            # Can't delete root if it's a leaf
            if isinstance(node, LeafNode):
                print("Cannot delete root pane")
                return
        
        parent = node.parent
        
        # Get the sibling node
        sibling = parent.right if parent.left == node else parent.left
        
        # Replace parent with sibling
        if parent.parent:
            # Update grandparent to point to sibling
            grandparent = parent.parent
            if grandparent.left == parent:
                grandparent.left = sibling
            else:
                grandparent.right = sibling
            sibling.parent = grandparent
        else:
            # Parent was root, sibling becomes new root
            self.root = sibling
            sibling.parent = None
        
        # Re-render entire tree to avoid widget path issues
        self._render_entire_tree()
        
        print("Deleted pane")
    
    def _render_entire_tree(self):
        """Re-render the entire tree from scratch."""
        # Clear all widgets in our dedicated container
        try:
            for widget in self.parent.winfo_children():
                widget.destroy()
        except tk.TclError:
            pass
        
        # Wait for cleanup
        self.parent.update_idletasks()
        
        # Render from root
        if self.root:
            print(f"Rendering tree with root type: {type(self.root).__name__}")
            self.root.render(self.parent, self)
            
            # Only distribute space for the newly created split, if any
            if hasattr(self, 'newly_created_split') and self.newly_created_split:
                self.parent.after(10, lambda: self._distribute_new_split_space(self.newly_created_split))
                self.newly_created_split = None  # Clear the reference
    
    def _distribute_new_split_space(self, split_node: 'SplitNode'):
        """Distribute space equally only in the newly created split node."""
        if isinstance(split_node, SplitNode):
            split_node._distribute_space_equally()
    
    def _render_from_node(self, node: PaneNode):
        """Re-render the tree from the given node."""
        if not node:
            return
        
        # Find the top-level container for this node
        current = node
        while current.parent:
            current = current.parent
        
        # Clear the parent widget and re-render
        try:
            for widget in self.parent.winfo_children():
                widget.destroy()
        except tk.TclError:
            pass  # Ignore errors if widgets are already destroyed
        
        # Small delay to ensure cleanup is complete
        self.parent.update_idletasks()
        
        current.render(self.parent, self)
    
    def get_all_leaves(self) -> List[LeafNode]:
        """Get all leaf panes in the system."""
        return self.root.get_all_leaves() if self.root else []


# ============================
# DEMO IMPLEMENTATION
# ============================

class DemoPaneType(BasePaneType):
    """Demo implementation of pane types for testing."""
    
    def __init__(self, identifier: str, value: str):
        self._identifier = identifier
        self._value = value
    
    @property
    def value(self) -> str:
        return self._value
    
    @property
    def identifier(self) -> str:
        return self._identifier


# Demo pane type instances
class DemoPaneTypes:
    EMPTY = DemoPaneType("empty", "Empty")
    PLOT_VIEW = DemoPaneType("plot_view", "Plot View")
    DATA_TABLE = DemoPaneType("data_table", "Data Table")
    PARAMETER_CONTROLS = DemoPaneType("parameter_controls", "Parameter Controls")
    LOG_OUTPUT = DemoPaneType("log_output", "Log Output")
    FILE_BROWSER = DemoPaneType("file_browser", "File Browser")
    PROPERTIES = DemoPaneType("properties", "Properties")


class DemoContentProvider(PaneContentProvider):
    """Demo implementation of content provider."""
    
    def get_available_pane_types(self) -> List[BasePaneType]:
        """Get all available demo pane types."""
        return [
            DemoPaneTypes.EMPTY,
            DemoPaneTypes.PLOT_VIEW,
            DemoPaneTypes.DATA_TABLE,
            DemoPaneTypes.PARAMETER_CONTROLS,
            DemoPaneTypes.LOG_OUTPUT,
            DemoPaneTypes.FILE_BROWSER,
            DemoPaneTypes.PROPERTIES
        ]
    
    def create_content(self, parent_frame: tk.Frame, pane_type: BasePaneType) -> None:
        """Create content for the given pane type."""
        if pane_type == DemoPaneTypes.EMPTY:
            label = tk.Label(
                parent_frame, 
                text="Empty Pane\nSelect a type from dropdown above",
                bg='#3c3c3c', 
                fg='gray',
                font=('Arial', 10),
                justify=tk.CENTER
            )
            label.pack(expand=True)
            
        elif pane_type == DemoPaneTypes.PLOT_VIEW:
            # Create a mock matplotlib-like plot
            canvas_frame = tk.Frame(parent_frame, bg='white', relief='sunken', bd=2)
            canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Mock plot content
            plot_label = tk.Label(
                canvas_frame,
                text="📊 Interactive Plot\n\nThis would contain matplotlib\nwith error analysis plots",
                bg='white',
                fg='black',
                font=('Arial', 11),
                justify=tk.CENTER
            )
            plot_label.pack(expand=True)
            
        elif pane_type == DemoPaneTypes.DATA_TABLE:
            label = tk.Label(
                parent_frame,
                text="Data Table\n📋\nData table would go here",
                bg='#3c3c3c',
                fg='lightgreen',
                font=('Arial', 12),
                justify=tk.CENTER
            )
            label.pack(expand=True)
            
        elif pane_type == DemoPaneTypes.PARAMETER_CONTROLS:
            # Create mock parameter controls
            controls_frame = tk.Frame(parent_frame, bg='#3c3c3c')
            controls_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            tk.Label(controls_frame, text="Parameter Controls", bg='#3c3c3c', 
                    fg='white', font=('Arial', 12, 'bold')).pack(pady=5)
            
            # Mock sliders
            for i, param in enumerate(['Timestep', 'Tolerance', 'Order']):
                param_frame = tk.Frame(controls_frame, bg='#3c3c3c')
                param_frame.pack(fill=tk.X, padx=10, pady=5)
                
                tk.Label(param_frame, text=f"{param}:", bg='#3c3c3c', 
                        fg='white', width=10, anchor='w').pack(side=tk.LEFT)
                
                scale = tk.Scale(param_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                               bg='#3c3c3c', fg='white', highlightthickness=0)
                scale.set(50 - i*10)
                scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            
        elif pane_type == DemoPaneTypes.LOG_OUTPUT:
            # Create a scrollable text area for logs
            text_frame = tk.Frame(parent_frame, bg='#3c3c3c')
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            log_text = tk.Text(
                text_frame,
                bg='#1e1e1e',
                fg='#00ff00',
                font=('Consolas', 9),
                wrap=tk.WORD
            )
            scrollbar = tk.Scrollbar(text_frame, command=log_text.yview)
            log_text.configure(yscrollcommand=scrollbar.set)
            
            log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Add some sample log content
            log_text.insert(tk.END, "Log Output Window\n")
            log_text.insert(tk.END, "[INFO] System initialized\n")
            log_text.insert(tk.END, "[DEBUG] Window pane created\n")
            log_text.insert(tk.END, "[INFO] Ready for logging...\n")
            log_text.configure(state=tk.DISABLED)
            
        else:
            # Generic content for other types
            type_name = pane_type.value
            label = tk.Label(
                parent_frame,
                text=f"{type_name}\n\nContent for {type_name.lower()} would go here",
                bg='#3c3c3c',
                fg='white',
                font=('Arial', 10),
                justify=tk.CENTER
            )
            label.pack(expand=True)


def create_demo_window():
    """Create a demo window to test the generalized window splitter."""
    root = tk.Tk()
    root.title("Generalized Window Splitter - Blender Style")
    root.geometry("1200x800")
    root.configure(bg='#2b2b2b')
    
    # Instructions label at the bottom
    info_frame = tk.Frame(root, bg='#404040', height=50)
    info_frame.pack(side=tk.BOTTOM, fill=tk.X)
    info_frame.pack_propagate(False)
    
    info_text = ("Generalized Window Splitter: Use ⬌/⬍ buttons to split panes. "
                "Drag splitters to resize. Make panes small to delete them (they'll gray out). "
                "Click ✕ to delete directly. Change pane types via dropdown.")
    tk.Label(info_frame, text=info_text, bg='#404040', fg='white', 
            font=('Arial', 9), wraplength=1100, justify=tk.LEFT).pack(pady=10, padx=10)
    
    # Create a main container for the splitter (this will be the parent)
    main_container = tk.Frame(root, bg='#2b2b2b')
    main_container.pack(fill=tk.BOTH, expand=True)
    
    # Create the splitter manager with the main container as parent
    splitter_manager = WindowSplitterManager(main_container, DemoPaneTypes.PLOT_VIEW)
    
    # Set up the demo content provider
    content_provider = DemoContentProvider()
    splitter_manager.set_content_provider(content_provider)
    
    return root, splitter_manager


if __name__ == "__main__":
    # Run the demo
    root, splitter_manager = create_demo_window()
    root.mainloop()
