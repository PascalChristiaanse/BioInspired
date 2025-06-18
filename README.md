# BioInspired: Evolutionary Docking System

A machine learning project that uses evolutionary algorithms to train neural networks for automated molecular docking procedures. The system supports progressive environment complexity, comprehensive trajectory storage, and advanced analysis capabilities.

## 🧬 Features

- **Evolutionary Algorithms**: Train neural networks using bio-inspired optimization
- **Progressive Environments**: Start simple and increase complexity for better learning
- **Trajectory Database**: Store millions of trajectories with full reproducibility
- **Multi-processing Support**: Concurrent simulations with safe database access
- **Visualization Tools**: Plot results with matplotlib and render in Blender
- **3D Rendering**: High-quality trajectory visualization in Blender
- **Species Tracking**: Identify and analyze evolutionary species
- **Comprehensive Analysis**: Performance metrics, generation statistics, and more

## 🏗️ Architecture

The project is structured for modularity and reusability:

```
src/bioinspired/
├── algorithm/          # Evolutionary algorithms
├── data/              # Database models and management
├── docking/           # Docking simulation logic
└── visualization/     # Plotting and rendering tools
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Git

### Installation

1. **Clone the repository**:
   ```powershell
   git clone <your-repo-url>
   cd BioInspired
   ```

2. **Run the setup script** (creates virtual environment and sets up everything):
   ```powershell
   python setup.py
   ```

   This will:
   - Create a virtual environment in `venv/`
   - Install Python dependencies in the virtual environment
   - Start PostgreSQL database
   - Initialize database tables
   - Run example usage

3. **Activate the virtual environment** (for future work):
   ```powershell
   # Windows PowerShell
   .\activate.ps1
   
   # Or manually
   .\venv\Scripts\Activate.ps1
   
   # Unix/Linux/Mac
   source activate.sh
   # Or manually
   source venv/bin/activate
   ```

### Manual Setup

If you prefer manual setup:

1. **Create and activate virtual environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows
   # source venv/bin/activate   # Unix/Linux/Mac
   ```

2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Start the database**:
   ```powershell
   docker-compose up -d
   ```

4. **Initialize database tables**:
   ```powershell
   python init_db.py
   ```

5. **Run the example**:
   ```powershell
   python example_usage.py
   ```

## 📊 Database Schema

The system uses PostgreSQL with the following main entities:

- **Environment**: Simulation environments with increasing complexity
- **Algorithm**: Algorithm runs with hyperparameters and settings
- **Individual**: Candidate solutions with fitness scores
- **Trajectory**: Large numerical trajectory data (stored as files)
- **Annotation**: Analysis results and metadata

## 🔬 Usage Examples

### Basic Usage

```python
from bioinspired.data import DatabaseManager
import numpy as np

# Initialize database manager
db = DatabaseManager()

# Create an environment
env = db.create_environment(
    name="Basic Docking Environment",
    description="Simple environment for initial training",
    parameters={"complexity_level": 1}
)

# Create algorithm run
algorithm = db.create_algorithm(
    environment_id=env.id,
    population_id="experiment_001",
    seed=42,
    hyperparameters={
        "population_size": 100,
        "num_generations": 50
    }
)

# Create individual with trajectory
individual = db.create_individual(
    algorithm_id=algorithm.id,
    generation=0,
    fitness=0.85,
    species="species_A"
)

# Save trajectory data
trajectory_data = np.random.rand(1000, 3)  # 1000 steps, 3D
trajectory = db.save_trajectory_data(
    individual_id=individual.id,
    trajectory_data=trajectory_data
)
```

### Querying and Analysis

```python
# Get best individuals
best = db.get_best_individuals(algorithm.id, limit=10)

# Get generation statistics
stats = db.get_generation_stats(algorithm.id)

# Load trajectory data
trajectory_obj, data = db.load_trajectory_data(trajectory.id)
```

## 🔧 Configuration

### Database Configuration

Edit `.env` file:
```env
DATABASE_URL=postgresql://bioinspired_user:bioinspired_pass@localhost:5432/bioinspired
DEBUG=True
LOG_LEVEL=INFO
```

### Docker Configuration

The `docker-compose.yml` sets up PostgreSQL with persistent storage. Modify ports or credentials as needed.

## 🎨 **Blender Rendering Package Structure**

The project now includes a separate `blender_bio` package for 3D visualization:

```
src/
├── bioinspired/          # Core evolutionary algorithm package
│   ├── algorithm/        # Evolutionary algorithms
│   ├── data/            # Database and trajectory management
│   ├── docking/         # Docking simulation
│   └── visualization/   # Basic matplotlib plotting
└── blender_bio/         # Advanced 3D rendering with Blender
    ├── core/            # Scene setup and materials
    ├── trajectory/      # Trajectory rendering and animation
    ├── molecules/       # Molecular visualization (future)
    ├── exporters/       # Video/image export utilities (future)
    └── standalone/      # Standalone rendering scripts
```

### **Blender Integration**

The Blender package can be used in two ways:

1. **From Python** (subprocess): Use `blender_rendering_example.py`
2. **Within Blender**: Run scripts directly in Blender's Python console

### **Usage Examples**

```python
# Render a trajectory from outside Blender
from blender_rendering_example import render_trajectory_with_blender
render_trajectory_with_blender(trajectory_id=1)

# Or use the interactive script
python blender_rendering_example.py
```

```bash
# Render from command line with Blender
blender --background --python src/blender_bio/standalone/render_trajectory.py -- --trajectory_id 1 --output_path ./renders/
```

## 🧪 Development

### Adding New Algorithms

1. Create algorithm modules in `src/bioinspired/algorithm/`
2. Use the `DatabaseManager` to store results
3. Follow the existing patterns for reproducibility

### Adding Visualization

1. Create visualization modules in `src/bioinspired/visualization/`
2. Use matplotlib for plotting, Blender for 3D rendering
3. Query data using the database interface

### Testing

```powershell
# Run tests (once you add them)
python -m pytest tests/

# Check database connectivity
python init_db.py
```

## 📈 Performance Considerations

- **Large Datasets**: Trajectory data is stored in compressed NumPy files
- **Concurrent Access**: PostgreSQL handles multiple processes safely
- **Indexing**: Database tables are optimized for common queries
- **Memory Usage**: Only load trajectory data when needed

## 🤝 Contributing

This project uses a custom license that allows modification but not incorporation into other programs, and requires attribution. See `LICENSE` for details.

## 📝 License

Custom Non-Commercial, No-Derivative License with Attribution.
- ✅ Modification allowed
- ❌ Commercial use prohibited  
- ❌ Incorporation into other programs prohibited
- ✅ Attribution required

## 🆘 Troubleshooting

### Database Connection Issues
```powershell
# Check if database is running
docker-compose ps

# Restart database
docker-compose restart

# Check logs
docker-compose logs postgres
```

### Python Import Issues
```powershell
# Install in development mode
pip install -e .
```

### Large Data Files
- Trajectory files are automatically excluded from git
- Use `.gitignore` patterns to exclude large datasets
- Consider using Git LFS for sharing large files