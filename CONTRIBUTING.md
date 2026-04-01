# Contributing to Nocle

First off, thank you for considering contributing to Nocle! It's people like you that make Nocle such a great tool. 

## Where do I go from here?

If you've noticed a bug or have a question, you can [search our issues](https://github.com/haydarkadioglu/nocle-app/issues) to see if someone else in the community has already addressed it. If not, go ahead and create an issue!

## How to Contribute

We welcome contributions from everyone. Here is the standard workflow for contributing to the repository:

### 1. Fork & Clone

1. **Fork** the repository on GitHub by clicking the "Fork" button in the upper right corner of the repo.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/nocle-app.git
   cd nocle-app
   ```

### 2. Set Up Development Environment

Please follow the installation instructions in the `README.md` to set up your Python virtual environment and install the required dependencies:

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Create a Branch

Create a new branch for the feature or bug you're working on:
```bash
git checkout -b feature/your-awesome-feature
# or
git checkout -b bugfix/issue-number
```

### 4. Code & Test

- Make your changes in the codebase.
- The UI is built with **Flet** and audio processing uses **Librosa** & **Tensorflow**.
- Please make sure the app still starts and processes audio without errors (`python main.py`).

### 5. Commit Your Changes

Write clear, concise commit messages. 
```bash
git commit -m "feat: added a new high-pass audio filter"
```

### 6. Push and Open a Pull Request

1. Push your branch to your forked repository:
   ```bash
   git push origin feature/your-awesome-feature
   ```
2. Go to the original Nocle repository and click **Compare & pull request**.
3. Fill out the PR template/description clearly explaining what changes you made and why.

## Code Style

- Please follow standard Python PEP-8 conventions.
- Keep the UI (Flet) components modular if you are adding new heavy features.
- Any new filter added to `filters.py` should ideally be purely mathematical (`numpy`/`scipy`/`librosa`) to keep performance optimal.

## Need Help?
Feel free to open a discussion or reach out to the project maintainer via the contact info in the `README.md`.

Thank you for your time and contribution!
