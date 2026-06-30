# Contribution guidelines for aigrids

## Table of Contents
- [Getting Started](#-getting-started)
- [Keeping Your Branch Up to Date](#-keeping-your-branch-up-to-date)
- [Working with Feature Branches (Optional)](#-working-with-feature-branches-optional)
- [Important Note on Rebasing](#️-important-note-on-rebasing)
- [Pull Request Naming](#️-pull-request-naming)
- [Code Review Etiquette](#️-code-review-etiquette)
- [Collaboration Tips](#️-collaboration-tips)


We use a single long-lived branch for this project:

```cpp
----------------------------- main (default branch) -----------------------------
```

All feature and personal development branches should originate from `main`.


## Getting Started
To begin contributing:
1. Clone the repository
```bash
git clone https://github.com/aigrids/aigrids

# or use your personal access token 
git clone https://<your_personal_access_token>@github.com/aigrids/aigrids

cd aigrids
```

2. Create and switch to your personal development branch for making changes.
```bash
git checkout -b <your_personal_branch>
```

```lua
-------------------------------------main-------------------------------------
            ----------------<your_personal_branch>----------------
```

3. Install cloned package in virtual enviornment:
```bash
python -m venv .venv
source .venv/bin/activate 
pip install -e ".[dev]"
```

4. Run tests
```bash
ruff check .
pytest
pytest --cov=aigrids
```


## Keeping Your Branch Up to Date

If multiple contributors are working simultaneously, it’s important to sync your 
branch with the latest changes from `main` before opening a pull request:

1. Ensure you're on your branch:
```bash
git checkout <your_personal_branch>
```

2. Fetch the latest changes and update `main`:
```bash
git fetch origin
git checkout main
git pull origin main
```

3. Merge `main` into your branch:
```bash
git checkout <your_personal_branch>
git merge main
```
This helps reduce merge conflicts during pull requests and ensures your work is 
based on the latest codebase. Resolve any conflicts that occur.

4. Push your updated branch to GitHub:
```bash
git push -u origin <your_personal_branch>
```

5. Open a pull request from `<your_personal_branch>` into `main` on GitHub.


## Working with Feature Branches (Optional)
You may create short-lived feature branches from your personal branch to isolate 
work on specific tasks:
```lua
-------------------------------------main-------------------------------------
            ----------------<your_personal_branch>----------------
                     -------<your_feature_branch>-------
```


1. Create a feature branch from your personal branch:
```bash
git checkout -b <your_feature_branch> <your_personal_branch>
```

2. When complete, test and merge (or rebase) your feature branch back into your 
personal branch:
```bash
git checkout <your_personal_branch>
git merge <your_feature_branch>
```

3. Delete the feature branch if it is no longer needed.
```bash
git branch -d <your_feature_branch>
```


## Important Note on Rebasing
* Allowed: Use git rebase locally to tidy up your commits before merging into a 
shared branch.
* Avoid: Rebasing commits that have already been pushed to the remote repository. 
This can cause issues for other collaborators.


## Pull Request Naming
Use concise and descriptive titles, e.g.:
- `feat: add transformer encoder`
- `fix: correct power calculation bug`
- `refactor: simplify model initialization`


## Code Review Etiquette
- Keep pull requests small and focused
- Add meaningful descriptions
- Tag reviewers (`@username`)
- Respond to feedback promptly


## Collaboration Tips
- Communicate often—don’t hesitate to ask for feedback early.
- Prefer clarity over cleverness: readable code helps everyone.