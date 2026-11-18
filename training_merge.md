# Training merge

When merging the training branch into `train-measure`, take these files from
the training branch:

```text
language/params.yaml
language/dvc.lock
outputs/language/loss.yaml
outputs/language/lr.yaml
outputs/language/accuracy.yaml
```

Run:

```bash
git fetch origin
git switch train-measure
git pull --ff-only origin train-measure

git merge origin/language-first-config

# Resolve the listed files using the training branch version:
git checkout origin/language-first-config -- \
  language/params.yaml \
  language/dvc.lock \
  outputs/language/loss.yaml \
  outputs/language/lr.yaml \
  outputs/language/accuracy.yaml

git add \
  language/params.yaml \
  language/dvc.lock \
  outputs/language/loss.yaml \
  outputs/language/lr.yaml \
  outputs/language/accuracy.yaml

git status
git commit
git push origin train-measure
```

If the merge is already in progress, omit the `git merge` command and run the
checkout, add, commit, and push commands.
